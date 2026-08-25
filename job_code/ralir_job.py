import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_file(root, filename, required=True):
    root = Path(root)

    direct = root / filename
    if direct.is_file():
        return direct

    matches = sorted(
        root.rglob(filename),
        key=lambda path: (len(path.parts), str(path)),
    )

    if matches:
        return matches[0]

    if required:
        raise FileNotFoundError(f"Cannot find {filename} under {root}")

    return None


def find_model_dir(root):
    root = Path(root)

    candidates = []

    if (root / "config.json").is_file():
        candidates.append(root)

    candidates.extend(
        path.parent for path in root.rglob("config.json")
    )

    if not candidates:
        raise FileNotFoundError(f"No config.json found under {root}")

    candidates = sorted(
        set(candidates),
        key=lambda path: (len(path.parts), str(path)),
    )

    return candidates[0]


def recursive_size(path):
    return sum(
        file.stat().st_size
        for file in Path(path).rglob("*")
        if file.is_file()
    )


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def verify_checksum_manifest(manifest):
    manifest = Path(manifest)
    results = []

    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Invalid checksum line: {line}")

            expected = fields[0]
            relative = fields[-1].lstrip("*")
            target = manifest.parent / relative

            if not target.is_file():
                matches = list(manifest.parent.rglob(Path(relative).name))
                if len(matches) == 1:
                    target = matches[0]
                else:
                    raise FileNotFoundError(
                        f"Checksum target not found: {relative}"
                    )

            actual = sha256_file(target)
            if actual != expected:
                raise ValueError(
                    f"Checksum mismatch for {target}: "
                    f"{actual} != {expected}"
                )

            results.append(
                {
                    "file": str(target),
                    "sha256": actual,
                    "valid": True,
                }
            )

    return results


def validate_gzip(path):
    with gzip.open(path, "rb") as handle:
        while handle.read(8 * 1024 * 1024):
            pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_answer(text):
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def token_f1(prediction, answer):
    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()

    if not prediction_tokens or not answer_tokens:
        return float(prediction_tokens == answer_tokens)

    common = Counter(prediction_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(answer_tokens)

    return 2.0 * precision * recall / (precision + recall)


def answer_f1(prediction, answers):
    return max(
        (token_f1(prediction, answer) for answer in answers),
        default=0.0,
    )


def answer_em(prediction, answers):
    normalized = normalize_answer(prediction)
    return float(
        any(normalized == normalize_answer(answer) for answer in answers)
    )


def passage_contains_answer(text, answers):
    normalized_text = f" {normalize_answer(text)} "

    for answer in answers:
        normalized_answer = normalize_answer(answer)
        if normalized_answer and f" {normalized_answer} " in normalized_text:
            return True

    return False


def select_rows(rows, limit, seed):
    rows = list(rows)

    if not limit or len(rows) <= limit:
        return rows

    indices = sorted(
        random.Random(seed).sample(range(len(rows)), limit)
    )

    return [rows[index] for index in indices]


def load_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def context_passage_ids(record):
    passage_ids = []

    for context in record.get("positive_ctxs", []):
        value = context.get("passage_id", context.get("id"))
        if value is not None:
            passage_ids.append(str(value).strip())

    return passage_ids


def parse_qa_file(path, dataset):
    path = Path(path)

    opener = gzip.open if path.suffix == ".gz" else open

    rows = []

    with opener(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.rstrip("\r\n")
            if not line:
                continue

            fields = line.split("\t", 1)
            if len(fields) != 2:
                fields = next(csv.reader([line]))

            if len(fields) < 2:
                continue

            question = fields[0].strip()
            encoded_answers = fields[1].strip()

            try:
                answers = ast.literal_eval(encoded_answers)
            except Exception:
                try:
                    answers = json.loads(encoded_answers)
                except Exception:
                    answers = [encoded_answers]

            if isinstance(answers, str):
                answers = [answers]

            answers = [
                str(answer).strip()
                for answer in answers
                if str(answer).strip()
            ]

            rows.append(
                {
                    "qid": f"{dataset}-{index}",
                    "dataset": dataset,
                    "question": question,
                    "answers": answers,
                    "positive_pids": [],
                }
            )

    return rows


class PassageStore:
    def __init__(self, prepared):
        prepared = Path(prepared)
        self.collection_path = prepared / "collection.tsv"
        self.offsets_path = prepared / "collection.offsets.u64"

        self.offsets = np.memmap(
            self.offsets_path,
            dtype=np.uint64,
            mode="r",
        )
        self.handle = self.collection_path.open("rb")

    def get(self, pid):
        pid = int(pid)

        if pid < 0 or pid >= len(self.offsets):
            return ""

        self.handle.seek(int(self.offsets[pid]))
        line = self.handle.readline().decode("utf-8", errors="replace")
        fields = line.rstrip("\r\n").split("\t", 1)

        return fields[1] if len(fields) == 2 else ""

    def close(self):
        self.handle.close()


def iter_collection(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue

            fields = line.split("\t", 1)
            if len(fields) != 2:
                continue

            yield int(fields[0]), fields[1]


def load_eval_records(prepared, max_queries, seed):
    records = []

    for path in sorted(Path(prepared).glob("eval_*.jsonl")):
        dataset_rows = list(iter_jsonl(path))
        dataset_rows = select_rows(dataset_rows, max_queries, seed)
        records.extend(dataset_rows)

    return records


def render_chat_prompt(tokenizer, system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return (
            f"System: {system}\n"
            f"User: {user}\n"
            "Assistant:"
        )


def load_causal_lm(model_root):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = find_model_dir(model_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_texts(
    model,
    tokenizer,
    device,
    prompts,
    batch_size,
    max_new_tokens,
    max_input_tokens=1536,
):
    predictions = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        input_length = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        continuations = generated[:, input_length:]

        predictions.extend(
            text.strip()
            for text in tokenizer.batch_decode(
                continuations,
                skip_special_tokens=True,
            )
        )

    return predictions


def command_validate(args):
    output = ensure_dir(args.output)

    report = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "raw_files": {},
        "models": {},
        "checksums": [],
        "warnings": [],
        "errors": [],
    }

    required_raw_files = [
        "psgs_w100.tsv.gz",
        "biencoder-nq-train.json.gz",
        "biencoder-nq-dev.json.gz",
        "biencoder-nq-adv-hn-train.json.gz",
        "nq-test.qa.csv",
        "nq-test_gold_info.json.gz",
        "trivia-test.qa.csv.gz",
        "LICENSE",
        "manifest.sha256",
    ]

    for filename in required_raw_files:
        try:
            path = find_file(args.raw, filename)
            report["raw_files"][filename] = {
                "path": str(path),
                "bytes": path.stat().st_size,
            }

            if path.suffix == ".gz":
                validate_gzip(path)

        except Exception as error:
            report["errors"].append(f"{filename}: {error}")

    manifest = find_file(args.raw, "manifest.sha256", required=False)

    if manifest:
        try:
            report["checksums"].extend(
                verify_checksum_manifest(manifest)
            )
        except Exception as error:
            report["errors"].append(f"Raw checksum validation: {error}")

    from transformers import AutoConfig

    model_inputs = {
        "colbert": args.colbert,
        "qwen": args.qwen,
        "phi": args.phi,
        "e5": args.e5,
    }

    for name, root in model_inputs.items():
        try:
            model_dir = find_model_dir(root)
            config = AutoConfig.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True,
            )

            report["models"][name] = {
                "path": str(model_dir),
                "model_type": getattr(config, "model_type", None),
                "bytes": recursive_size(model_dir),
            }

            checksum_file = find_file(
                root,
                "SHA256SUMS",
                required=False,
            )

            if checksum_file:
                report["checksums"].extend(
                    verify_checksum_manifest(checksum_file)
                )
            else:
                report["warnings"].append(
                    f"No SHA256SUMS found for {name}"
                )

        except Exception as error:
            report["errors"].append(f"Model {name}: {error}")

    try:
        import colbert
        report["colbert_import"] = True
        report["colbert_module"] = str(Path(colbert.__file__).resolve())
    except Exception as error:
        report["errors"].append(f"ColBERT import: {error}")

    try:
        import faiss
        report["faiss_import"] = True
        report["faiss_version"] = getattr(faiss, "__version__", None)
    except Exception as error:
        report["errors"].append(f"FAISS import: {error}")

    provenance_files = [
        str(path.relative_to(args.provenance))
        for path in Path(args.provenance).rglob("*")
        if path.is_file()
    ]
    report["provenance_files"] = provenance_files

    if not provenance_files:
        report["errors"].append("Provenance asset is empty")

    if not report["cuda_available"]:
        report["errors"].append("CUDA is unavailable on the job compute")

    report["valid"] = not report["errors"]

    write_json(output / "validation.json", report)
    write_json(output / "metrics.json", report)

    if report["errors"]:
        raise RuntimeError(
            "Validation failed:\n" + "\n".join(report["errors"])
        )


def command_prepare(args):
    set_seed(args.seed)

    raw = Path(args.raw)
    output = ensure_dir(args.output)

    train_source = load_gzip_json(
        find_file(raw, "biencoder-nq-train.json.gz")
    )
    train_source = select_rows(
        train_source,
        args.pilot_train_queries,
        args.seed,
    )

    needed_original_ids = set()

    for record in train_source:
        needed_original_ids.update(context_passage_ids(record))

    passages_path = find_file(raw, "psgs_w100.tsv.gz")
    collection_path = output / "collection.tsv"
    offsets_path = output / "collection.offsets.u64"

    original_to_pid = {}
    collection_count = 0
    offset_buffer = array("Q")

    with gzip.open(
        passages_path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as source, collection_path.open("wb") as collection, \
            offsets_path.open("wb") as offsets:

        reader = csv.reader(source, delimiter="\t")
        header = next(reader)

        header_lookup = {
            name.strip().lower(): index
            for index, name in enumerate(header)
        }

        id_index = header_lookup.get("id", 0)
        text_index = header_lookup.get("text", 1)
        title_index = header_lookup.get("title", 2)

        for row_number, row in enumerate(reader):
            if len(row) <= max(id_index, text_index, title_index):
                continue

            original_id = str(row[id_index]).strip()

            include = (
                args.pilot_passages == 0
                or row_number < args.pilot_passages
                or original_id in needed_original_ids
            )

            if not include:
                continue

            text = row[text_index].replace("\t", " ")
            text = text.replace("\r", " ").replace("\n", " ")

            title = row[title_index].replace("\t", " ")
            title = title.replace("\r", " ").replace("\n", " ")

            passage = f"{title} | {text}" if title else text

            offset_buffer.append(collection.tell())
            encoded = (
                f"{collection_count}\t{passage}\n"
            ).encode("utf-8")
            collection.write(encoded)

            if original_id in needed_original_ids:
                original_to_pid[original_id] = collection_count

            collection_count += 1

            if len(offset_buffer) >= 1_000_000:
                offset_buffer.tofile(offsets)
                offset_buffer = array("Q")

        if offset_buffer:
            offset_buffer.tofile(offsets)

    train_rows = []

    for index, source_record in enumerate(train_source):
        positive_pids = [
            original_to_pid[original_id]
            for original_id in context_passage_ids(source_record)
            if original_id in original_to_pid
        ]

        positive_pids = list(dict.fromkeys(positive_pids))

        if not positive_pids:
            continue

        train_rows.append(
            {
                "qid": f"nq-train-{index}",
                "dataset": "nq-train",
                "question": source_record["question"],
                "answers": source_record.get("answers", []),
                "positive_pids": positive_pids,
            }
        )

    nq_test = parse_qa_file(
        find_file(raw, "nq-test.qa.csv"),
        "nq-test",
    )
    trivia_test = parse_qa_file(
        find_file(raw, "trivia-test.qa.csv.gz"),
        "trivia-test",
    )

    nq_test = select_rows(
        nq_test,
        args.pilot_eval_queries,
        args.seed + 1,
    )
    trivia_test = select_rows(
        trivia_test,
        args.pilot_eval_queries,
        args.seed + 2,
    )

    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "eval_nq.jsonl", nq_test)
    write_jsonl(output / "eval_trivia.jsonl", trivia_test)

    metadata = {
        "mode": args.mode,
        "collection_count": collection_count,
        "train_queries": len(train_rows),
        "nq_eval_queries": len(nq_test),
        "trivia_eval_queries": len(trivia_test),
        "seed": args.seed,
    }

    write_json(output / "manifest.json", metadata)
    write_json(output / "metrics.json", metadata)

    print(json.dumps(metadata, indent=2))


def command_index(args):
    from colbert import Indexer
    from colbert.infra import ColBERTConfig, Run, RunConfig

    prepared = Path(args.prepared)
    model_dir = find_model_dir(args.model)
    output = ensure_dir(args.output)
    metrics_dir = ensure_dir(args.metrics)

    collection = prepared / "collection.tsv"

    started = time.time()

    config = ColBERTConfig(
        nbits=args.nbits,
        doc_maxlen=180,
        index_bsize=32,
        kmeans_niters=4,
        index_path=str(output),
    )

    with Run().context(
        RunConfig(
            nranks=1,
            experiment="ralir",
            root=str(output.parent),
            gpus=1 if torch.cuda.is_available() else 0,
            avoid_fork_if_possible=True,
        )
    ):
        indexer = Indexer(
            checkpoint=str(model_dir),
            config=config,
        )

        built_path = Path(
            indexer.index(
                name=f"colbert-nbits{args.nbits}",
                collection=str(collection),
                overwrite="force_silent_overwrite",
            )
        )

    if built_path.resolve() != output.resolve():
        shutil.copytree(built_path, output, dirs_exist_ok=True)

    elapsed = time.time() - started

    receipt = {
        "nbits": args.nbits,
        "index_path": str(output),
        "collection": str(collection),
        "documents": sum(1 for _ in iter_collection(collection)),
        "index_bytes": recursive_size(output),
        "elapsed_seconds": elapsed,
    }

    write_json(output / "index_receipt.json", receipt)
    write_json(metrics_dir / "metrics.json", receipt)


def load_searcher(index, model, prepared):
    from colbert import Searcher

    model_dir = find_model_dir(model)
    collection = Path(prepared) / "collection.tsv"

    return Searcher(
        index=str(Path(index).resolve()),
        checkpoint=str(model_dir),
        collection=str(collection),
        verbose=1,
    )


def force_positive_candidates(retrieved, positives, k):
    selected = list(dict.fromkeys(int(pid) for pid in retrieved[:k]))
    positives = list(dict.fromkeys(int(pid) for pid in positives))
    protected = set(positives)

    for positive in positives:
        if positive in selected:
            continue

        if len(selected) < k:
            selected.append(positive)
            continue

        replacement = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if selected[index] not in protected
            ),
            None,
        )

        if replacement is not None:
            selected[replacement] = positive

    return list(dict.fromkeys(selected))[:k]


def command_candidates(args):
    set_seed(args.seed)

    candidates_dir = ensure_dir(args.candidates)
    cache_dir = ensure_dir(args.cache)

    train_rows = list(
        iter_jsonl(Path(args.prepared) / "train.jsonl")
    )
    train_rows = select_rows(
        train_rows,
        args.max_queries,
        args.seed,
    )

    searcher = load_searcher(
        args.index,
        args.model,
        args.prepared,
    )
    store = PassageStore(args.prepared)

    candidate_path = candidates_dir / "candidates.jsonl"
    shard = []
    shard_index = 0
    total_candidates = 0

    started = time.time()

    with candidate_path.open("w", encoding="utf-8") as candidate_file:
        for query_index, record in enumerate(train_rows):
            retrieved, _, retrieval_scores = searcher.search(
                record["question"],
                k=args.k,
            )

            score_lookup = {
                int(pid): float(score)
                for pid, score in zip(retrieved, retrieval_scores)
            }

            candidate_pids = force_positive_candidates(
                retrieved,
                record.get("positive_pids", []),
                args.k,
            )

            if not candidate_pids:
                continue

            pid_tensor = torch.tensor(
                candidate_pids,
                dtype=torch.int32,
                device=(
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                ),
            )

            document_embeddings, document_lengths = (
                searcher.ranker.lookup_pids(pid_tensor)
            )

            document_embeddings = (
                document_embeddings.detach().cpu().half()
            )
            document_lengths = (
                document_lengths.detach().cpu().tolist()
            )

            documents = []
            offset = 0

            for length in document_lengths:
                length = int(length)
                documents.append(
                    document_embeddings[offset : offset + length].clone()
                )
                offset += length

            query_embedding = (
                searcher.encode(record["question"])
                .squeeze(0)
                .detach()
                .cpu()
                .half()
            )

            candidate_row = {
                "qid": record["qid"],
                "dataset": record["dataset"],
                "question": record["question"],
                "answers": record["answers"],
                "positive_pids": record.get("positive_pids", []),
                "candidates": [
                    {
                        "pid": pid,
                        "rank": rank + 1,
                        "retrieval_score": score_lookup.get(pid),
                        "text": store.get(pid),
                    }
                    for rank, pid in enumerate(candidate_pids)
                ],
            }

            candidate_file.write(
                json.dumps(candidate_row, ensure_ascii=False) + "\n"
            )

            shard.append(
                {
                    "qid": record["qid"],
                    "question": record["question"],
                    "answers": record["answers"],
                    "positive_pids": record.get("positive_pids", []),
                    "candidate_pids": candidate_pids,
                    "query": query_embedding,
                    "documents": documents,
                }
            )

            total_candidates += len(candidate_pids)

            if len(shard) >= 50:
                torch.save(
                    shard,
                    cache_dir / f"cache-{shard_index:05d}.pt",
                )
                shard = []
                shard_index += 1

            print(
                f"Candidates {query_index + 1}/{len(train_rows)}",
                flush=True,
            )

    if shard:
        torch.save(
            shard,
            cache_dir / f"cache-{shard_index:05d}.pt",
        )

    store.close()

    metrics = {
        "queries": len(train_rows),
        "candidates": total_candidates,
        "cache_shards": len(list(cache_dir.glob("cache-*.pt"))),
        "cache_bytes": recursive_size(cache_dir),
        "elapsed_seconds": time.time() - started,
    }

    write_json(candidates_dir / "metrics.json", metrics)
    write_json(cache_dir / "metrics.json", metrics)


def command_utility(args):
    output = ensure_dir(args.output)

    candidate_file = find_file(
        args.candidates,
        "candidates.jsonl",
    )

    rows = list(iter_jsonl(candidate_file))
    rows = rows[: args.max_queries] if args.max_queries else rows

    model, tokenizer, device = load_causal_lm(args.model)

    system = (
        "Answer factual questions using the supplied evidence. "
        "Return only a short answer, without explanation."
    )

    output_path = output / "utility_labels.jsonl"
    started = time.time()

    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            question = row["question"]
            answers = row["answers"]

            baseline_prompt = render_chat_prompt(
                tokenizer,
                system,
                f"Question: {question}\nShort answer:",
            )

            baseline_answer = generate_texts(
                model,
                tokenizer,
                device,
                [baseline_prompt],
                batch_size=1,
                max_new_tokens=args.max_new_tokens,
            )[0]

            baseline_f1 = answer_f1(baseline_answer, answers)

            context_prompts = []

            for candidate in row["candidates"]:
                evidence = candidate["text"][:7000]
                user = (
                    f"Evidence:\n{evidence}\n\n"
                    f"Question: {question}\n"
                    "Short answer:"
                )
                context_prompts.append(
                    render_chat_prompt(tokenizer, system, user)
                )

            context_answers = generate_texts(
                model,
                tokenizer,
                device,
                context_prompts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )

            utilities = []

            for candidate, prediction in zip(
                row["candidates"],
                context_answers,
            ):
                contextual_f1 = answer_f1(prediction, answers)

                utilities.append(
                    {
                        "pid": int(candidate["pid"]),
                        "prediction": prediction,
                        "context_f1": contextual_f1,
                        "no_context_f1": baseline_f1,
                        "utility": contextual_f1 - baseline_f1,
                    }
                )

            result = {
                "qid": row["qid"],
                "question": question,
                "answers": answers,
                "baseline_prediction": baseline_answer,
                "baseline_f1": baseline_f1,
                "utilities": utilities,
            }

            handle.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

            print(
                f"Utility {index + 1}/{len(rows)}",
                flush=True,
            )

    metrics = {
        "queries": len(rows),
        "labels": sum(
            len(row["candidates"])
            for row in rows
        ),
        "elapsed_seconds": time.time() - started,
    }

    write_json(output / "metrics.json", metrics)


class LowRankQueryAdapter(torch.nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()

        self.dimension = dimension
        self.rank = rank

        self.down = torch.nn.Linear(
            dimension,
            rank,
            bias=False,
        )
        self.up = torch.nn.Linear(
            rank,
            dimension,
            bias=False,
        )

        torch.nn.init.normal_(self.down.weight, std=0.02)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, query):
        adapted = query + self.up(self.down(query))
        return F.normalize(adapted, p=2, dim=-1)


def load_utility_labels(path):
    label_file = find_file(path, "utility_labels.jsonl")
    labels = {}

    for row in iter_jsonl(label_file):
        labels[row["qid"]] = {
            int(item["pid"]): float(item["utility"])
            for item in row["utilities"]
        }

    return labels


def maxsim_scores(query, documents, device):
    scores = []

    for document in documents:
        document = document.to(device=device, dtype=torch.float32)
        similarities = document @ query.transpose(0, 1)
        score = similarities.max(dim=0).values.sum()
        scores.append(score)

    return torch.stack(scores)


def command_train(args):
    set_seed(args.seed)

    output = ensure_dir(args.output)
    metrics_dir = ensure_dir(args.metrics)

    cache_files = sorted(Path(args.cache).rglob("cache-*.pt"))
    if not cache_files:
        raise FileNotFoundError("No candidate cache shards found")

    labels = load_utility_labels(args.labels)

    first_records = torch.load(
        cache_files[0],
        map_location="cpu",
        weights_only=False,
    )

    dimension = int(first_records[0]["query"].shape[-1])
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    adapter = LowRankQueryAdapter(
        dimension=dimension,
        rank=args.rank,
    ).to(device)

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    history = []
    started = time.time()

    for epoch in range(args.epochs):
        shuffled_files = list(cache_files)
        random.Random(args.seed + epoch).shuffle(shuffled_files)

        epoch_loss = 0.0
        epoch_utility = 0.0
        epoch_nq = 0.0
        epoch_anchor = 0.0
        updates = 0

        for cache_file in shuffled_files:
            records = torch.load(
                cache_file,
                map_location="cpu",
                weights_only=False,
            )
            random.Random(args.seed + epoch).shuffle(records)

            for record in records:
                candidate_pids = [
                    int(pid) for pid in record["candidate_pids"]
                ]
                documents = record["documents"]

                if not documents:
                    continue

                base_query = record["query"].to(
                    device=device,
                    dtype=torch.float32,
                )

                adapted_query = adapter(base_query)
                scores = maxsim_scores(
                    adapted_query,
                    documents,
                    device,
                )

                positive_set = set(
                    int(pid)
                    for pid in record.get("positive_pids", [])
                )
                positive_indices = [
                    index
                    for index, pid in enumerate(candidate_pids)
                    if pid in positive_set
                ]

                nq_loss = torch.tensor(0.0, device=device)

                if positive_indices:
                    positive_tensor = torch.tensor(
                        positive_indices,
                        dtype=torch.long,
                        device=device,
                    )
                    nq_loss = (
                        torch.logsumexp(scores, dim=0)
                        - torch.logsumexp(
                            scores[positive_tensor],
                            dim=0,
                        )
                    )

                utility_loss = torch.tensor(0.0, device=device)

                if args.objective == "utility":
                    query_labels = labels.get(record["qid"], {})
                    utility_values = torch.tensor(
                        [
                            query_labels.get(pid, 0.0)
                            for pid in candidate_pids
                        ],
                        dtype=torch.float32,
                        device=device,
                    )

                    target_distribution = torch.softmax(
                        utility_values / 0.10,
                        dim=0,
                    )

                    utility_loss = -(
                        target_distribution
                        * torch.log_softmax(scores, dim=0)
                    ).sum()

                    task_loss = (
                        utility_loss
                        + args.lambda_nq * nq_loss
                    )
                else:
                    if not positive_indices:
                        continue
                    task_loss = nq_loss

                anchor_loss = F.mse_loss(
                    adapted_query,
                    base_query,
                )

                loss = (
                    task_loss
                    + args.gamma_anchor * anchor_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                epoch_loss += float(loss.detach().cpu())
                epoch_utility += float(
                    utility_loss.detach().cpu()
                )
                epoch_nq += float(nq_loss.detach().cpu())
                epoch_anchor += float(
                    anchor_loss.detach().cpu()
                )
                updates += 1

        denominator = max(updates, 1)

        epoch_metrics = {
            "epoch": epoch + 1,
            "updates": updates,
            "loss": epoch_loss / denominator,
            "utility_loss": epoch_utility / denominator,
            "nq_loss": epoch_nq / denominator,
            "anchor_loss": epoch_anchor / denominator,
        }
        history.append(epoch_metrics)

        print(json.dumps(epoch_metrics, indent=2))

    checkpoint = {
        "state_dict": {
            key: value.detach().cpu()
            for key, value in adapter.state_dict().items()
        },
        "dimension": dimension,
        "rank": args.rank,
        "objective": args.objective,
    }

    torch.save(checkpoint, output / "adapter.pt")

    configuration = {
        "dimension": dimension,
        "rank": args.rank,
        "objective": args.objective,
        "learning_rate": args.learning_rate,
        "lambda_nq": args.lambda_nq,
        "gamma_anchor": args.gamma_anchor,
        "epochs": args.epochs,
        "seed": args.seed,
    }

    write_json(output / "adapter_config.json", configuration)

    metrics = {
        **configuration,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }

    write_json(metrics_dir / "metrics.json", metrics)


def load_adapter(path):
    checkpoint_path = find_file(path, "adapter.pt")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    adapter = LowRankQueryAdapter(
        checkpoint["dimension"],
        checkpoint["rank"],
    )
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()

    return adapter


def load_encoder(model_root):
    from transformers import AutoModel, AutoTokenizer

    model_dir = find_model_dir(model_root)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.float32
        ),
    )
    model.to(device)
    model.eval()

    return model, tokenizer, device


def encode_e5(
    model,
    tokenizer,
    device,
    texts,
    prefix,
    batch_size,
):
    vectors = []

    for start in range(0, len(texts), batch_size):
        batch = [
            prefix + text
            for text in texts[start : start + batch_size]
        ]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(**encoded).last_hidden_state

            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (
                (output.float() * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1)
            )
            pooled = F.normalize(pooled, p=2, dim=-1)

        vectors.append(pooled.cpu().numpy().astype("float32"))

    return np.concatenate(vectors, axis=0)


def command_e5_index(args):
    import faiss

    output = ensure_dir(args.output)
    metrics_dir = ensure_dir(args.metrics)

    collection_path = Path(args.prepared) / "collection.tsv"
    model, tokenizer, device = load_encoder(args.model)

    sample_texts = []

    for _, text in iter_collection(collection_path):
        sample_texts.append(text)
        if len(sample_texts) >= args.train_sample:
            break

    if not sample_texts:
        raise RuntimeError("The prepared collection is empty")

    started = time.time()

    train_vectors = encode_e5(
        model,
        tokenizer,
        device,
        sample_texts,
        prefix="passage: ",
        batch_size=args.batch_size,
    )

    dimension = train_vectors.shape[1]

    maximum_nlist = max(1, len(train_vectors) // 40)
    nlist = 2 ** int(
        math.floor(math.log2(maximum_nlist))
    )
    nlist = max(64, min(2048, nlist))

    while nlist >= len(train_vectors):
        nlist //= 2

    pq_subquantizers = 64
    while dimension % pq_subquantizers != 0:
        pq_subquantizers //= 2

    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        nlist,
        pq_subquantizers,
        8,
        faiss.METRIC_INNER_PRODUCT,
    )

    index.train(train_vectors)

    batch_pids = []
    batch_texts = []

    for pid, text in iter_collection(collection_path):
        batch_pids.append(pid)
        batch_texts.append(text)

        if len(batch_texts) >= args.batch_size:
            vectors = encode_e5(
                model,
                tokenizer,
                device,
                batch_texts,
                prefix="passage: ",
                batch_size=args.batch_size,
            )
            index.add_with_ids(
                vectors,
                np.asarray(batch_pids, dtype=np.int64),
            )
            batch_pids = []
            batch_texts = []

    if batch_texts:
        vectors = encode_e5(
            model,
            tokenizer,
            device,
            batch_texts,
            prefix="passage: ",
            batch_size=args.batch_size,
        )
        index.add_with_ids(
            vectors,
            np.asarray(batch_pids, dtype=np.int64),
        )

    index.nprobe = min(32, nlist)

    index_path = output / "index.faiss"
    faiss.write_index(index, str(index_path))

    configuration = {
        "dimension": dimension,
        "nlist": nlist,
        "pq_subquantizers": pq_subquantizers,
        "pq_bits": 8,
        "nprobe": int(index.nprobe),
        "documents": int(index.ntotal),
    }

    write_json(output / "index_config.json", configuration)

    metrics = {
        **configuration,
        "index_bytes": index_path.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }

    write_json(metrics_dir / "metrics.json", metrics)


def retrieval_observation(
    pids,
    store,
    answers,
    positive_pids,
):
    answer_rank = None

    for rank, pid in enumerate(pids, start=1):
        if passage_contains_answer(store.get(pid), answers):
            answer_rank = rank
            break

    positive_set = set(int(pid) for pid in positive_pids)
    positive_rank = None

    if positive_set:
        for rank, pid in enumerate(pids, start=1):
            if int(pid) in positive_set:
                positive_rank = rank
                break

    return {
        "answer_hit_rank": answer_rank,
        "positive_hit_rank": positive_rank,
        "has_positive_labels": bool(positive_set),
    }


def aggregate_retrieval(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[(row["method"], row["dataset"])].append(row)

    result = {}

    for (method, dataset), values in groups.items():
        key = f"{method}/{dataset}"

        metrics = {
            "queries": len(values),
            "mrr@100": float(
                np.mean(
                    [
                        1.0 / value["answer_hit_rank"]
                        if value["answer_hit_rank"]
                        else 0.0
                        for value in values
                    ]
                )
            ),
            "mean_latency_ms": float(
                np.mean(
                    [value["latency_ms"] for value in values]
                )
            ),
        }

        for cutoff in [1, 5, 10, 20, 100]:
            metrics[f"answer_recall@{cutoff}"] = float(
                np.mean(
                    [
                        value["answer_hit_rank"] is not None
                        and value["answer_hit_rank"] <= cutoff
                        for value in values
                    ]
                )
            )

        labeled = [
            value
            for value in values
            if value["has_positive_labels"]
        ]

        if labeled:
            for cutoff in [1, 5, 10, 20, 100]:
                metrics[f"positive_recall@{cutoff}"] = float(
                    np.mean(
                        [
                            value["positive_hit_rank"] is not None
                            and value["positive_hit_rank"] <= cutoff
                            for value in labeled
                        ]
                    )
                )

        result[key] = metrics

    return result


def command_eval_colbert(args):
    rankings_dir = ensure_dir(args.rankings)
    metrics_dir = ensure_dir(args.metrics)

    searcher = load_searcher(
        args.index,
        args.model,
        args.prepared,
    )
    store = PassageStore(args.prepared)

    utility_adapter = load_adapter(args.utility_adapter)
    supervised_adapter = load_adapter(
        args.supervised_adapter
    )

    records = load_eval_records(
        args.prepared,
        args.max_queries,
        args.seed,
    )

    receipt = find_file(
        args.index,
        "index_receipt.json",
        required=False,
    )
    nbits = (
        json.loads(receipt.read_text())["nbits"]
        if receipt
        else "unknown"
    )

    variants = {
        f"colbert-base-nbits{nbits}": None,
        f"colbert-utility-nbits{nbits}": utility_adapter,
        f"colbert-supervised-nbits{nbits}": supervised_adapter,
    }

    ranking_path = rankings_dir / "rankings.jsonl"
    observations = []

    with ranking_path.open("w", encoding="utf-8") as ranking_file:
        for query_index, record in enumerate(records):
            encode_start = time.perf_counter()
            base_query = searcher.encode(record["question"])
            encode_ms = (
                time.perf_counter() - encode_start
            ) * 1000.0

            for method, adapter in variants.items():
                if adapter is None:
                    query = base_query
                else:
                    with torch.inference_mode():
                        query = adapter(
                            base_query.squeeze(0).float()
                        ).unsqueeze(0)

                search_start = time.perf_counter()
                pids, _, scores = searcher.dense_search(
                    query,
                    k=args.k,
                )
                search_ms = (
                    time.perf_counter() - search_start
                ) * 1000.0

                pids = [int(pid) for pid in pids]
                scores = [float(score) for score in scores]

                ranking_row = {
                    "qid": record["qid"],
                    "dataset": record["dataset"],
                    "question": record["question"],
                    "answers": record["answers"],
                    "method": method,
                    "pids": pids,
                    "scores": scores,
                }

                ranking_file.write(
                    json.dumps(
                        ranking_row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                observation = retrieval_observation(
                    pids,
                    store,
                    record["answers"],
                    record.get("positive_pids", []),
                )
                observation.update(
                    {
                        "qid": record["qid"],
                        "dataset": record["dataset"],
                        "method": method,
                        "latency_ms": encode_ms + search_ms,
                    }
                )
                observations.append(observation)

            print(
                f"ColBERT evaluation "
                f"{query_index + 1}/{len(records)}",
                flush=True,
            )

    store.close()

    metrics = {
        "nbits": nbits,
        "index_bytes": recursive_size(args.index),
        "results": aggregate_retrieval(observations),
    }

    write_json(metrics_dir / "metrics.json", metrics)
    write_jsonl(metrics_dir / "per_query.jsonl", observations)


def command_eval_e5(args):
    import faiss

    rankings_dir = ensure_dir(args.rankings)
    metrics_dir = ensure_dir(args.metrics)

    index_path = find_file(args.index, "index.faiss")
    index = faiss.read_index(str(index_path))

    if hasattr(index, "nprobe"):
        index.nprobe = min(32, index.nlist)

    model, tokenizer, device = load_encoder(args.model)
    store = PassageStore(args.prepared)

    records = load_eval_records(
        args.prepared,
        args.max_queries,
        args.seed,
    )

    observations = []
    ranking_path = rankings_dir / "rankings.jsonl"

    with ranking_path.open("w", encoding="utf-8") as ranking_file:
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            questions = [
                record["question"]
                for record in batch_records
            ]

            batch_start = time.perf_counter()

            vectors = encode_e5(
                model,
                tokenizer,
                device,
                questions,
                prefix="query: ",
                batch_size=args.batch_size,
            )
            scores, pids = index.search(vectors, args.k)

            latency_ms = (
                (time.perf_counter() - batch_start)
                * 1000.0
                / len(batch_records)
            )

            for record, query_pids, query_scores in zip(
                batch_records,
                pids,
                scores,
            ):
                valid = [
                    (int(pid), float(score))
                    for pid, score in zip(
                        query_pids,
                        query_scores,
                    )
                    if int(pid) >= 0
                ]

                result_pids = [pid for pid, _ in valid]
                result_scores = [score for _, score in valid]

                ranking_row = {
                    "qid": record["qid"],
                    "dataset": record["dataset"],
                    "question": record["question"],
                    "answers": record["answers"],
                    "method": "e5-ivfpq",
                    "pids": result_pids,
                    "scores": result_scores,
                }

                ranking_file.write(
                    json.dumps(
                        ranking_row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                observation = retrieval_observation(
                    result_pids,
                    store,
                    record["answers"],
                    record.get("positive_pids", []),
                )
                observation.update(
                    {
                        "qid": record["qid"],
                        "dataset": record["dataset"],
                        "method": "e5-ivfpq",
                        "latency_ms": latency_ms,
                    }
                )
                observations.append(observation)

            print(
                f"E5 evaluation "
                f"{min(start + args.batch_size, len(records))}/"
                f"{len(records)}",
                flush=True,
            )

    store.close()

    metrics = {
        "index_bytes": index_path.stat().st_size,
        "results": aggregate_retrieval(observations),
    }

    write_json(metrics_dir / "metrics.json", metrics)
    write_jsonl(metrics_dir / "per_query.jsonl", observations)


def load_rankings(path):
    rows = []

    for ranking_file in Path(path).rglob("rankings.jsonl"):
        rows.extend(iter_jsonl(ranking_file))

    return rows


def command_rag(args):
    set_seed(args.seed)

    predictions_dir = ensure_dir(args.predictions)
    metrics_dir = ensure_dir(args.metrics)

    question_records = {
        record["qid"]: record
        for record in load_eval_records(
            args.prepared,
            max_queries=0,
            seed=args.seed,
        )
    }

    ranking_rows = (
        load_rankings(args.colbert_rankings)
        + load_rankings(args.e5_rankings)
    )

    rows_by_dataset = defaultdict(list)

    for record in question_records.values():
        rows_by_dataset[record["dataset"]].append(record)

    selected_qids = set()

    for dataset, records in rows_by_dataset.items():
        selected = select_rows(
            records,
            args.max_queries,
            args.seed + len(dataset),
        )
        selected_qids.update(record["qid"] for record in selected)

    ranking_lookup = {
        (row["qid"], row["method"]): row
        for row in ranking_rows
        if row["qid"] in selected_qids
    }

    methods = sorted(
        {
            method
            for qid, method in ranking_lookup
            if qid in selected_qids
        }
    )

    store = PassageStore(args.prepared)
    model, tokenizer, device = load_causal_lm(args.model)

    system = (
        "Answer the question using the supplied evidence. "
        "Return only a short factual answer."
    )

    requests = []

    for qid in sorted(selected_qids):
        record = question_records[qid]

        no_context_prompt = render_chat_prompt(
            tokenizer,
            system,
            f"Question: {record['question']}\nShort answer:",
        )

        requests.append(
            {
                "qid": qid,
                "dataset": record["dataset"],
                "method": "phi-no-context",
                "question": record["question"],
                "answers": record["answers"],
                "pids": [],
                "prompt": no_context_prompt,
            }
        )

        for method in methods:
            ranking = ranking_lookup.get((qid, method))
            if ranking is None:
                continue

            selected_pids = ranking["pids"][: args.top_contexts]

            contexts = [
                f"[{index + 1}] {store.get(pid)}"
                for index, pid in enumerate(selected_pids)
            ]

            user = (
                "Evidence:\n"
                + "\n\n".join(contexts)
                + f"\n\nQuestion: {record['question']}\n"
                "Short answer:"
            )

            requests.append(
                {
                    "qid": qid,
                    "dataset": record["dataset"],
                    "method": method,
                    "question": record["question"],
                    "answers": record["answers"],
                    "pids": selected_pids,
                    "prompt": render_chat_prompt(
                        tokenizer,
                        system,
                        user,
                    ),
                }
            )

    store.close()

    started = time.time()

    generated = generate_texts(
        model,
        tokenizer,
        device,
        [request["prompt"] for request in requests],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=2048,
    )

    predictions = []

    for request, prediction in zip(requests, generated):
        predictions.append(
            {
                "qid": request["qid"],
                "dataset": request["dataset"],
                "method": request["method"],
                "question": request["question"],
                "answers": request["answers"],
                "pids": request["pids"],
                "prediction": prediction,
                "em": answer_em(
                    prediction,
                    request["answers"],
                ),
                "f1": answer_f1(
                    prediction,
                    request["answers"],
                ),
            }
        )

    write_jsonl(
        predictions_dir / "predictions.jsonl",
        predictions,
    )

    groups = defaultdict(list)

    for row in predictions:
        groups[(row["method"], row["dataset"])].append(row)

    results = {}

    for (method, dataset), rows in groups.items():
        results[f"{method}/{dataset}"] = {
            "queries": len(rows),
            "em": float(np.mean([row["em"] for row in rows])),
            "f1": float(np.mean([row["f1"] for row in rows])),
        }

    metrics = {
        "elapsed_seconds": time.time() - started,
        "results": results,
    }

    write_json(metrics_dir / "metrics.json", metrics)


def paired_bootstrap(
    rows,
    method_a,
    method_b,
    metric,
    samples,
    seed,
):
    by_method = defaultdict(dict)

    for row in rows:
        key = (row["dataset"], row["qid"])
        by_method[row["method"]][key] = float(row[metric])

    common = sorted(
        set(by_method.get(method_a, {}))
        & set(by_method.get(method_b, {}))
    )

    if not common:
        return None

    differences = np.asarray(
        [
            by_method[method_a][key]
            - by_method[method_b][key]
            for key in common
        ],
        dtype=np.float64,
    )

    generator = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)

    for index in range(samples):
        selected = generator.integers(
            0,
            len(differences),
            size=len(differences),
        )
        bootstrap[index] = differences[selected].mean()

    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "pairs": len(common),
        "mean_difference": float(differences.mean()),
        "ci_95_low": float(np.quantile(bootstrap, 0.025)),
        "ci_95_high": float(np.quantile(bootstrap, 0.975)),
    }


def command_analyze(args):
    output = ensure_dir(args.output)

    metric_documents = []

    for path in sorted(Path(args.metrics).rglob("metrics.json")):
        try:
            metric_documents.append(
                {
                    "source": str(path),
                    "metrics": json.loads(
                        path.read_text(encoding="utf-8")
                    ),
                }
            )
        except Exception as error:
            metric_documents.append(
                {
                    "source": str(path),
                    "error": str(error),
                }
            )

    prediction_file = find_file(
        args.predictions,
        "predictions.jsonl",
    )
    predictions = list(iter_jsonl(prediction_file))

    methods = sorted({row["method"] for row in predictions})

    comparisons = []

    utility_methods = [
        method
        for method in methods
        if "colbert-utility" in method
    ]
    base_methods = [
        method
        for method in methods
        if "colbert-base" in method
    ]
    supervised_methods = [
        method
        for method in methods
        if "colbert-supervised" in method
    ]
    e5_methods = [
        method
        for method in methods
        if method == "e5-ivfpq"
    ]

    if utility_methods:
        utility = utility_methods[0]

        for comparison_method in (
            base_methods + supervised_methods + e5_methods
        ):
            result = paired_bootstrap(
                predictions,
                utility,
                comparison_method,
                metric="f1",
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            if result:
                comparisons.append(result)

    grouped = defaultdict(list)

    for row in predictions:
        grouped[(row["method"], row["dataset"])].append(row)

    rag_table = []

    for (method, dataset), rows in sorted(grouped.items()):
        rag_table.append(
            {
                "method": method,
                "dataset": dataset,
                "queries": len(rows),
                "em": float(np.mean([row["em"] for row in rows])),
                "f1": float(np.mean([row["f1"] for row in rows])),
            }
        )

    summary = {
        "rag_results": rag_table,
        "paired_bootstrap": comparisons,
        "metric_documents": metric_documents,
    }

    write_json(output / "summary.json", summary)

    with (output / "rag_results.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "dataset",
                "queries",
                "em",
                "f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rag_table)

    with (output / "bootstrap.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fields = [
            "method_a",
            "method_b",
            "metric",
            "pairs",
            "mean_difference",
            "ci_95_low",
            "ci_95_high",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)

    markdown = [
        "# Reward-Aligned ColBERT Results",
        "",
        "## RAG results",
        "",
        "| Method | Dataset | Queries | EM | F1 |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in rag_table:
        markdown.append(
            f"| {row['method']} | {row['dataset']} | "
            f"{row['queries']} | {row['em']:.4f} | "
            f"{row['f1']:.4f} |"
        )

    markdown.extend(
        [
            "",
            "## Paired bootstrap comparisons",
            "",
            "| A | B | Difference | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )

    for row in comparisons:
        markdown.append(
            f"| {row['method_a']} | {row['method_b']} | "
            f"{row['mean_difference']:.4f} | "
            f"[{row['ci_95_low']:.4f}, "
            f"{row['ci_95_high']:.4f}] |"
        )

    (output / "report.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )


def create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    validate = subparsers.add_parser("validate")
    validate.add_argument("--raw", required=True)
    validate.add_argument("--colbert", required=True)
    validate.add_argument("--qwen", required=True)
    validate.add_argument("--phi", required=True)
    validate.add_argument("--e5", required=True)
    validate.add_argument("--provenance", required=True)
    validate.add_argument("--output", required=True)
    validate.set_defaults(function=command_validate)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--raw", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--mode", choices=["pilot", "full"], required=True)
    prepare.add_argument("--pilot-passages", type=int, default=200000)
    prepare.add_argument("--pilot-train-queries", type=int, default=200)
    prepare.add_argument("--pilot-eval-queries", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=13)
    prepare.set_defaults(function=command_prepare)

    index = subparsers.add_parser("index")
    index.add_argument("--prepared", required=True)
    index.add_argument("--model", required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--metrics", required=True)
    index.add_argument("--nbits", type=int, choices=[1, 2, 4], required=True)
    index.set_defaults(function=command_index)

    candidates = subparsers.add_parser("candidates")
    candidates.add_argument("--prepared", required=True)
    candidates.add_argument("--index", required=True)
    candidates.add_argument("--model", required=True)
    candidates.add_argument("--candidates", required=True)
    candidates.add_argument("--cache", required=True)
    candidates.add_argument("--max-queries", type=int, default=200)
    candidates.add_argument("--k", type=int, default=16)
    candidates.add_argument("--seed", type=int, default=17)
    candidates.set_defaults(function=command_candidates)

    utility = subparsers.add_parser("utility")
    utility.add_argument("--prepared", required=True)
    utility.add_argument("--candidates", required=True)
    utility.add_argument("--model", required=True)
    utility.add_argument("--output", required=True)
    utility.add_argument("--max-queries", type=int, default=200)
    utility.add_argument("--batch-size", type=int, default=4)
    utility.add_argument("--max-new-tokens", type=int, default=32)
    utility.set_defaults(function=command_utility)

    train = subparsers.add_parser("train")
    train.add_argument("--cache", required=True)
    train.add_argument("--labels", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--metrics", required=True)
    train.add_argument(
        "--objective",
        choices=["utility", "supervised"],
        required=True,
    )
    train.add_argument("--rank", type=int, default=16)
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--lambda-nq", type=float, default=0.25)
    train.add_argument("--gamma-anchor", type=float, default=0.02)
    train.add_argument("--seed", type=int, default=17)
    train.set_defaults(function=command_train)

    e5_index = subparsers.add_parser("e5-index")
    e5_index.add_argument("--prepared", required=True)
    e5_index.add_argument("--model", required=True)
    e5_index.add_argument("--output", required=True)
    e5_index.add_argument("--metrics", required=True)
    e5_index.add_argument("--train-sample", type=int, default=20000)
    e5_index.add_argument("--batch-size", type=int, default=64)
    e5_index.set_defaults(function=command_e5_index)

    eval_colbert = subparsers.add_parser("eval-colbert")
    eval_colbert.add_argument("--prepared", required=True)
    eval_colbert.add_argument("--index", required=True)
    eval_colbert.add_argument("--model", required=True)
    eval_colbert.add_argument("--utility-adapter", required=True)
    eval_colbert.add_argument("--supervised-adapter", required=True)
    eval_colbert.add_argument("--rankings", required=True)
    eval_colbert.add_argument("--metrics", required=True)
    eval_colbert.add_argument("--max-queries", type=int, default=100)
    eval_colbert.add_argument("--k", type=int, default=100)
    eval_colbert.add_argument("--seed", type=int, default=29)
    eval_colbert.set_defaults(function=command_eval_colbert)

    eval_e5 = subparsers.add_parser("eval-e5")
    eval_e5.add_argument("--prepared", required=True)
    eval_e5.add_argument("--index", required=True)
    eval_e5.add_argument("--model", required=True)
    eval_e5.add_argument("--rankings", required=True)
    eval_e5.add_argument("--metrics", required=True)
    eval_e5.add_argument("--max-queries", type=int, default=100)
    eval_e5.add_argument("--k", type=int, default=100)
    eval_e5.add_argument("--batch-size", type=int, default=32)
    eval_e5.add_argument("--seed", type=int, default=29)
    eval_e5.set_defaults(function=command_eval_e5)

    rag = subparsers.add_parser("rag")
    rag.add_argument("--prepared", required=True)
    rag.add_argument("--colbert-rankings", required=True)
    rag.add_argument("--e5-rankings", required=True)
    rag.add_argument("--model", required=True)
    rag.add_argument("--predictions", required=True)
    rag.add_argument("--metrics", required=True)
    rag.add_argument("--max-queries", type=int, default=50)
    rag.add_argument("--top-contexts", type=int, default=3)
    rag.add_argument("--batch-size", type=int, default=2)
    rag.add_argument("--max-new-tokens", type=int, default=32)
    rag.add_argument("--seed", type=int, default=41)
    rag.set_defaults(function=command_rag)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--metrics", required=True)
    analyze.add_argument("--predictions", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--bootstrap-samples", type=int, default=10000)
    analyze.add_argument("--seed", type=int, default=53)
    analyze.set_defaults(function=command_analyze)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    print("Command:", args.command, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)

    args.function(args)

    print("Completed:", args.command, flush=True)


if __name__ == "__main__":
    main()