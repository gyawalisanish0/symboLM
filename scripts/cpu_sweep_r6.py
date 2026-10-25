"""Stratified CPU sweep of the verified r6 SFT adapter.

This is a diagnostic sweep, not the preregistered v3 benchmark.  It selects
two distinct prompts from each of the 19 held-out categories (38 prompts),
serves Arm C with greedy decoding at the 256-token edge budget, and records
strict declared-answer scores plus raw generations.

The sweep is deterministic, resumable, and fail-closed:

* the held-out benchmark hash is checked before loading any model;
* selection is a SHA-256 ordering with a fixed seed, not an RNG;
* a malformed checkpoint line or a generation exception aborts the run;
* each cell is appended and fsynced before the next cell starts.

Run from the repository root:

    .venv\\Scripts\\python.exe scripts\\cpu_sweep_r6.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.strict_scorer import score_declared_answer
from symbolic.prompts import REGISTER_SYSTEM_PROMPT, build_chat_prompt

BASE_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BASE_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
ADAPTER = ROOT / "checkpoints" / "sft_v3_r6" / "adapter"
BENCHMARK = ROOT / "data" / "held_out_eval_200_v2.jsonl"
BENCHMARK_SHA256 = "F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707"
OUT_DIR = ROOT / "results" / "r6_closeout"
JSONL_PATH = OUT_DIR / "CPU_SWEEP.jsonl"
SUMMARY_PATH = OUT_DIR / "CPU_SWEEP.json"

SELECTION_SEED = "20261025"
N_PER_CATEGORY = 2
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.15
THREADS = 8


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Non-object JSONL row at {path}:{line_no}")
            rows.append(row)
    return rows


def load_benchmark() -> List[Dict[str, Any]]:
    actual = sha256_file(BENCHMARK)
    if actual != BENCHMARK_SHA256:
        raise RuntimeError(
            f"Held-out benchmark hash mismatch: expected {BENCHMARK_SHA256}, got {actual}"
        )
    rows = load_jsonl(BENCHMARK)
    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 benchmark rows, got {len(rows)}")
    return rows


def select_sweep(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select two distinct prompts per category by fixed SHA-256 ordering."""
    by_category: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        category = str(row.get("category", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        if not category or not prompt:
            raise ValueError(f"Benchmark row missing category/prompt: {row.get('id')!r}")
        # One representative per distinct prompt, keeping the lowest ID.
        current = by_category[category].get(prompt)
        if current is None or str(row["id"]) < str(current["id"]):
            by_category[category][prompt] = row

    categories = sorted(by_category)
    if len(categories) != 19:
        raise RuntimeError(f"Expected 19 categories, got {len(categories)}: {categories}")

    selected: List[Dict[str, Any]] = []
    for category in categories:
        unique_rows = list(by_category[category].values())
        if len(unique_rows) < N_PER_CATEGORY:
            raise RuntimeError(
                f"Category {category!r} has only {len(unique_rows)} unique prompts"
            )
        ranked = sorted(
            unique_rows,
            key=lambda r: sha256_text(f"{SELECTION_SEED}|{r['id']}|{r['prompt']}"),
        )
        selected.extend(ranked[:N_PER_CATEGORY])

    selected.sort(key=lambda r: str(r["id"]))
    ids = [str(r["id"]) for r in selected]
    if len(selected) != len(categories) * N_PER_CATEGORY:
        raise RuntimeError(f"Expected {len(categories) * N_PER_CATEGORY} rows, got {len(selected)}")
    if len(set(ids)) != len(ids):
        raise RuntimeError("Selection contains duplicate problem IDs")
    return selected


def load_completed_records(selected_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not JSONL_PATH.exists():
        return {}
    records = load_jsonl(JSONL_PATH)
    completed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        problem_id = str(record.get("problem_id", ""))
        if not problem_id:
            raise ValueError(f"Checkpoint row missing problem_id: {record!r}")
        if problem_id not in selected_ids:
            raise ValueError(
                f"Checkpoint contains {problem_id!r}, which is not in the frozen selection"
            )
        if record.get("status") != "completed":
            raise ValueError(
                f"Checkpoint contains non-completed row {problem_id!r}: "
                f"{record.get('status')!r}"
            )
        if problem_id in completed:
            raise ValueError(f"Duplicate checkpoint row for {problem_id!r}")
        completed[problem_id] = record
    return completed


def extract_plan_concept(text: str) -> Tuple[Optional[str], Optional[str]]:
    match = re.search(r"<reg:plan>(.*?)</reg:plan>", text or "", re.DOTALL | re.IGNORECASE)
    if not match:
        return None, None
    plan_text = match.group(1).strip()
    concept_match = re.search(r"concept\s*[:=]\s*([A-Za-z0-9_\-]+)", plan_text)
    return plan_text, (concept_match.group(1) if concept_match else None)


def make_record(
    item: Dict[str, Any],
    prompt: str,
    generated: str,
    prompt_tokens: int,
    generated_tokens: int,
    elapsed_sec: float,
    eos_emitted: bool,
) -> Dict[str, Any]:
    expected_kind = "bypass" if item["category"] == "atomic_scientific_fact" else "reasoning"
    strict = score_declared_answer(
        generated,
        arm="C",
        ground_truth=str(item["ground_truth"]),
        tokens_emitted=generated_tokens,
        budget=MAX_NEW_TOKENS,
        eos_emitted=eos_emitted,
        expected_kind=expected_kind,
    )
    plan_text, plan_concept = extract_plan_concept(generated)
    return {
        "status": "completed",
        "problem_id": str(item["id"]),
        "category": str(item["category"]),
        "prompt": str(item["prompt"]),
        "ground_truth": str(item["ground_truth"]),
        "expected_kind": expected_kind,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_sec": round(elapsed_sec, 3),
        "eos_emitted": bool(eos_emitted),
        "rendered_prompt_sha256": sha256_text(prompt),
        "generated_text_sha256": sha256_text(generated),
        "generated_text": generated,
        "plan_text": plan_text,
        "plan_concept": plan_concept,
        "strict": strict,
    }


def append_record(record: Dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSONL_PATH, "a", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def summarize(records: List[Dict[str, Any]], adapter_sha256: str) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if r["strict"]["is_correct"])
    format_valid = sum(1 for r in records if r["strict"]["format_valid"])
    strict_and_format = sum(
        1 for r in records if r["strict"]["strict_correct_and_format"]
    )
    eos = sum(1 for r in records if r["eos_emitted"])
    truncated = sum(1 for r in records if r["strict"]["truncated"])
    total_tokens = sum(int(r["generated_tokens"]) for r in records)
    total_seconds = sum(float(r["elapsed_sec"]) for r in records)

    by_category: Dict[str, Dict[str, Any]] = {}
    for category in sorted({r["category"] for r in records}):
        rows = [r for r in records if r["category"] == category]
        by_category[category] = {
            "n": len(rows),
            "strict_correct": sum(1 for r in rows if r["strict"]["is_correct"]),
            "format_valid": sum(1 for r in rows if r["strict"]["format_valid"]),
            "strict_correct_and_format": sum(
                1 for r in rows if r["strict"]["strict_correct_and_format"]
            ),
            "eos": sum(1 for r in rows if r["eos_emitted"]),
            "truncated": sum(1 for r in rows if r["strict"]["truncated"]),
            "mean_generated_tokens": round(
                statistics.fmean(int(r["generated_tokens"]) for r in rows), 3
            ),
            "mean_elapsed_sec": round(
                statistics.fmean(float(r["elapsed_sec"]) for r in rows), 3
            ),
            "problem_ids": [r["problem_id"] for r in rows],
        }

    failures = []
    for r in records:
        strict = r["strict"]
        if strict["is_correct"] and strict["format_valid"] and not strict["truncated"]:
            continue
        failures.append(
            {
                "problem_id": r["problem_id"],
                "category": r["category"],
                "expected_kind": r["expected_kind"],
                "strict_correct": strict["is_correct"],
                "format_valid": strict["format_valid"],
                "format_reason": strict["format_reason"],
                "truncated": strict["truncated"],
                "answer": strict["answer"],
                "ground_truth": r["ground_truth"],
                "generated_tokens": r["generated_tokens"],
                "eos_emitted": r["eos_emitted"],
                "plan_concept": r.get("plan_concept"),
            }
        )

    plan_concepts = Counter(
        r["plan_concept"] for r in records if r.get("plan_concept")
    )

    return {
        "schema": "symboLM.cpu_sweep.v1",
        "status": "diagnostic_only_not_a_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "arm": "C",
        "budget": MAX_NEW_TOKENS,
        "decoding": {
            "do_sample": False,
            "repetition_penalty": REPETITION_PENALTY,
        },
        "selection": {
            "method": "two distinct prompts per category, ranked by SHA-256(seed|id|prompt)",
            "seed": SELECTION_SEED,
            "n_per_category": N_PER_CATEGORY,
            "categories": len(by_category),
            "problem_ids": [r["problem_id"] for r in records],
        },
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
        },
        "library_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "base_model": {"id": BASE_ID, "revision": BASE_REVISION},
        "adapter": {
            "path": str(ADAPTER),
            "adapter_model_sha256": adapter_sha256,
        },
        "benchmark": {
            "path": str(BENCHMARK),
            "sha256": BENCHMARK_SHA256,
        },
        "counts": {
            "n": total,
            "strict_correct": correct,
            "strict_correct_rate": round(correct / total, 6) if total else 0.0,
            "format_valid": format_valid,
            "format_valid_rate": round(format_valid / total, 6) if total else 0.0,
            "strict_correct_and_format": strict_and_format,
            "strict_correct_and_format_rate": (
                round(strict_and_format / total, 6) if total else 0.0
            ),
            "eos": eos,
            "eos_rate": round(eos / total, 6) if total else 0.0,
            "truncated": truncated,
            "truncation_rate": round(truncated / total, 6) if total else 0.0,
            "total_generated_tokens": total_tokens,
            "total_elapsed_sec": round(total_seconds, 3),
            "mean_generated_tokens": round(total_tokens / total, 3) if total else 0.0,
            "mean_elapsed_sec": round(total_seconds / total, 3) if total else 0.0,
            "tokens_per_correct": (
                round(total_tokens / correct, 3) if correct else None
            ),
        },
        "by_category": by_category,
        "plan_concept_counts": dict(sorted(plan_concepts.items())),
        "failures": failures,
        "jsonl_path": str(JSONL_PATH),
        "jsonl_sha256": sha256_file(JSONL_PATH),
        "note": (
            "38-prompt stratified CPU diagnostic only. It is not the preregistered "
            "200x4x6 v3 experiment and supports no accuracy, token-efficiency, "
            "latency, or mobile claim."
        ),
    }


def main() -> None:
    torch.set_num_threads(THREADS)
    rows = load_benchmark()
    selected = select_sweep(rows)
    selected_ids = [str(r["id"]) for r in selected]
    completed = load_completed_records(selected_ids)

    print(
        f"[sweep] selected {len(selected)} prompts across "
        f"{len({r['category'] for r in selected})} categories",
        flush=True,
    )
    print(f"[sweep] already completed: {len(completed)}", flush=True)

    adapter_sha256 = sha256_file(ADAPTER / "adapter_model.safetensors")
    print(f"[sweep] adapter_sha256={adapter_sha256}", flush=True)
    print(f"[sweep] loading tokenizer from {ADAPTER}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER), trust_remote_code=True, local_files_only=True
    )

    print(f"[sweep] loading base {BASE_ID} @ {BASE_REVISION} in bfloat16", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        revision=BASE_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    base.resize_token_embeddings(len(tokenizer))

    print("[sweep] loading r6 PEFT adapter", flush=True)
    model = PeftModel.from_pretrained(base, str(ADAPTER))
    model.eval()

    for index, item in enumerate(selected, start=1):
        problem_id = str(item["id"])
        if problem_id in completed:
            continue

        prompt = build_chat_prompt(
            tokenizer, REGISTER_SYSTEM_PROMPT, str(item["prompt"]), open_think=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        print(
            f"[{index:02d}/{len(selected)}] {problem_id} ({item['category']}) "
            f"prompt_tokens={prompt_tokens}",
            flush=True,
        )

        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=REPETITION_PENALTY,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        except Exception as exc:
            error_record = {
                "status": "error",
                "problem_id": problem_id,
                "category": str(item["category"]),
                "prompt": str(item["prompt"]),
                "ground_truth": str(item["ground_truth"]),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            append_record(error_record)
            raise RuntimeError(
                f"Generation failed for {problem_id}; sweep aborted fail-closed"
            ) from exc

        elapsed = time.perf_counter() - t0
        generated_ids = out[0][prompt_tokens:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=False)
        eos_emitted = bool(
            len(generated_ids) > 0
            and int(generated_ids[-1].item()) == int(tokenizer.eos_token_id)
        )
        record = make_record(
            item=item,
            prompt=prompt,
            generated=generated,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(generated_ids),
            elapsed_sec=elapsed,
            eos_emitted=eos_emitted,
        )
        append_record(record)
        completed[problem_id] = record

        strict = record["strict"]
        print(
            f"    tokens={record['generated_tokens']} sec={record['elapsed_sec']:.1f} "
            f"eos={record['eos_emitted']} correct={strict['is_correct']} "
            f"format={strict['format_valid']} ({strict['format_reason']}) "
            f"answer={strict['answer']!r}",
            flush=True,
        )

    records = load_jsonl(JSONL_PATH)
    if len(records) != len(selected):
        raise RuntimeError(f"Expected {len(selected)} records, got {len(records)}")
    summary = summarize(records, adapter_sha256)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n[sweep] summary written -> {SUMMARY_PATH}", flush=True)
    print(
        f"[sweep] strict={summary['counts']['strict_correct']}/{summary['counts']['n']} "
        f"format={summary['counts']['format_valid']}/{summary['counts']['n']} "
        f"eos={summary['counts']['eos']}/{summary['counts']['n']} "
        f"truncated={summary['counts']['truncated']}/{summary['counts']['n']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
