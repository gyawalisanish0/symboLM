#!/usr/bin/env python3
"""Kaggle wrapper for the 2,400-cell SymboLM v3 diagnostic.

This wrapper is orchestration only.  It pins the evaluated repository commit,
verifies the r6 adapter hash before loading the model, runs the hardened
evaluator with ``--run-kind diagnostic --subset-size 100``, and writes a
wrapper receipt.  It does not authorize or launch the 4,800-cell confirmatory
run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_URL = "https://github.com/gyawalisanish0/symboLM.git"
REPO_COMMIT = "711030907f52789a3ccfbf1fa964f10ab718053d"
EXPECTED_ADAPTER_SHA256 = (
    "204823BC216BBF7329D6C7C5159ACF8F6688FD321AD1652757C2A9A9D066B31C"
)
EXPECTED_ADAPTER_CONFIG_SHA256 = (
    "03BDD025BE3DF781655939749FFA2CFEC2CB651F89863112FA3D15A621B70EA7"
)
EXPECTED_DATASET_SHA256 = (
    "F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707"
)
SUBSET_SIZE = 100
SUBSET_SEED = "20261025"
EXPECTED_CELLS = 2400
WORK = Path("/kaggle/working")
REPO = WORK / "symboLM"
OUT_DIR = WORK / "v3_diagnostic"
REPORT = OUT_DIR / "ROBUST_EVAL_REPORT.md"
WRAPPER_RECEIPT = OUT_DIR / "WRAPPER_RECEIPT.json"
WRAPPER_FAILURE = OUT_DIR / "WRAPPER_FAILURE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def fail(message: str) -> None:
    raise RuntimeError(message)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def find_adapter() -> tuple[Path, str]:
    candidates: list[Path] = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [name for name in dirs if name not in {".git", "trainer"}]
        if "adapter_model.safetensors" in files and "adapter_config.json" in files:
            candidates.append(Path(root))
    print(f"[INFO] Adapter candidates: {candidates}", flush=True)
    for candidate in candidates:
        weights = candidate / "adapter_model.safetensors"
        digest = sha256_file(weights)
        print(f"[INFO] Candidate {candidate} sha256={digest}", flush=True)
        if digest == EXPECTED_ADAPTER_SHA256:
            config_digest = sha256_file(candidate / "adapter_config.json")
            if config_digest != EXPECTED_ADAPTER_CONFIG_SHA256:
                fail(
                    "Adapter config SHA-256 mismatch: "
                    f"expected {EXPECTED_ADAPTER_CONFIG_SHA256}, got {config_digest}"
                )
            return candidate, digest
    fail("No mounted adapter matched the frozen r6 adapter SHA-256")


def count_results() -> tuple[int, int, dict]:
    results_path = OUT_DIR / "robust_eval_results.jsonl"
    if not results_path.is_file():
        fail(f"Missing results file: {results_path}")
    keys: set[str] = set()
    counts = {
        "strict_correct": 0,
        "strict_correct_and_format": 0,
        "format_valid": 0,
        "eos_emitted": 0,
        "truncated": 0,
        "answer_declared": 0,
    }
    with open(results_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = f"{record['problem_id']}|{record['arm']}|{record['budget']}"
            if key in keys:
                fail(f"Duplicate result cell {key!r} at line {line_no}")
            keys.add(key)
            for name in counts:
                if bool(record.get(name, False)):
                    counts[name] += 1
            if record.get("adapter_sha256") != EXPECTED_ADAPTER_SHA256:
                fail(f"Result cell {key!r} has the wrong adapter hash")
            if record.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
                fail(f"Result cell {key!r} has the wrong dataset hash")
            if record.get("run_kind") != "diagnostic":
                fail(f"Result cell {key!r} is not labeled diagnostic")
            if "\ufffd" in str(record.get("generated_text", "")):
                fail(f"Result cell {key!r} contains a Unicode replacement character")
    return len(keys), len(keys), counts


def main() -> None:
    started = time.time()
    print(f"[INFO] Python: {sys.version}", flush=True)
    print(f"[INFO] Working dir: {WORK}", flush=True)

    if REPO.exists():
        shutil.rmtree(REPO)
    run(
        [
            "git",
            "clone",
            "--branch",
            "research/rescue-v3",
            "--single-branch",
            REPO_URL,
            str(REPO),
        ]
    )
    run(["git", "checkout", "--detach", REPO_COMMIT], cwd=REPO)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if head != REPO_COMMIT:
        fail(f"Repository HEAD mismatch: expected {REPO_COMMIT}, got {head}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip()
    if status:
        fail(f"Repository checkout is dirty:\n{status}")

    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
        check=False,
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers==5.20.1",
            "peft==0.21.2",
            "accelerate==1.15.0",
            "sympy==1.14.0",
            "huggingface_hub==2.2.0",
            "tokenizers==0.23.3",
            "safetensors==0.8.0",
        ]
    )

    import torch
    import transformers
    import peft
    import accelerate
    import tokenizers
    import safetensors

    print(f"[INFO] torch={torch.__version__}", flush=True)
    print(f"[INFO] transformers={transformers.__version__}", flush=True)
    print(f"[INFO] peft={peft.__version__}", flush=True)
    print(f"[INFO] accelerate={accelerate.__version__}", flush=True)
    print(f"[INFO] tokenizers={tokenizers.__version__}", flush=True)
    print(f"[INFO] safetensors={safetensors.__version__}", flush=True)
    if transformers.__version__ != "5.20.1":
        fail(f"Unexpected transformers version: {transformers.__version__}")
    if not torch.cuda.is_available():
        fail("CUDA is not available; the v3 diagnostic requires a GPU")
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    adapter, adapter_sha = find_adapter()
    print(f"[INFO] Using adapter: {adapter}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO / "eval" / "robust_eval.py"),
        "--input",
        str(REPO / "data" / "held_out_eval_200_v2.jsonl"),
        "--adapter",
        str(adapter),
        "--expected-adapter-sha256",
        EXPECTED_ADAPTER_SHA256,
        "--expected-sha256",
        EXPECTED_DATASET_SHA256,
        "--subset-size",
        str(SUBSET_SIZE),
        "--subset-seed",
        SUBSET_SEED,
        "--run-kind",
        "diagnostic",
        "--out-dir",
        str(OUT_DIR),
        "--report",
        str(REPORT),
        "--arms",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "--budgets",
        "256",
        "512",
        "1024",
        "2048",
        "--device",
        "cuda",
        "--seed",
        "42",
        "--backup-every",
        "20",
    ]
    run(command, cwd=REPO)

    manifest_path = OUT_DIR / "EVAL_RUN_MANIFEST.json"
    if not manifest_path.is_file():
        fail(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "run_kind": "diagnostic",
        "expected_cells": EXPECTED_CELLS,
        "adapter_sha256": EXPECTED_ADAPTER_SHA256,
        "adapter_sha256_match": True,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "strict_scoring": True,
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            fail(
                f"Manifest field {field!r} mismatch: "
                f"expected {expected!r}, got {manifest.get(field)!r}"
            )
    subset = manifest.get("subset") or {}
    if subset.get("n") != SUBSET_SIZE:
        fail(f"Manifest subset size mismatch: {subset.get('n')!r}")
    if str(subset.get("seed")) != SUBSET_SEED:
        fail(f"Manifest subset seed mismatch: {subset.get('seed')!r}")
    if subset.get("method") != "proportional_stratified_sha256":
        fail(f"Manifest subset method mismatch: {subset.get('method')!r}")
    if len(subset.get("selected_ids") or []) != SUBSET_SIZE:
        fail("Manifest selected_ids length does not match the subset size")
    contracts = manifest.get("contracts") or {}
    for field in (
        "evaluator_sha256",
        "run_contract_sha256",
        "strict_scorer_sha256",
        "prompts_sha256",
    ):
        if not contracts.get(field):
            fail(f"Manifest contract hash is missing: {field}")

    total_cells, unique_cells, counts = count_results()
    if total_cells != EXPECTED_CELLS or unique_cells != EXPECTED_CELLS:
        fail(
            f"Completed grid mismatch: expected {EXPECTED_CELLS}, "
            f"got total={total_cells} unique={unique_cells}"
        )

    receipt = {
        "schema": "symboLM.v3_diagnostic_wrapper.v1",
        "status": "completed",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim_boundary": (
            "2,400-cell diagnostic only. Not the 4,800-cell confirmatory run. "
            "This receipt establishes pipeline completion and grid coverage, "
            "not benchmark accuracy, token efficiency, latency, or mobile "
            "performance."
        ),
        "repo_url": REPO_URL,
        "repo_commit": REPO_COMMIT,
        "repo_head": head,
        "adapter_path": str(adapter),
        "adapter_sha256": adapter_sha,
        "adapter_config_sha256": EXPECTED_ADAPTER_CONFIG_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "subset": {
            "size": SUBSET_SIZE,
            "seed": SUBSET_SEED,
            "method": subset.get("method"),
            "selected_ids": subset.get("selected_ids"),
        },
        "expected_cells": EXPECTED_CELLS,
        "completed_cells": total_cells,
        "unique_cells": unique_cells,
        "counts": counts,
        "energy_measurement": manifest.get("energy_measurement"),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "tokenizers": tokenizers.__version__,
            "safetensors": safetensors.__version__,
        },
        "manifest_sha256": sha256_file(manifest_path),
        "results_sha256": sha256_file(OUT_DIR / "robust_eval_results.jsonl"),
        "report_sha256": sha256_file(REPORT) if REPORT.is_file() else None,
        "wrapper_sha256": sha256_file(Path(__file__)),
        "command": command,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(WRAPPER_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)
    print("[DONE] 2,400-cell v3 diagnostic completed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(
            WRAPPER_FAILURE,
            {
                "schema": "symboLM.v3_diagnostic_wrapper.v1",
                "status": "failed",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "repo_commit": REPO_COMMIT,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(f"[FATAL] {exc}", flush=True)
        raise
