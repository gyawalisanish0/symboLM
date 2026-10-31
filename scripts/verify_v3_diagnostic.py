#!/usr/bin/env python3
"""Independently verify a downloaded 2,400-cell SymboLM v3 diagnostic.

This verifier is deliberately model-free.  It does not trust the wrapper
receipt alone: it reloads the frozen benchmark, recomputes the deterministic
subset, checks the completed Cartesian grid, checks every row's provenance
fields, recomputes the code-contract hashes, and cross-checks the wrapper
receipt against the actual artifacts.

Usage:
    python scripts/verify_v3_diagnostic.py --output-dir path/to/v3_diagnostic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_contract import (  # noqa: E402
    EvalContractError,
    adapter_artifact_sha256,
    expected_cell_keys,
    load_benchmark_rows,
    select_stratified_subset,
    sha256_file,
    sha256_text,
)

EXPECTED_ADAPTER_SHA256 = (
    "204823BC216BBF7329D6C7C5159ACF8F6688FD321AD1652757C2A9A9D066B31C"
)
EXPECTED_ADAPTER_CONFIG_SHA256 = (
    "03BDD025BE3DF781655939749FFA2CFEC2CB651F89863112FA3D15A621B70EA7"
)
EXPECTED_DATASET_SHA256 = (
    "F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707"
)
EXPECTED_COMMIT = "17d62d1bc8e26f6a02bee67e777ebf8984a0ca02"
EXPECTED_BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EXPECTED_BASE_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
EXPECTED_ARMS = ["A", "B", "C", "D", "E", "F"]
EXPECTED_BUDGETS = [256, 512, 1024, 2048]
EXPECTED_SUBSET_SIZE = 100
EXPECTED_SUBSET_SEED = "20261025"
EXPECTED_CELLS = 2400


class VerificationError(RuntimeError):
    """Raised when a diagnostic artifact violates its contract."""


def fail(message: str) -> None:
    raise VerificationError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_json(path: Path) -> Any:
    if not path.is_file():
        fail(f"Missing JSON artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"Malformed JSON artifact {path}: {exc}")


def normalize_subset(subset: Any) -> dict:
    require(isinstance(subset, dict), "Manifest subset must be an object")
    selected_ids = subset.get("selected_ids")
    require(isinstance(selected_ids, list), "Manifest subset selected_ids must be a list")
    return {
        "method": subset.get("method"),
        "seed": None if subset.get("seed") is None else str(subset.get("seed")),
        "n": subset.get("n"),
        "available": subset.get("available"),
        "categories": subset.get("categories"),
        "quotas": subset.get("quotas"),
        "selected_ids": [str(value) for value in selected_ids],
    }


def compute_contract_hashes(repo_root: Path) -> dict:
    files = {
        "evaluator_sha256": repo_root / "eval" / "robust_eval.py",
        "run_contract_sha256": repo_root / "eval" / "run_contract.py",
        "strict_scorer_sha256": repo_root / "eval" / "strict_scorer.py",
        "prompts_sha256": repo_root / "symbolic" / "prompts.py",
    }
    hashes: dict[str, Any] = {}
    for name, path in files.items():
        require(path.is_file(), f"Required contract file is missing: {path}")
        hashes[name] = sha256_file(path)
    overrides = repo_root / "data" / "ground_truth_overrides.json"
    hashes["ground_truth_overrides_sha256"] = (
        sha256_file(overrides) if overrides.is_file() else None
    )
    return hashes


def parse_results(results_path: Path) -> list[dict]:
    if not results_path.is_file():
        fail(f"Missing results file: {results_path}")
    records: list[dict] = []
    with open(results_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                fail(f"Blank line in results at line {line_no}")
            try:
                record = json.loads(line)
            except Exception as exc:
                fail(f"Malformed result JSON at line {line_no}: {exc}")
            require(isinstance(record, dict), f"Result line {line_no} is not an object")
            records.append(record)
    return records


def verify(
    output_dir: Path,
    repo_root: Path,
    adapter_path: Path | None,
) -> dict:
    checks: list[str] = []

    manifest_path = output_dir / "EVAL_RUN_MANIFEST.json"
    results_path = output_dir / "robust_eval_results.jsonl"
    report_path = output_dir / "ROBUST_EVAL_REPORT.md"
    wrapper_path = output_dir / "WRAPPER_RECEIPT.json"
    failure_path = output_dir / "WRAPPER_FAILURE.json"

    require(not failure_path.exists(), f"Wrapper failure receipt exists: {failure_path}")
    checks.append("no_wrapper_failure")

    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict), "Run manifest must be an object")
    checks.append("manifest_present")

    require(manifest.get("run_kind") == "diagnostic", "Run kind is not diagnostic")
    require(manifest.get("dataset_sha256") == EXPECTED_DATASET_SHA256, "Dataset hash mismatch")
    require(manifest.get("dataset_sha256_match") is True, "Dataset hash was not verified")
    require(manifest.get("adapter_sha256") == EXPECTED_ADAPTER_SHA256, "Adapter hash mismatch")
    require(manifest.get("adapter_sha256_match") is True, "Adapter hash was not verified")
    require(manifest.get("expected_cells") == EXPECTED_CELLS, "Expected-cell count mismatch")
    require(manifest.get("strict_scoring") is True, "Strict scoring is not enabled")
    require(
        manifest.get("truncation_rule") == "hit_budget_without_eos_is_fail",
        "Truncation rule mismatch",
    )
    require(manifest.get("commit_hash") == EXPECTED_COMMIT, "Commit hash mismatch")
    require(manifest.get("base_model") == EXPECTED_BASE_MODEL, "Base model mismatch")
    require(manifest.get("base_revision") == EXPECTED_BASE_REVISION, "Base revision mismatch")
    require(manifest.get("arms") == EXPECTED_ARMS, "Arm order/content mismatch")
    require(manifest.get("budgets") == EXPECTED_BUDGETS, "Budget order/content mismatch")
    require(manifest.get("random_seed") == 42, "Random seed mismatch")
    require(bool(manifest.get("energy_measurement")), "Energy measurement mode is missing")
    checks.append("manifest_identity")

    contracts = manifest.get("contracts")
    require(isinstance(contracts, dict), "Manifest contracts must be an object")
    expected_contracts = compute_contract_hashes(repo_root)
    for name, expected in expected_contracts.items():
        require(contracts.get(name) == expected, f"Contract hash mismatch: {name}")
    checks.append("contract_hashes")

    benchmark_path = repo_root / "data" / "held_out_eval_200_v2.jsonl"
    rows = load_benchmark_rows(
        benchmark_path,
        expected_sha256=EXPECTED_DATASET_SHA256,
        expected_rows=200,
    )
    selected, recomputed_subset = select_stratified_subset(
        rows, EXPECTED_SUBSET_SIZE, EXPECTED_SUBSET_SEED
    )
    manifest_subset = normalize_subset(manifest.get("subset"))
    require(
        manifest_subset == normalize_subset(recomputed_subset),
        "Manifest subset does not match the deterministic recomputation",
    )
    checks.append("subset_recomputed")

    selected_ids = [str(row["id"]) for row in selected]
    require(len(selected_ids) == EXPECTED_SUBSET_SIZE, "Subset size mismatch")
    require(len(set(selected_ids)) == EXPECTED_SUBSET_SIZE, "Subset IDs are not unique")
    expected_keys = expected_cell_keys(selected, EXPECTED_ARMS, EXPECTED_BUDGETS)
    require(len(expected_keys) == EXPECTED_CELLS, "Expected grid size mismatch")
    checks.append("expected_grid")

    records = parse_results(results_path)
    require(len(records) == EXPECTED_CELLS, f"Result count mismatch: {len(records)}")
    seen: set[str] = set()
    counts = {
        "strict_correct": 0,
        "strict_correct_and_format": 0,
        "format_valid": 0,
        "eos_emitted": 0,
        "truncated": 0,
        "answer_declared": 0,
    }
    breakdown: dict[str, dict[str, int]] = {}
    for line_no, record in enumerate(records, start=1):
        pid = record.get("problem_id")
        arm = record.get("arm")
        budget = record.get("budget")
        require(isinstance(pid, str) and pid, f"Bad problem_id at line {line_no}")
        require(arm in EXPECTED_ARMS, f"Bad arm at line {line_no}")
        require(isinstance(budget, int) and budget in EXPECTED_BUDGETS, f"Bad budget at line {line_no}")
        key = f"{pid}|{arm}|{budget}"
        require(key not in seen, f"Duplicate cell {key!r} at line {line_no}")
        seen.add(key)
        require(record.get("run_kind") == "diagnostic", f"Cell {key!r} is not diagnostic")
        require(
            record.get("adapter_sha256") == EXPECTED_ADAPTER_SHA256,
            f"Cell {key!r} has the wrong adapter hash",
        )
        require(
            record.get("dataset_sha256") == EXPECTED_DATASET_SHA256,
            f"Cell {key!r} has the wrong dataset hash",
        )
        require(
            record.get("evaluator_sha256") == contracts.get("evaluator_sha256"),
            f"Cell {key!r} has the wrong evaluator hash",
        )
        require(
            record.get("strict_scorer_sha256") == contracts.get("strict_scorer_sha256"),
            f"Cell {key!r} has the wrong strict-scorer hash",
        )
        for field in (
            "strict_correct",
            "strict_correct_and_format",
            "format_valid",
            "eos_emitted",
            "truncated",
            "answer_declared",
            "hit_budget",
        ):
            require(isinstance(record.get(field), bool), f"Cell {key!r} has non-boolean {field}")
        require(
            record.get("is_correct") == record.get("strict_correct"),
            f"Cell {key!r} has inconsistent is_correct",
        )
        require(
            record.get("strict_correct_and_format")
            == bool(record.get("strict_correct") and record.get("format_valid")),
            f"Cell {key!r} has inconsistent strict_correct_and_format",
        )
        require(
            isinstance(record.get("tokens_emitted"), int)
            and record.get("tokens_emitted") >= 0,
            f"Cell {key!r} has invalid tokens_emitted",
        )
        require(
            record.get("hit_budget") == (record.get("tokens_emitted") >= budget),
            f"Cell {key!r} has inconsistent hit_budget",
        )
        if record.get("truncated"):
            require(record.get("hit_budget") is True, f"Cell {key!r} truncates without hitting budget")
        text = record.get("generated_text")
        require(isinstance(text, str) and text.strip(), f"Cell {key!r} has empty generated text")
        require("\ufffd" not in text, f"Cell {key!r} contains a replacement character")
        for hash_field in ("rendered_prompt_sha256", "generated_text_sha256"):
            value = record.get(hash_field)
            require(
                isinstance(value, str) and len(value) == 64
                and all(ch in "0123456789ABCDEFabcdef" for ch in value),
                f"Cell {key!r} has an invalid {hash_field}",
            )
        if not record.get("generated_text_was_cut_for_storage", False):
            require(
                record.get("generated_text_sha256") == sha256_text(text),
                f"Cell {key!r} generated-text hash mismatch",
            )
        for name in counts:
            if record.get(name):
                counts[name] += 1
        bucket = breakdown.setdefault(
            f"{arm}|{budget}",
            {
                "cells": 0,
                "strict_correct": 0,
                "format_valid": 0,
                "eos_emitted": 0,
                "truncated": 0,
            },
        )
        bucket["cells"] += 1
        for name in ("strict_correct", "format_valid", "eos_emitted", "truncated"):
            if record.get(name):
                bucket[name] += 1

    missing = expected_keys - seen
    extra = seen - expected_keys
    require(not missing, f"Missing cells: {sorted(missing)[:5]}")
    require(not extra, f"Unexpected cells: {sorted(extra)[:5]}")
    checks.append("complete_unique_grid")

    wrapper = read_json(wrapper_path)
    require(isinstance(wrapper, dict), "Wrapper receipt must be an object")
    require(wrapper.get("status") == "completed", "Wrapper receipt is not completed")
    require(wrapper.get("repo_commit") == EXPECTED_COMMIT, "Wrapper commit mismatch")
    require(wrapper.get("adapter_sha256") == EXPECTED_ADAPTER_SHA256, "Wrapper adapter hash mismatch")
    require(wrapper.get("dataset_sha256") == EXPECTED_DATASET_SHA256, "Wrapper dataset hash mismatch")
    require(wrapper.get("expected_cells") == EXPECTED_CELLS, "Wrapper expected-cell mismatch")
    require(wrapper.get("completed_cells") == EXPECTED_CELLS, "Wrapper completed-cell mismatch")
    require(wrapper.get("unique_cells") == EXPECTED_CELLS, "Wrapper unique-cell mismatch")
    require(wrapper.get("counts") == counts, "Wrapper counts do not match recomputed counts")
    require(wrapper.get("manifest_sha256") == sha256_file(manifest_path), "Wrapper manifest hash mismatch")
    require(wrapper.get("results_sha256") == sha256_file(results_path), "Wrapper results hash mismatch")
    if report_path.is_file():
        require(wrapper.get("report_sha256") == sha256_file(report_path), "Wrapper report hash mismatch")
    require(bool(wrapper.get("wrapper_sha256")), "Wrapper script hash is missing")
    require(
        "diagnostic" in str(wrapper.get("claim_boundary", "")).lower(),
        "Wrapper claim boundary does not identify the diagnostic",
    )
    checks.append("wrapper_receipt")

    require(report_path.is_file(), f"Missing report: {report_path}")
    report = report_path.read_text(encoding="utf-8")
    require("Diagnostic only." in report, "Report lacks the diagnostic banner")
    require(
        "not the 200-problem x 4-budget x 6-arm preregistered experiment" in report,
        "Report lacks the non-confirmatory warning",
    )
    require(EXPECTED_ADAPTER_SHA256 in report, "Report lacks the adapter hash")
    require(EXPECTED_DATASET_SHA256 in report, "Report lacks the dataset hash")
    if manifest.get("energy_measurement") != "nvml":
        require(
            "CPU TDP estimate only" in report,
            "Report does not label the non-NVML energy mode as a CPU estimate",
        )
    checks.append("report_boundary")

    if adapter_path is not None and adapter_path.exists():
        require(
            adapter_artifact_sha256(adapter_path) == EXPECTED_ADAPTER_SHA256,
            "Local adapter weights do not match the frozen hash",
        )
        config_path = adapter_path / "adapter_config.json"
        require(config_path.is_file(), "Local adapter config is missing")
        require(
            sha256_file(config_path) == EXPECTED_ADAPTER_CONFIG_SHA256,
            "Local adapter config does not match the frozen hash",
        )
        checks.append("local_adapter_hash")
    else:
        checks.append("local_adapter_unavailable")

    return {
        "schema": "symboLM.v3_diagnostic_verification.v1",
        "status": "verified",
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(output_dir),
        "checks": checks,
        "counts": counts,
        "breakdown": breakdown,
        "manifest_sha256": sha256_file(manifest_path),
        "results_sha256": sha256_file(results_path),
        "report_sha256": sha256_file(report_path),
        "wrapper_sha256": wrapper.get("wrapper_sha256"),
        "energy_measurement": manifest.get("energy_measurement"),
        "claim_boundary": (
            "Verified pipeline completion and grid coverage for the 2,400-cell "
            "diagnostic only. This is not the 4,800-cell confirmatory run and is "
            "not a benchmark, token-efficiency, latency, or mobile claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=ROOT / "checkpoints" / "sft_v3_r6" / "adapter",
    )
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args()

    receipt = verify(
        output_dir=args.output_dir.resolve(),
        repo_root=args.repo_root.resolve(),
        adapter_path=args.adapter_path.resolve() if args.adapter_path else None,
    )
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (VerificationError, EvalContractError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
