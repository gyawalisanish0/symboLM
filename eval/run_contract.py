"""Fail-closed contracts for the SymboLM evaluation harness.

This module deliberately has no torch/transformers dependency.  The evaluator
imports it before any model is loaded, and the model-free test suite can verify
the dataset, adapter, subset, and completion contracts without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


class EvalContractError(RuntimeError):
    """Raised when an evaluation input or run artifact violates its contract."""


REQUIRED_ROW_FIELDS = ("id", "prompt", "ground_truth", "category")


def sha256_file(path: str | Path) -> str:
    """Return the uppercase SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def adapter_artifact_sha256(adapter_path: str | Path) -> str:
    """Hash the PEFT adapter weights, or a merged model's safetensors file.

    PEFT writes ``adapter_model.safetensors`` (current) or
    ``adapter_model.bin`` (legacy).  A merged checkpoint is accepted for
    historical rescoring, but the resulting hash is still recorded and checked.
    """
    path = Path(adapter_path)
    if path.is_file():
        return sha256_file(path)

    if not path.exists():
        raise EvalContractError(f"Adapter path does not exist: {path}")

    candidates = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    )
    for name in candidates:
        candidate = path / name
        if candidate.is_file():
            return sha256_file(candidate)
    raise EvalContractError(
        f"No adapter/model weight file found under {path}; "
        f"looked for {', '.join(candidates)}"
    )


def validate_adapter_config(
    adapter_path: str | Path,
    base_model: str,
    base_revision: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Check a PEFT ``adapter_config.json`` against the requested base model.

    The adapter config records the base model path/name.  It does not always
    record a revision, so the revision is checked only when present.  A missing
    config is treated as a merged checkpoint and is not an error.
    """
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvalContractError(f"Malformed adapter config: {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise EvalContractError(f"Adapter config must be a JSON object: {config_path}")

    raw_base = config.get("base_model_name_or_path")
    recorded_base = "" if raw_base is None else str(raw_base).strip()
    if recorded_base and recorded_base != str(base_model).strip():
        raise EvalContractError(
            "Adapter/base-model mismatch: "
            f"adapter records {recorded_base!r}, requested {base_model!r}"
        )

    raw_revision = config.get("revision")
    recorded_revision = "" if raw_revision is None else str(raw_revision).strip()
    if base_revision and recorded_revision and recorded_revision != str(base_revision).strip():
        raise EvalContractError(
            "Adapter/base-revision mismatch: "
            f"adapter records {recorded_revision!r}, requested {base_revision!r}"
        )
    return config


def load_benchmark_rows(
    path: str | Path,
    expected_sha256: Optional[str] = None,
    expected_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load and validate a JSONL benchmark, fail closed on any bad row."""
    benchmark_path = Path(path)
    if not benchmark_path.is_file():
        raise EvalContractError(f"Benchmark file does not exist: {benchmark_path}")

    if expected_sha256:
        actual = sha256_file(benchmark_path)
        if actual.upper() != str(expected_sha256).upper():
            raise EvalContractError(
                f"Dataset SHA-256 mismatch: expected {str(expected_sha256).upper()}, "
                f"got {actual}"
            )

    rows: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    with open(benchmark_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalContractError(
                    f"Malformed JSONL at {benchmark_path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise EvalContractError(
                    f"Non-object JSONL row at {benchmark_path}:{line_no}"
                )
            missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
            if missing:
                raise EvalContractError(
                    f"Benchmark row at {benchmark_path}:{line_no} is missing "
                    f"required field(s): {', '.join(missing)}"
                )
            problem_id = str(row.get("id", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            ground_truth = str(row.get("ground_truth", "")).strip()
            category = str(row.get("category", "")).strip()
            if not problem_id:
                raise EvalContractError(
                    f"Benchmark row at {benchmark_path}:{line_no} has an empty id"
                )
            if not prompt:
                raise EvalContractError(
                    f"Benchmark row {problem_id!r} at {benchmark_path}:{line_no} "
                    "has an empty prompt"
                )
            if not ground_truth:
                raise EvalContractError(
                    f"Benchmark row {problem_id!r} at {benchmark_path}:{line_no} "
                    "has an empty ground_truth"
                )
            if not category:
                raise EvalContractError(
                    f"Benchmark row {problem_id!r} at {benchmark_path}:{line_no} "
                    "has an empty category"
                )
            if problem_id in seen_ids:
                raise EvalContractError(
                    f"Duplicate benchmark id {problem_id!r} at {benchmark_path}:{line_no}"
                )
            seen_ids.add(problem_id)
            normalized = dict(row)
            normalized["id"] = problem_id
            rows.append(normalized)

    if not rows:
        raise EvalContractError(f"Benchmark contains no rows: {benchmark_path}")
    if expected_rows is not None and len(rows) != expected_rows:
        raise EvalContractError(
            f"Benchmark row-count mismatch: expected {expected_rows}, got {len(rows)}"
        )
    return rows


def _rank_key(seed: int | str, row: Dict[str, Any]) -> str:
    return sha256_text(f"{seed}|{row.get('id', '')}|{row.get('prompt', '')}")


def select_stratified_subset(
    rows: Sequence[Dict[str, Any]],
    n: int,
    seed: int | str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select exactly *n* rows with deterministic, stratified SHA-256 ranking.

    Every category receives at least one row when ``n`` is at least the number
    of categories.  The remainder is allocated proportionally to category size
    using largest remainders, and ties are broken by a SHA-256 key.  No RNG is
    used, so the same input, seed, and n always produce the same row set.
    """
    if n <= 0:
        raise EvalContractError("Subset size must be positive")
    if n > len(rows):
        raise EvalContractError(
            f"Subset size {n} exceeds available rows {len(rows)}"
        )

    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        category = str(row.get("category", "")).strip()
        if not category:
            raise EvalContractError(f"Row {row.get('id')!r} has no category")
        item = dict(row)
        item["_benchmark_index"] = index
        by_category[category].append(item)

    categories = sorted(by_category)
    counts = {category: len(by_category[category]) for category in categories}
    total = len(rows)

    if n == total:
        selected = [dict(row) for row in rows]
        for item in selected:
            item.pop("_benchmark_index", None)
        return selected, {
            "method": "all_rows",
            "seed": seed,
            "n": n,
            "available": total,
            "categories": categories,
            "quotas": counts,
            "selected_ids": [str(row["id"]) for row in selected],
        }

    if n < len(categories):
        # The diagnostic grid is expected to be larger than 19 categories, but
        # keep a deterministic fallback for smaller requests.
        ranked_categories = sorted(
            categories, key=lambda category: sha256_text(f"{seed}|category|{category}")
        )
        chosen = ranked_categories[:n]
        selected = []
        for category in chosen:
            ranked = sorted(
                by_category[category], key=lambda row: _rank_key(seed, row)
            )
            selected.append(ranked[0])
        selected.sort(key=lambda row: int(row["_benchmark_index"]))
        quotas = {category: (1 if category in chosen else 0) for category in categories}
        for item in selected:
            item.pop("_benchmark_index", None)
        return selected, {
            "method": "category_hash_fallback",
            "seed": seed,
            "n": n,
            "available": total,
            "categories": categories,
            "quotas": quotas,
            "selected_ids": [str(row["id"]) for row in selected],
        }

    quotas = {category: 1 for category in categories}
    remaining = n - len(categories)
    capacity = {category: counts[category] - 1 for category in categories}

    ideals = {
        category: (counts[category] * remaining / total) for category in categories
    }
    floors = {
        category: min(capacity[category], int(math.floor(ideals[category])))
        for category in categories
    }
    for category in categories:
        quotas[category] += floors[category]
    assigned = sum(quotas.values())

    leftovers = n - assigned
    if leftovers:
        order = sorted(
            categories,
            key=lambda category: (
                -(ideals[category] - floors[category]),
                sha256_text(f"{seed}|remainder|{category}"),
            ),
        )
        for category in order:
            if leftovers <= 0:
                break
            room = counts[category] - quotas[category]
            if room <= 0:
                continue
            take = min(room, leftovers)
            quotas[category] += take
            leftovers -= take
        if leftovers:
            # Largest-remainder rounding can leave slots when a category was
            # capped.  Fill deterministically by SHA-256 over the categories.
            for category in sorted(
                categories, key=lambda category: sha256_text(f"{seed}|fill|{category}")
            ):
                if leftovers <= 0:
                    break
                room = counts[category] - quotas[category]
                if room <= 0:
                    continue
                take = min(room, leftovers)
                quotas[category] += take
                leftovers -= take
    if leftovers:
        raise EvalContractError(
            f"Could not allocate a {n}-row subset across {len(categories)} categories"
        )

    selected: List[Dict[str, Any]] = []
    for category in categories:
        ranked = sorted(by_category[category], key=lambda row: _rank_key(seed, row))
        selected.extend(ranked[: quotas[category]])
    selected.sort(key=lambda row: int(row["_benchmark_index"]))

    selected_ids = [str(row["id"]) for row in selected]
    if len(selected_ids) != n:
        raise EvalContractError(f"Subset selected {len(selected_ids)} rows, expected {n}")
    if len(set(selected_ids)) != n:
        raise EvalContractError("Subset contains duplicate problem ids")
    for item in selected:
        item.pop("_benchmark_index", None)

    return selected, {
        "method": "proportional_stratified_sha256",
        "seed": seed,
        "n": n,
        "available": total,
        "categories": categories,
        "quotas": quotas,
        "selected_ids": selected_ids,
    }


def expected_cell_keys(
    problems: Sequence[Dict[str, Any]],
    arms: Sequence[str],
    budgets: Sequence[int],
) -> Set[str]:
    keys: Set[str] = set()
    for problem in problems:
        problem_id = str(problem["id"])
        for arm in arms:
            for budget in budgets:
                keys.add(f"{problem_id}|{arm}|{int(budget)}")
    return keys


def verify_completed_cells(expected: Set[str], completed: Set[str]) -> None:
    missing = sorted(expected - completed)
    extra = sorted(completed - expected)
    if missing or extra:
        pieces = []
        if missing:
            pieces.append(f"{len(missing)} missing (first: {missing[:3]})")
        if extra:
            pieces.append(f"{len(extra)} unexpected (first: {extra[:3]})")
        raise EvalContractError(
            "Evaluation grid is incomplete or contaminated: " + "; ".join(pieces)
        )


def _subset_identity(subset: Any) -> Optional[Dict[str, Any]]:
    """Return the provenance-bearing fields of a subset descriptor."""
    if subset is None:
        return None
    if not isinstance(subset, dict):
        raise EvalContractError("Manifest subset descriptor must be an object or null")
    selected_ids = subset.get("selected_ids")
    if selected_ids is not None:
        if not isinstance(selected_ids, list):
            raise EvalContractError("Manifest subset selected_ids must be a list")
        selected_ids = [str(value) for value in selected_ids]
    return {
        "method": subset.get("method"),
        "seed": None if subset.get("seed") is None else str(subset.get("seed")),
        "n": subset.get("n"),
        "selected_ids": selected_ids,
    }


def validate_resume_manifest(
    out_dir: str | Path,
    manifest: Dict[str, Any],
    replace_arms: Optional[Sequence[str]] = None,
) -> None:
    """Refuse to resume a checkpoint directory under a different contract.

    A JSONL checkpoint is only meaningful together with the exact dataset,
    adapter, prompt/scorer code, subset, and run kind that produced it.  This
    check runs *before* the new manifest overwrites the old one.  A checkpoint
    without a manifest, or a manifest without contract hashes, is not resumable.
    """
    manifest_path = Path(out_dir) / "EVAL_RUN_MANIFEST.json"
    results_path = Path(out_dir) / "robust_eval_results.jsonl"
    if not results_path.is_file():
        # A manifest without a checkpoint is a failed preflight, not a run;
        # it can be replaced by a fresh attempt.
        return
    if not manifest_path.is_file():
        raise EvalContractError(
            "Refusing to resume: a checkpoint exists without a run manifest"
        )
    try:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvalContractError(
            f"Cannot read existing run manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(old, dict):
        raise EvalContractError(f"Existing run manifest is not an object: {manifest_path}")

    for field in ("dataset_sha256", "adapter_sha256"):
        old_value = old.get(field)
        new_value = manifest.get(field)
        old_text = "" if old_value is None else str(old_value).upper()
        new_text = "" if new_value is None else str(new_value).upper()
        if old_text != new_text:
            raise EvalContractError(
                f"Refusing to resume: {field} changed "
                f"({old_value!r} -> {new_value!r})"
            )

    for field in ("base_model", "base_revision", "run_kind", "truncation_rule"):
        if old.get(field) != manifest.get(field):
            raise EvalContractError(
                f"Refusing to resume: {field} changed "
                f"({old.get(field)!r} -> {manifest.get(field)!r})"
            )

    if old.get("strict_scoring") is not True or manifest.get("strict_scoring") is not True:
        raise EvalContractError(
            "Refusing to resume: both the existing and new run must be strict-scored"
        )

    if _subset_identity(old.get("subset")) != _subset_identity(manifest.get("subset")):
        raise EvalContractError(
            "Refusing to resume: the diagnostic subset changed"
        )

    old_budgets = [int(value) for value in old.get("budgets", [])]
    new_budgets = [int(value) for value in manifest.get("budgets", [])]
    if old_budgets != new_budgets:
        raise EvalContractError(
            f"Refusing to resume: budgets changed ({old_budgets} -> {new_budgets})"
        )
    if not replace_arms:
        old_arms = sorted(str(value) for value in old.get("arms", []))
        new_arms = sorted(str(value) for value in manifest.get("arms", []))
        if old_arms != new_arms:
            raise EvalContractError(
                f"Refusing to resume: arms changed ({old_arms} -> {new_arms})"
            )

    old_contracts = old.get("contracts")
    new_contracts = manifest.get("contracts")
    if not isinstance(old_contracts, dict) or not old_contracts:
        raise EvalContractError(
            "Refusing to resume: the existing manifest has no contract hashes"
        )
    if not isinstance(new_contracts, dict) or not new_contracts:
        raise EvalContractError("New manifest is missing contract hashes")
    for field in (
        "evaluator_sha256",
        "run_contract_sha256",
        "strict_scorer_sha256",
        "prompts_sha256",
        "ground_truth_overrides_sha256",
    ):
        if old_contracts.get(field) != new_contracts.get(field):
            raise EvalContractError(
                f"Refusing to resume: {field} changed "
                f"({old_contracts.get(field)!r} -> {new_contracts.get(field)!r})"
            )
