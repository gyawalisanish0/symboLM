"""
Build the canonical SymboLM SFT split for the rescue experiment.
===============================================================

Source: ``data/concept_math_contrastive_v2_1.jsonl``

Transformations
---------------
* Keep canonical reasoning targets that end in ``<ans>...</ans>``.
* For contrastive-negative rows, keep the **final corrected branch** (the trace
  after the backtrack), not the deliberately wrong branch.  This gives the model
  a clean canonical target while preserving the corrected arithmetic.
* Keep atomic ``<reg:bypass>...</reg:bypass>`` targets.
* Drop dialogue/intent rows: they are not part of the math/edge thesis.
* Split 90/10 into train/val, stratified by target kind.
* Write a manifest with source/output hashes and exact filter counts.

Usage::

    python scripts/prepare_research_sft.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbolic.prompts import (  # noqa: E402
    REGISTER_SYSTEM_PROMPT,
    classify_target_body,
    normalize_target_body,
)

DEFAULT_INPUT = ROOT / "data" / "concept_math_contrastive_v2_1.jsonl"
DEFAULT_OUT = ROOT / "data" / "research_sft"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def canonicalize_contrastive(full_target: str) -> Optional[str]:
    """
    Convert a wrong-branch -> backtrack -> corrected-branch target into a
    canonical plan/deduce/verify/think/ans target.

    Uses the first plan, the last deduce, the last verify, and the last answer.
    """
    plans = re.findall(r"<reg:plan>.*?</reg:plan>", full_target, re.DOTALL)
    deduces = re.findall(r"<reg:deduce>.*?</reg:deduce>", full_target, re.DOTALL)
    verifies = re.findall(r"<reg:verify>.*?</reg:verify>", full_target, re.DOTALL)
    answers = re.findall(r"<ans>.*?</ans>", full_target, re.DOTALL)

    if not plans or not deduces or not answers:
        return None

    plan = plans[0].strip()
    deduce = deduces[-1].strip()
    verify = verifies[-1].strip() if verifies else ""
    answer = answers[-1].strip()

    # The corrected branch begins with "replan:"; remove the marker so the
    # final trace reads like a normal canonical deduction.
    deduce = re.sub(r"(<reg:deduce>)\s*replan:\s*", r"\1", deduce)

    target = f"<think>\n{plan}\n{deduce}"
    if verify:
        target += f"\n{verify}"
    target += f"\n</think>\n{answer}"
    return target


def transform_record(
    record: Dict[str, Any], canonicalize: bool
) -> Optional[Dict[str, Any]]:
    full_target = record.get("full_target", "")
    body = normalize_target_body(full_target)
    kind, reason = classify_target_body(body)

    if kind == "dialogue":
        return None
    if kind == "contrastive":
        if not canonicalize:
            return None
        corrected = canonicalize_contrastive(body)
        if not corrected:
            return None
        full_target = corrected
        body = normalize_target_body(corrected)
        kind, reason = classify_target_body(body)

    if kind not in {"reasoning", "bypass"}:
        return None

    return {
        "id": record.get("id"),
        "source_id": record.get("id"),
        "prompt": record.get("prompt", ""),
        "full_target": full_target,
        "system_prompt": REGISTER_SYSTEM_PROMPT,
        "ground_truth": record.get("ground_truth", ""),
        "type": record.get("type", ""),
        "concept": record.get("concept", ""),
        "target_kind": kind,
        "source": "concept_math_contrastive_v2_1",
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare canonical SymboLM SFT data")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-canonicalize-contrastive",
        action="store_true",
        help="Drop contrastive-negative rows instead of keeping their corrected branch.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not (0.0 < args.val_fraction < 0.5):
        raise ValueError("--val-fraction must be between 0 and 0.5")

    raw = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    filters: Counter = Counter()
    transformed: List[Dict[str, Any]] = []
    seen_ids = set()
    renamed_duplicate_ids = 0

    for source_index, record in enumerate(raw):
        result = transform_record(record, canonicalize=not args.no_canonicalize_contrastive)
        if result is None:
            body = normalize_target_body(record.get("full_target", ""))
            kind, _ = classify_target_body(body)
            filters[kind] += 1
            continue
        # The source contains 350 legitimate problems whose IDs collide with a
        # different problem.  Keep both and make the training ID unique; the
        # original source ID is retained for provenance.
        source_id = result["id"]
        unique_id = source_id
        if unique_id in seen_ids:
            unique_id = f"{source_id}#{source_index}"
            renamed_duplicate_ids += 1
        seen_ids.add(unique_id)
        result["id"] = unique_id
        result["source_index"] = source_index
        transformed.append(result)

    if not transformed:
        raise ValueError("No usable records after filtering; refusing to write an empty split.")

    random.Random(args.seed).shuffle(transformed)

    by_kind: Dict[str, List[Dict[str, Any]]] = {"reasoning": [], "bypass": []}
    for row in transformed:
        by_kind[row["target_kind"]].append(row)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for kind, rows in by_kind.items():
        n_val = max(1, round(len(rows) * args.val_fraction)) if len(rows) > 1 else 0
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])

    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(val)

    for row in train:
        row["split"] = "train"
    for row in val:
        row["split"] = "val"

    counts = {
        "source_records": len(raw),
        "kept": len(transformed),
        "train": len(train),
        "val": len(val),
        "target_kinds": dict(Counter(row["target_kind"] for row in transformed)),
        "filtered": dict(filters),
        "renamed_duplicate_ids": renamed_duplicate_ids,
        "canonicalize_contrastive": not args.no_canonicalize_contrastive,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
    }

    if args.dry_run:
        print(json.dumps(counts, indent=2))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "source": str(args.input),
        "source_sha256": sha256_file(args.input),
        "train": str(train_path),
        "train_sha256": sha256_file(train_path),
        "val": str(val_path),
        "val_sha256": sha256_file(val_path),
        "system_prompt": REGISTER_SYSTEM_PROMPT,
        **counts,
    }
    manifest_path = args.out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Canonical SFT split written")
    print(f"  train: {train_path}  ({len(train):,} rows)")
    print(f"  val  : {val_path}  ({len(val):,} rows)")
    print(f"  manifest: {manifest_path}")
    print(f"  counts: {json.dumps(counts, indent=2)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
