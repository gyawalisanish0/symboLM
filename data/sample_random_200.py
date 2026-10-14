"""
Sample 200 Random Multi-Year Questions for SymboLM v2 Benchmark Evaluation
===========================================================================
Draws a stratified random sample of 200 questions (40 items/year from 2021 to 2026)
from `data/benchmark_1000_multi_year.jsonl`.

Author: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parent.parent
IN_FILE = ROOT_DIR / "data" / "benchmark_1000_multi_year.jsonl"
OUT_FILE = ROOT_DIR / "data" / "eval_200_random_multi_year.jsonl"
CHECKSUMS_FILE = ROOT_DIR / "data" / "CHECKSUMS.sha256"


def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {IN_FILE}")

    # Read items grouped by year
    by_year = defaultdict(list)
    with open(IN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            year = item.get("year", 2021)
            by_year[year].append(item)

    print(f"Loaded {sum(len(v) for v in by_year.values())} total questions across {len(by_year)} years.")

    # Stratified random sample of 40 per year = 200 total
    rng = random.Random(42)
    sampled_items = []
    years = sorted(by_year.keys())
    for yr in years:
        items = by_year[yr]
        k = min(40, len(items))
        sampled = rng.sample(items, k)
        sampled_items.extend(sampled)
        print(f"  • Year {yr}: sampled {len(sampled)} / {len(items)} items")

    # Shuffle to interleave years randomly
    rng.shuffle(sampled_items)
    print(f"Total sampled: {len(sampled_items)} questions.")

    # Ensure each item has category and answer fields expected by evaluator
    for item in sampled_items:
        if "category" not in item:
            item["category"] = item.get("source_benchmark", f"Year-{item.get('year', 2021)}")
        if "answer" not in item:
            item["answer"] = item.get("ground_truth", "")

    # Write output
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for s in sampled_items:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Compute SHA-256
    with open(OUT_FILE, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"[OK] Saved {len(sampled_items)} items to {OUT_FILE}")
    print(f"[OK] SHA-256: {sha256}")

    # Update CHECKSUMS.sha256
    lines = []
    if CHECKSUMS_FILE.exists():
        lines = [ln for ln in CHECKSUMS_FILE.read_text(encoding="utf-8").splitlines() if "eval_200_random_multi_year.jsonl" not in ln]
    lines.append(f"{sha256}  data/eval_200_random_multi_year.jsonl")
    CHECKSUMS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Updated checksums in {CHECKSUMS_FILE}")


if __name__ == "__main__":
    main()
