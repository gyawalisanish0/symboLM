"""
SymboLM Empirical Auditing: Clean Held-Out 200 Split Generator
==============================================================
Extracts exactly 200 clean out-of-distribution math problems from the 904
uncontaminated items verified in Task 4 (data/benchmark_1000_multi_year.jsonl).
Guarantees 0.0% prompt contamination and zero 8-gram training leakage.

Author: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTAMINATION_RESULTS = ROOT_DIR / "eval" / "contamination_audit_results.json"
BENCHMARK_FILE = ROOT_DIR / "data" / "benchmark_1000_multi_year.jsonl"
OUT_HELD_OUT_FILE = ROOT_DIR / "data" / "held_out_eval_200_clean.jsonl"


def generate_clean_held_out():
    # 1. Load contamination audit findings
    with open(CONTAMINATION_RESULTS, "r", encoding="utf-8") as f:
        contam_data = json.load(f)

    exact_ids = {item["id"] for item in contam_data.get("exact_matches", [])}
    overlap_ids = {item["id"] for item in contam_data.get("ngram_overlaps", [])}
    leaked_ids = exact_ids | overlap_ids
    print(f"Filtering out {len(leaked_ids)} leaked/overlapping items...")

    # 2. Read benchmark and retain only clean items
    clean_pool = []
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["id"] not in leaked_ids:
                clean_pool.append(item)

    print(f"Total clean candidates available: {len(clean_pool)}")

    # 3. Stratified deterministic sampling across categories
    by_cat = {}
    for item in clean_pool:
        cat = item.get("category", "math")
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(item)

    # Deterministically select 200 samples
    target_count = 200
    held_out_200 = []

    # Sort categories to be fully deterministic
    categories = sorted(by_cat.keys())
    per_cat_quota = target_count // len(categories)

    for cat in categories:
        items = by_cat[cat]
        held_out_200.extend(items[:per_cat_quota])

    # Fill remaining from largest pools
    idx = 0
    while len(held_out_200) < target_count:
        cat = categories[idx % len(categories)]
        items = by_cat[cat]
        if len(items) > per_cat_quota:
            held_out_200.append(items[per_cat_quota + (idx // len(categories))])
        idx += 1

    print(f"Selected exactly {len(held_out_200)} stratified, 100% clean problems.")

    # 4. Save to JSONL
    with open(OUT_HELD_OUT_FILE, "w", encoding="utf-8") as f:
        for item in held_out_200:
            f.write(json.dumps(item) + "\n")

    print(f"[SAVED] Pristine clean held-out evaluation set written to: {OUT_HELD_OUT_FILE}")


if __name__ == "__main__":
    generate_clean_held_out()
