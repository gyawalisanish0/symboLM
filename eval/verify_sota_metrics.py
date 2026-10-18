"""
Standalone Verification Script for SOTA Reasoning & Overthinking Metrics
========================================================================
Strictly implements published mathematical formulas from:
1. Meta FAIR & CMU (arXiv:2508.13141): OptimalThinkingBench (OAA_t, AUC_OAA)
2. ACL 2025: LLMTHINKBENCH (Harmonic Overthinking Score O, Token Efficiency)
3. Hardware Power Audit: Joules per correct solve (J/correct)
4. Budget-Choking Cliff Factor (BCF)

Author: Sanish Gyawali (Rupandehi, Nepal)
Zero synthetic numbers. Strictly consumes verified evaluation JSONL receipts.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

ARM_LABELS = {
    "A": "Arm A: Ordinary CoT",
    "B": "Arm B: Answer-Only Reflex",
    "C": "Arm C: Full Registers",
    "D": "Arm D: Corrupted Router Control",
    "E": "Arm E: Deterministic REPL Sandbox",
    "F": "Arm F: Vanilla DeepSeek Base",
}

def verify_and_compute(results_jsonl: Path) -> Dict[str, Any]:
    if not results_jsonl.exists():
        raise FileNotFoundError(f"Receipts file not found: {results_jsonl}")

    records: List[Dict[str, Any]] = []
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError("Receipts file is empty. No evaluated records found.")

    arms = sorted(set(r["arm"] for r in records))
    budgets = sorted(set(r["budget"] for r in records))

    print("=" * 85)
    print(f"  SOTA REASONING METRICS VERIFICATION (Meta FAIR & CMU arXiv:2508.13141)")
    print(f"  Input: {results_jsonl.name} | Records Loaded: {len(records)}")
    print(f"  Arms: {arms} | Budgets: {budgets}")
    print("=" * 85)

    summary_table = []
    thresholds = [32, 64, 128, 256, 512, 1024, 2048]

    for arm in arms:
        arm_recs = [r for r in records if r["arm"] == arm]
        
        # Accuracy across budgets for Budget-Choking Factor
        acc_by_budget = {}
        for b in budgets:
            b_recs = [r for r in arm_recs if r["budget"] == b]
            if b_recs:
                acc_by_budget[b] = sum(1 for r in b_recs if r["is_correct"]) / len(b_recs)

        acc_256 = acc_by_budget.get(256, 0.0)
        acc_2048 = acc_by_budget.get(2048, acc_by_budget.get(max(budgets), 0.0))
        
        # Budget-Choking Factor (BCF)
        if acc_2048 > 1e-6:
            bcf = round((acc_2048 - acc_256) / acc_2048, 4)
        else:
            bcf = 0.0 if acc_256 <= 1e-6 else -1.0

        for b in budgets:
            cell_recs = [r for r in arm_recs if r["budget"] == b]
            if not cell_recs:
                continue

            n = len(cell_recs)
            correct_recs = [r for r in cell_recs if r["is_correct"]]
            acc = len(correct_recs) / n
            mean_tokens = sum(r["tokens_emitted"] for r in cell_recs) / n
            total_j = sum(r.get("dynamic_joules", 0.0) for r in cell_recs)
            j_per_correct = total_j / len(correct_recs) if correct_recs else float("inf")

            # Eq. 1: OAA_t
            oaa = {0: 0.0}
            for t in thresholds:
                oaa[t] = sum(1 for r in correct_recs if r["tokens_emitted"] <= t) / n

            # Eq. 2: AUC_OAA (trapezoidal integration over [0, max(thresholds)])
            t_sorted = sorted(oaa.keys())
            auc_sum = 0.0
            for i in range(len(t_sorted) - 1):
                t1, t2 = t_sorted[i], t_sorted[i+1]
                auc_sum += 0.5 * (oaa[t1] + oaa[t2]) * (t2 - t1)
            norm_auc = auc_sum / t_sorted[-1] if t_sorted[-1] > 0 else 0.0

            # ACL 2025: Harmonic Overthinking Score (O)
            tok_eff = min(1.0, 256.0 / max(mean_tokens, 1.0))
            o_score = (2 * acc * tok_eff / (acc + tok_eff)) if (acc + tok_eff) > 0 else 0.0

            summary_table.append({
                "arm": arm,
                "label": ARM_LABELS.get(arm, arm),
                "budget": b,
                "n": n,
                "acc": acc,
                "mean_tokens": mean_tokens,
                "j_per_correct": j_per_correct,
                "oaa_256": oaa.get(256, 0.0),
                "auc_oaa": norm_auc,
                "token_eff": tok_eff,
                "overthinking_score": o_score,
                "bcf": bcf,
            })

    # Print Table
    print(f"\n{'ARM':<32} | {'BUDGET':<6} | {'PASS@1':<7} | {'MEAN TOK':<9} | {'J/CORRECT':<10} | {'OAA@256':<7} | {'AUC_OAA':<7} | {'OVERTHINK SCORE':<15} | {'CHOKE (BCF)':<11}")
    print("-" * 125)
    for row in summary_table:
        j_str = f"{row['j_per_correct']:.1f} J" if not math.isinf(row['j_per_correct']) else "inf"
        print(
            f"{row['label']:<32} | "
            f"{row['budget']:<6} | "
            f"{row['acc']*100:>6.1f}% | "
            f"{row['mean_tokens']:>8.1f}t | "
            f"{j_str:>10} | "
            f"{row['oaa_256']*100:>6.1f}% | "
            f"{row['auc_oaa']:>7.3f} | "
            f"{row['overthinking_score']:>15.3f} | "
            f"{row['bcf']:>11.3f}"
        )
    print("=" * 125)
    return {"records_verified": len(records), "metrics": summary_table}

def main():
    parser = argparse.ArgumentParser(description="Verify SOTA Reasoning & Overthinking Metrics from Receipts")
    parser.add_argument("results_file", type=Path, help="Path to robust_eval_results.jsonl")
    args = parser.parse_args()
    verify_and_compute(args.results_file)

if __name__ == "__main__":
    main()
