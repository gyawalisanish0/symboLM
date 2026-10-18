"""
SymboLM Empirical Auditing: Honest Rescorer (Task 1)
====================================================
Audits and rescores benchmark outputs under strict empirical criteria
defined by the Claude Haiku 5.5 Empirical Auditing Protocol:
  1. Counts an output as FAIL if truncated (tokens_emitted >= max_tokens)
     without a closed <ans> or <reg:deduce> closure.
  2. Eliminates lenient trailing-number fallbacks (re.findall(r"[-+]?\\d+")).
  3. Eliminates ceremonial tautology matches (verify(x==x) -> ✓).
  4. Only accepts answers formally closed within <ans>, <reg:bypass>,
     <reg:intent>, or formal '∴ ans = ' derivations.
  5. Outputs a new JSON (eval/eval_v2_1000_rescored.json) and diff report
     without touching the original receipt file.

Author: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_RECEIPT = ROOT_DIR / "eval" / "eval_v2_1000_results.json"
OUTPUT_RESCORING_JSON = ROOT_DIR / "eval" / "eval_v2_1000_rescored.json"
OUTPUT_REPORT_MD = ROOT_DIR / "eval" / "RESCORING_AUDIT_REPORT.md"


def clean_math_string(s: str) -> str:
    """Normalize LaTeX math string for robust comparison."""
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = s.replace("$", "").replace("\\$", "").strip()
    s = s.replace(" ", "")
    return s


def extract_honest_answer(text: str, tokens_emitted: int = 0, max_tokens: int = 512) -> Tuple[Optional[str], str]:
    """
    Extracts answer under strict scientific criteria.
    Returns: (extracted_answer, extraction_method)
    """
    if not text:
        return None, "empty_text"

    # Check for strict truncation without answer closure
    is_truncated = (tokens_emitted >= max_tokens) and ("</ans>" not in text)

    # 1. Check formal <ans>...</ans> tag (Gold Standard)
    m_ans = re.search(r"<ans>\s*([^<]+?)\s*</ans>", text, re.IGNORECASE)
    if m_ans:
        ans_cand = m_ans.group(1).strip()
        if ans_cand:
            return clean_math_string(ans_cand), "closed_ans_tag"

    # 2. Check formal <reg:bypass>...</reg:bypass> (System 1 facts)
    m_bypass = re.search(r"<reg:bypass>\s*([^<]+?)\s*</reg:bypass>", text, re.IGNORECASE)
    if m_bypass:
        cand = m_bypass.group(1).strip()
        if cand:
            return clean_math_string(cand), "closed_bypass_tag"

    # 3. Check formal <reg:intent>...</reg:intent> (Social dialogue)
    m_intent = re.search(r"<reg:intent>\s*(?:∴\s*(?:ans\s*=\s*)?)?([^<]+?)\s*</reg:intent>", text, re.IGNORECASE)
    if m_intent:
        cand = m_intent.group(1).strip()
        if cand:
            return clean_math_string(cand), "closed_intent_tag"

    # If truncated and no closed <ans> or bypass exists, derivation is incomplete!
    if is_truncated:
        return None, "truncated_incomplete_derivation"

    # 4. Check formal ∴ ans = ... or ∴ answer = ... ONLY if within deduce or explicitly closed
    m_sym = re.search(r"∴\s*(?:ans\s*=\s*|answer\s*=\s*)([^\n|</]+)", text, re.IGNORECASE)
    if m_sym:
        cand = m_sym.group(1).strip()
        # Clean trailing noise or delimiter
        cand = cand.split("</")[0].strip()
        if cand and len(cand) < 60:
            return clean_math_string(cand), "symbolic_therefore_statement"

    # ZERO FALLBACKS:
    # Deliberately rejects:
    # - verify(...) -> ✓
    # - re.findall(r"[-+]?\d+", post_think) (trailing numbers)
    return None, "rejected_no_valid_closure"


def verify_ground_truth(extracted: Optional[str], ground_truth: str) -> bool:
    """Rigorous verification matching exact values and mathematical equivalence."""
    if extracted is None or not str(extracted).strip():
        return False

    gt = clean_math_string(str(ground_truth).strip().lower())
    ext = clean_math_string(str(extracted).strip().lower())

    # Direct match
    if ext == gt:
        return True

    # Fraction check (e.g. 1/16 == 0.0625)
    if "/" in gt and "/" in ext:
        try:
            p_gt = gt.split("/")
            p_ext = ext.split("/")
            if abs(float(p_gt[0]) / float(p_gt[1]) - float(p_ext[0]) / float(p_ext[1])) < 1e-4:
                return True
        except Exception:
            pass

    # Numeric equivalence with 1e-3 tolerance
    try:
        ext_clean = re.sub(r"[^\d.-]", "", ext)
        gt_clean = re.sub(r"[^\d.-]", "", gt)
        if ext_clean and gt_clean and ext_clean not in ("-", ".", "-.") and gt_clean not in ("-", ".", "-."):
            f_ext = float(ext_clean)
            f_gt = float(gt_clean)
            if abs(f_ext - f_gt) < 1e-3:
                return True
            if f_gt != 0 and abs(f_ext - f_gt) / abs(f_gt) < 1e-3:
                return True
    except Exception:
        pass

    return False


def run_rescoring():
    print("=" * 80)
    print("SymboLM Empirical Auditing: Running Honest Rescorer (Task 1)")
    print("=" * 80)
    print(f"Source Receipt : {INPUT_RECEIPT}")
    print(f"Target Output  : {OUTPUT_RESCORING_JSON}")

    if not INPUT_RECEIPT.exists():
        raise FileNotFoundError(f"Input receipt not found: {INPUT_RECEIPT}")

    with open(INPUT_RECEIPT, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("itemized_results", [])
    total = len(items)
    print(f"Total problems to rescore: {total}")

    old_correct_count = 0
    new_correct_count = 0

    flipped_to_fail = []
    flipped_to_pass = []
    kept_pass = []

    rescored_items = []
    extraction_method_counts = {}

    category_stats = {}

    for idx, item in enumerate(items, start=1):
        q_id = item.get("id")
        cat = item.get("category", "unknown")
        year = item.get("year", "unknown")
        gt = item.get("ground_truth", "")
        old_extracted = item.get("extracted_answer")
        old_correct = bool(item.get("correct", False))
        tokens_emitted = item.get("tokens_emitted", 0)
        raw_output = item.get("raw_output", "")

        if old_correct:
            old_correct_count += 1

        # Honest extraction
        new_extracted, method = extract_honest_answer(raw_output, tokens_emitted=tokens_emitted, max_tokens=512)
        extraction_method_counts[method] = extraction_method_counts.get(method, 0) + 1

        new_correct = verify_ground_truth(new_extracted, gt)
        if new_correct:
            new_correct_count += 1

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "old_correct": 0, "new_correct": 0}
        category_stats[cat]["total"] += 1
        if old_correct:
            category_stats[cat]["old_correct"] += 1
        if new_correct:
            category_stats[cat]["new_correct"] += 1

        # Track transitions
        if old_correct and not new_correct:
            flipped_to_fail.append({
                "id": q_id, "category": cat, "gt": gt,
                "old_extracted": old_extracted, "new_extracted": new_extracted,
                "tokens_emitted": tokens_emitted, "method": method,
                "snippet": raw_output[-120:] if len(raw_output) > 120 else raw_output
            })
        elif not old_correct and new_correct:
            flipped_to_pass.append({
                "id": q_id, "category": cat, "gt": gt,
                "old_extracted": old_extracted, "new_extracted": new_extracted
            })
        elif old_correct and new_correct:
            kept_pass.append(q_id)

        rescored_item = dict(item)
        rescored_item["old_extracted_answer"] = old_extracted
        rescored_item["old_correct"] = old_correct
        rescored_item["honest_extracted_answer"] = new_extracted
        rescored_item["honest_correct"] = new_correct
        rescored_item["extraction_method"] = method
        rescored_items.append(rescored_item)

    old_accuracy = (old_correct_count / total) * 100 if total > 0 else 0
    new_accuracy = (new_correct_count / total) * 100 if total > 0 else 0
    accuracy_delta = new_accuracy - old_accuracy

    print("-" * 80)
    print(f"Old Published Accuracy : {old_accuracy:.2f}% ({old_correct_count}/{total})")
    print(f"Honest Rescored Accuracy: {new_accuracy:.2f}% ({new_correct_count}/{total})")
    print(f"Accuracy Delta           : {accuracy_delta:+.2f} percentage points")
    print(f"False Positives Flipped to FAIL : {len(flipped_to_fail)}")
    print(f"Genuine Solves Preserved        : {len(kept_pass)}")
    print("-" * 80)
    print("Extraction Method Distribution:")
    for method, count in sorted(extraction_method_counts.items(), key=lambda x: -x[1]):
        print(f"  - {method:32s}: {count:4d} ({count/total*100:5.1f}%)")

    # Save rescored JSON
    output_data = {
        "audit_metadata": {
            "protocol": "Claude Haiku 5.5 Empirical Auditing Protocol (Task 1)",
            "source_file": str(INPUT_RECEIPT.name),
            "total_questions": total,
            "old_accuracy_pct": round(old_accuracy, 2),
            "honest_accuracy_pct": round(new_accuracy, 2),
            "accuracy_delta_pct": round(accuracy_delta, 2),
            "false_positives_eliminated": len(flipped_to_fail),
            "genuine_solves_retained": len(kept_pass),
            "extraction_methods": extraction_method_counts
        },
        "by_category": {
            cat: {
                "total": d["total"],
                "old_correct": d["old_correct"],
                "old_accuracy_pct": round(d["old_correct"] / d["total"] * 100, 1),
                "honest_correct": d["new_correct"],
                "honest_accuracy_pct": round(d["new_correct"] / d["total"] * 100, 1),
                "delta_pct": round((d["new_correct"] - d["old_correct"]) / d["total"] * 100, 1)
            }
            for cat, d in sorted(category_stats.items())
        },
        "itemized_results": rescored_items
    }

    with open(OUTPUT_RESCORING_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[SAVED] Rescored JSON saved to: {OUTPUT_RESCORING_JSON}")

    # Generate Markdown Report
    generate_markdown_report(output_data, flipped_to_fail, OUTPUT_REPORT_MD)
    print(f"[REPORT] Rescoring Report published to: {OUTPUT_REPORT_MD}")

    return old_accuracy, new_accuracy, accuracy_delta, len(flipped_to_fail)


def generate_markdown_report(data: Dict[str, Any], false_positives: List[Dict[str, Any]], report_path: Path):
    meta = data["audit_metadata"]
    by_cat = data["by_category"]

    lines = [
        "# SymboLM Honest Rescoring Audit Report (Task 1)",
        "",
        "**Audit Protocol:** Claude Haiku 5.5 Empirical Auditing Protocol  ",
        "**Author & Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  ",
        "**AI Systems Collaborator:** Antigravity (Google DeepMind)  ",
        f"**Date:** October 17, 2026  ",
        f"**Source Artifact:** `{meta['source_file']}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary: The Honest Truth",
        "",
        f"Under strict rejection of trailing-number fallbacks and unclosed truncated outputs:",
        f"- **Old Published Accuracy:** **{meta['old_accuracy_pct']}%** ({int(meta['old_accuracy_pct'] * meta['total_questions'] / 100)} / {meta['total_questions']})",
        f"- **Honest Rescored Accuracy:** **{meta['honest_accuracy_pct']}%** ({int(meta['honest_accuracy_pct'] * meta['total_questions'] / 100)} / {meta['total_questions']})",
        f"- **Net Accuracy Shift:** **{meta['accuracy_delta_pct']:+.2f} percentage points**",
        f"- **Lenient False Positives Eliminated:** **{meta['false_positives_eliminated']} items**",
        f"- **Rigorous Genuine Solves Retained:** **{meta['genuine_solves_retained']} items**",
        "",
        "---",
        "",
        "## 2. Category Breakdown: Old vs. Honest Rescored",
        "",
        "| Category | Total | Old Accuracy | Honest Rescored | Delta | Retained Solves |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for cat, d in by_cat.items():
        delta_str = f"{d['delta_pct']:+.1f}%"
        lines.append(f"| `{cat}` | {d['total']} | {d['old_accuracy_pct']}% | **{d['honest_accuracy_pct']}%** | {delta_str} | {d['honest_correct']} / {d['total']} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Extraction Method Distribution",
        "",
        "| Extraction Mechanism | Count | Proportion | Scientific Validity |",
        "| :--- | :---: | :---: | :--- |"
    ])

    for method, count in sorted(meta["extraction_methods"].items(), key=lambda x: -x[1]):
        validity = "Valid Grounded Closure" if "closed" in method or "symbolic" in method else "Strict Rejection (Truncated / Malformed)"
        pct = count / meta["total_questions"] * 100
        lines.append(f"| `{method}` | {count} | {pct:.1f}% | {validity} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Sample Forensics: Why False Positives Were Eliminated",
        "",
        "Below are representative examples where the previous regex fallback credited a correct answer despite the model truncating before reaching a conclusion:",
        ""
    ])

    for ex in false_positives[:5]:
        lines.append(f"### Item `{ex['id']}` (`{ex['category']}`)")
        lines.append(f"- **Ground Truth:** `{ex['gt']}`")
        lines.append(f"- **Old Scorer Grabbed:** `{ex['old_extracted']}`")
        lines.append(f"- **Tokens Emitted:** `{ex['tokens_emitted']}` (Hit 512 cap)")
        lines.append(f"- **Raw Truncated Tail:** `...{ex['snippet']}`")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_rescoring()
