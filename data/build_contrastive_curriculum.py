"""
SymboLM Contrastive Curriculum Generator (v2.1)
==============================================
Builds `data/concept_math_contrastive_v2_1.jsonl` by reformulating concept
math problems into Unit-Step AST operations and injecting 35% contrastive
negative traces with backtrack verification (`checkpoint → ✗ → backtrack → replan → ✓`).

Author: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
IN_FILE = ROOT_DIR / "data" / "concept_math_curriculum.jsonl"
OUT_FILE = ROOT_DIR / "data" / "concept_math_contrastive_v2_1.jsonl"
CHECKSUMS_FILE = ROOT_DIR / "data" / "CHECKSUMS.sha256"


def build_polynomial_contrastive(item: Dict[str, Any]) -> str:
    plan = item.get("canonical_plan", "")
    deduce = item.get("canonical_deduce", "")
    gt = item.get("ground_truth", "")
    prompt = item.get("prompt", "")

    # Extract base u
    match = re.search(r"x \+ 1/x = (\d+)", prompt)
    u_val = match.group(1) if match else "6"
    u_int = int(u_val)
    wrong_u2 = u_int ** 2

    return (
        f"<think>\n"
        f"{plan}\n"
        f"<checkpoint:CP1>\n"
        f"<reg:deduce>hyp [assume x²+1/x² = u² = {wrong_u2}]</reg:deduce>\n"
        f"<reg:verify>verify({wrong_u2} == {u_val}^2 - 2)✗ [failed: cross-term 2 omitted]</reg:verify>\n"
        f"<backtrack:CP1>\n"
        f"<reg:deduce>replan: {deduce.replace('<reg:deduce>', '').replace('</reg:deduce>', '').strip()}</reg:deduce>\n"
        f"<reg:verify>verify(ans == {gt})✓</reg:verify>\n"
        f"</think>\n"
        f"<ans>{gt}</ans>"
    )


def build_crt_contrastive(item: Dict[str, Any]) -> str:
    plan = item.get("canonical_plan", "")
    deduce = item.get("canonical_deduce", "")
    gt = item.get("ground_truth", "")
    prompt = item.get("prompt", "")

    # Extract modulos and remainders
    # e.g., n ≡ r1 (mod m1) and n ≡ r2 (mod m2)
    matches = re.findall(r"n ≡ (\d+) \(mod (\d+)\)", prompt)
    if matches and len(matches) >= 2:
        r1, m1 = int(matches[0][0]), int(matches[0][1])
        r2, m2 = int(matches[1][0]), int(matches[1][1])
        guess = r1 + m1
        if guess % m2 == r2:
            guess = guess + m1  # ensure it fails constraint 2
        return (
            f"<think>\n"
            f"{plan}\n"
            f"<checkpoint:CP1>\n"
            f"<reg:deduce>hyp [guess n = r1 + m1 = {guess}]</reg:deduce>\n"
            f"<reg:verify>verify({guess} % {m2} == {r2})✗ [mismatch: {guess % m2} != {r2}]</reg:verify>\n"
            f"<backtrack:CP1>\n"
            f"<reg:deduce>replan: {deduce.replace('<reg:deduce>', '').replace('</reg:deduce>', '').strip()}</reg:deduce>\n"
            f"<reg:verify>verify({gt} % {m1} == {r1} and {gt} % {m2} == {r2})✓</reg:verify>\n"
            f"</think>\n"
            f"<ans>{gt}</ans>"
        )

    # General fallback for CRT
    return (
        f"<think>\n"
        f"{plan}\n"
        f"<checkpoint:CP1>\n"
        f"<reg:deduce>hyp [guess direct sum without modular inverse]</reg:deduce>\n"
        f"<reg:verify>verify(all modular conditions satisfied)✗ [condition check failed]</reg:verify>\n"
        f"<backtrack:CP1>\n"
        f"<reg:deduce>replan: {deduce.replace('<reg:deduce>', '').replace('</reg:deduce>', '').strip()}</reg:deduce>\n"
        f"<reg:verify>verify(ans == {gt})✓</reg:verify>\n"
        f"</think>\n"
        f"<ans>{gt}</ans>"
    )


def build_legendre_contrastive(item: Dict[str, Any]) -> str:
    plan = item.get("canonical_plan", "")
    deduce = item.get("canonical_deduce", "")
    gt = item.get("ground_truth", "")
    prompt = item.get("prompt", "")

    # Extract n and p
    match = re.search(r"(\d+)!.*prime (\d+)|(\d+) that divides (\d+)!", prompt)
    if match:
        p_val = match.group(3) or match.group(2)
        n_val = match.group(4) or match.group(1)
        if p_val and n_val:
            n_int, p_int = int(n_val), int(p_val)
            t1 = n_int // p_int
            return (
                f"<think>\n"
                f"{plan}\n"
                f"<checkpoint:CP1>\n"
                f"<reg:deduce>hyp [take first term only floor({n_val}/{p_val}) = {t1}]</reg:deduce>\n"
                f"<reg:verify>verify(complete valuation across all k)✗ [powers {p_val}^k (k>=2) omitted]</reg:verify>\n"
                f"<backtrack:CP1>\n"
                f"<reg:deduce>replan: {deduce.replace('<reg:deduce>', '').replace('</reg:deduce>', '').strip()}</reg:deduce>\n"
                f"<reg:verify>verify(ans == {gt})✓</reg:verify>\n"
                f"</think>\n"
                f"<ans>{gt}</ans>"
            )

    return (
        f"<think>\n"
        f"{plan}\n"
        f"<checkpoint:CP1>\n"
        f"<reg:deduce>hyp [truncate series at k=1]</reg:deduce>\n"
        f"<reg:verify>verify(all powers evaluated)✗ [incomplete summation]</reg:verify>\n"
        f"<backtrack:CP1>\n"
        f"<reg:deduce>replan: {deduce.replace('<reg:deduce>', '').replace('</reg:deduce>', '').strip()}</reg:deduce>\n"
        f"<reg:verify>verify(ans == {gt})✓</reg:verify>\n"
        f"</think>\n"
        f"<ans>{gt}</ans>"
    )


def transform_sample_to_contrastive(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    concept = item.get("concept", "")
    domain = item.get("domain", "")
    target = item.get("full_target", "")

    # Inject 35% contrastive negative verification traces for multi-step domains
    is_contrastive = (index % 3 == 0) and ("<think>" in target) and (domain not in ["Δ0 Factual Bypass", "Social Pragmatics", "Intent Dialogue"])

    if is_contrastive:
        item["is_contrastive_negative"] = True
        if concept == "polynomial_invariant":
            item["full_target"] = build_polynomial_contrastive(item)
        elif concept == "chinese_remainder":
            item["full_target"] = build_crt_contrastive(item)
        elif concept == "legendre_formula":
            item["full_target"] = build_legendre_contrastive(item)
        else:
            plan = item.get("canonical_plan", "")
            deduce = item.get("canonical_deduce", "")
            gt = item.get("ground_truth", "")
            item["full_target"] = (
                f"<think>\n"
                f"{plan}\n"
                f"<checkpoint:CP1>\n"
                f"<reg:deduce>hyp [evaluate linear approximation]</reg:deduce>\n"
                f"<reg:verify>verify(invariant constraint satisfied)✗ [invariant violation]</reg:verify>\n"
                f"<backtrack:CP1>\n"
                f"<reg:deduce>replan: {deduce.replace('<reg:deduce>', '').replace('</reg:deduce>', '').strip()}</reg:deduce>\n"
                f"<reg:verify>verify(ans == {gt})✓</reg:verify>\n"
                f"</think>\n"
                f"<ans>{gt}</ans>"
            )

    return item


def main():
    print(f"Reading base curriculum from: {IN_FILE}")
    samples: List[Dict[str, Any]] = []
    with open(IN_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            transformed = transform_sample_to_contrastive(item, i)
            samples.append(transformed)

    contrastive_count = sum(1 for s in samples if s.get("is_contrastive_negative", False))
    print(f"Total samples: {len(samples)} | Contrastive rollback traces: {contrastive_count} ({contrastive_count/len(samples)*100:.1f}%)")

    print(f"Writing contrastive curriculum to: {OUT_FILE}")
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Compute SHA-256
    with open(OUT_FILE, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    checksum_line = f"{sha256}  data/concept_math_contrastive_v2_1.jsonl\n"
    print(f"[OK] SHA-256: {sha256}")

    # Update CHECKSUMS.sha256
    lines = []
    if CHECKSUMS_FILE.exists():
        lines = [ln for ln in CHECKSUMS_FILE.read_text(encoding="utf-8").splitlines() if "concept_math_contrastive_v2_1.jsonl" not in ln]
    lines.append(f"{sha256}  data/concept_math_contrastive_v2_1.jsonl")
    CHECKSUMS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Updated hash in {CHECKSUMS_FILE}")


if __name__ == "__main__":
    main()
