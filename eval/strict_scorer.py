"""
Strict declared-answer scoring for SymboLM.
==========================================

The old scorer could harvest the last equality inside ``<reg:deduce>``.  That
made a truncated or malformed trace look correct.  This module implements the
scoring contract used by the rescue experiment:

* A reasoning answer must be declared in ``<ans>...</ans>``.
* An atomic answer must be declared in ``<reg:bypass>...</reg:bypass>``.
* Baseline arms may use a closed ``\\boxed{...}`` because their prompt asks for it.
* A generation that hits the token budget without EOS is truncated and fails.
* There is no trailing-number fallback, no ``deduce_equality_rhs`` fallback, and
  no acceptance of ``</reg:verify>`` as an answer.

The scorer also reports whether the register trace is structurally valid, so a
correct answer inside a malformed trace is visible instead of being hidden.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from symbolic.prompts import (
    KNOWN_LOOP_MARKERS,
    contains_loop_marker,
    normalize_target_body,
)

EOS_MARKERS = (
    "<｜end▁of▁sentence｜>",
    "<|endoftext|>",
    "<｜end of sentence｜>",
)

BOXED_ARMS = {"A", "B", "D", "E", "F"}


def clean_math_string(value: str) -> str:
    """Normalize a math answer for comparison."""
    if value is None:
        return ""
    s = str(value).strip()
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = s.replace("$", "").replace("\\$", "").strip()
    s = s.replace(" ", "")
    return s


def strip_eos(text: str) -> str:
    s = text or ""
    for marker in EOS_MARKERS:
        s = s.replace(marker, "")
    return s.strip()


def extract_boxed(text: str) -> Optional[str]:
    """Extract the last closed ``\\boxed{...}``, handling nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else None


def verify_answer(extracted: Optional[str], ground_truth: str) -> Tuple[bool, str]:
    """Compare a declared answer with ground truth using exact/symbolic checks."""
    if extracted is None or not str(extracted).strip():
        return False, "fail"

    gt = clean_math_string(str(ground_truth).strip().lower())
    ext = clean_math_string(str(extracted).strip().lower())

    if ext == gt:
        return True, "string_exact"

    if "/" in gt and "/" in ext:
        try:
            p_gt = [float(x) for x in gt.split("/")]
            p_ext = [float(x) for x in ext.split("/")]
            if (
                len(p_gt) == 2
                and len(p_ext) == 2
                and p_gt[1] != 0
                and p_ext[1] != 0
                and abs(p_gt[0] / p_gt[1] - p_ext[0] / p_ext[1]) < 1e-4
            ):
                return True, "fraction_approx"
        except Exception:
            pass

    try:
        f_ext = float(re.sub(r"[^\d.\-]", "", ext))
        f_gt = float(re.sub(r"[^\d.\-]", "", gt))
        if abs(f_ext - f_gt) < 1e-3:
            return True, "float_approx"
        if f_gt != 0 and abs(f_ext - f_gt) / abs(f_gt) < 1e-3:
            return True, "float_approx"
    except Exception:
        pass

    try:
        import sympy

        expr_ext = sympy.sympify(ext, evaluate=True)
        expr_gt = sympy.sympify(gt, evaluate=True)
        if sympy.simplify(expr_ext - expr_gt) == 0:
            return True, "sympy_symbolic"
    except Exception:
        pass

    return False, "fail"


def _meta(
    answer: Optional[str],
    method: str,
    hit_budget: bool,
    truncated: bool,
    closure: Optional[str],
) -> Dict[str, Any]:
    return {
        "answer": answer,
        "method": method,
        "answer_declared": bool(answer),
        "closure": closure,
        "hit_budget": hit_budget,
        "truncated": truncated,
    }


def extract_declared_answer(
    text: str,
    arm: str,
    tokens_emitted: int,
    budget: int,
    eos_emitted: Optional[bool] = None,
    allow_boxed: Optional[bool] = None,
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """
    Extract only declared answers.

    ``eos_emitted`` is optional for old receipts that did not record it.  When
    it is None, the strict rule is ``tokens_emitted >= budget``.  When it is
    known, an EOS at the final token means the model terminated naturally and
    is not counted as truncated.
    """
    if not (text or "").strip():
        return None, "empty_text", _meta(None, "empty_text", False, False, None)

    hit_budget = budget > 0 and tokens_emitted >= budget
    truncated = hit_budget and not bool(eos_emitted) if eos_emitted is not None else hit_budget
    if truncated:
        return None, "truncated", _meta(None, "truncated", hit_budget, True, None)

    # 1. Reasoning answer: <ans>...</ans>
    match = re.search(r"<ans>\s*([^<]+?)\s*</ans>", text, re.IGNORECASE)
    if match:
        answer = clean_math_string(match.group(1).strip())
        if answer:
            return answer, "closed_ans_tag", _meta(
                answer, "closed_ans_tag", hit_budget, False, "ans"
            )

    # 2. Atomic answer: <reg:bypass>...</reg:bypass>
    match = re.search(r"<reg:bypass>\s*([^<]+?)\s*</reg:bypass>", text, re.IGNORECASE)
    if match:
        answer = clean_math_string(match.group(1).strip())
        if answer:
            return answer, "closed_bypass_tag", _meta(
                answer, "closed_bypass_tag", hit_budget, False, "bypass"
            )

    # 3. Baseline arms explicitly request \boxed{}.
    if allow_boxed is None:
        allow_boxed = arm in BOXED_ARMS
    if allow_boxed:
        boxed = extract_boxed(text)
        if boxed:
            answer = clean_math_string(boxed)
            if answer:
                return answer, "latex_boxed", _meta(
                    answer, "latex_boxed", hit_budget, False, "boxed"
                )

    return None, "no_declared_answer", _meta(
        None, "no_declared_answer", hit_budget, False, None
    )


def _tag_names(text: str):
    return re.findall(r"</?([^<>]+)>", text)


def validate_register_trace(
    text: str,
    arm: str = "C",
    expected_kind: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate the outer structure of a SymboLM register trace.

    ``expected_kind`` is the route the router actually selected: ``reasoning``
    for a non-atomic problem and ``bypass`` (or the synonym ``atomic``) for an
    atomic one.  Passing it prevents Arm C from receiving format credit when it
    emits a bypass on a problem that required reasoning, or vice versa.

    This is intentionally a format validator, not a grammar-constrained decoder.
    It is used to measure whether the model actually learned the contract.
    """
    if not (text or "").strip():
        return False, "empty_trace"

    expected = (expected_kind or "").strip().lower()
    if expected == "atomic":
        expected = "bypass"
    if expected not in {"", "reasoning", "bypass"}:
        raise ValueError(f"Unknown expected_kind: {expected_kind!r}")

    loop_marker = contains_loop_marker(text)
    if loop_marker:
        return False, f"loop_marker:{loop_marker}"

    s = strip_eos(text)

    # Atomic bypass trace.
    if "</reg:bypass>" in s:
        if "</ans>" in s:
            return False, "bypass_with_ans"
        if expected == "reasoning":
            return False, "expected_reasoning_got_bypass"
        if s.count("<reg:bypass>") != 1 or s.count("</reg:bypass>") != 1:
            return False, "bypass_tag_count"
        pattern = r"^\s*<reg:bypass>[^<]+</reg:bypass>\s*(?:</think>)?\s*$"
        if not re.match(pattern, s):
            return False, "bypass_not_terminal"
        return True, "bypass"

    if expected == "bypass":
        return False, "expected_bypass_got_reasoning"

    if arm != "C":
        return False, "not_a_register_arm"

    if "</ans>" not in s:
        return False, "missing_ans"
    if "</think>" not in s:
        return False, "missing_think_close"

    required = [
        "<reg:plan>",
        "</reg:plan>",
        "<reg:deduce>",
        "</reg:deduce>",
        "</think>",
        "<ans>",
        "</ans>",
    ]
    positions = [s.find(tag) for tag in required]
    if any(pos < 0 for pos in positions):
        missing = [tag for tag, pos in zip(required, positions) if pos < 0]
        return False, "missing:" + ",".join(missing)
    if positions != sorted(positions):
        return False, "tag_order"

    if s.count("<reg:plan>") != 1 or s.count("</reg:plan>") != 1:
        return False, "plan_tag_count"
    if s.count("<reg:deduce>") != 1 or s.count("</reg:deduce>") != 1:
        return False, "deduce_tag_count"
    if s.count("<ans>") != 1 or s.count("</ans>") != 1:
        return False, "ans_tag_count"

    if "<reg:verify>" in s:
        if s.count("<reg:verify>") != 1 or s.count("</reg:verify>") != 1:
            return False, "verify_tag_count"
        v_open = s.find("<reg:verify>")
        v_close = s.find("</reg:verify>")
        if not (positions[3] < v_open < v_close < positions[4]):
            return False, "verify_order"

    allowed = {
        "think",
        "/think",
        "reg:plan",
        "/reg:plan",
        "reg:deduce",
        "/reg:deduce",
        "reg:verify",
        "/reg:verify",
        "ans",
        "/ans",
    }
    tags = _tag_names(s)
    unsupported = [tag for tag in tags if tag not in allowed]
    if unsupported:
        return False, "unsupported_tag:" + unsupported[0]

    # Text outside the known register blocks must be whitespace only.  Remove
    # whole blocks (not just their tags) so the content inside a legitimate
    # register is not mistaken for stray prose.
    residual = s
    for tag in ("reg:plan", "reg:deduce", "reg:verify", "ans"):
        residual = re.sub(
            rf"<{tag}>.*?</{tag}>", "", residual, flags=re.DOTALL
        )
    residual = residual.replace("</think>", "")
    if residual.strip():
        return False, "text_outside_registers"

    def block(tag: str) -> str:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", s, re.DOTALL)
        return match.group(1).strip() if match else ""

    if not block("reg:plan"):
        return False, "empty_plan"
    if not block("reg:deduce"):
        return False, "empty_deduce"
    if not block("ans"):
        return False, "empty_ans"

    return True, "reasoning"


def score_declared_answer(
    text: str,
    arm: str,
    ground_truth: str,
    tokens_emitted: int,
    budget: int,
    eos_emitted: Optional[bool] = None,
    expected_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Full strict score record for one generated trace.

    The preregistration's primary endpoint is declared-answer correctness
    (``strict_correct``).  ``strict_correct_and_format`` is a stricter,
    clearly-labeled secondary field; it is never silently substituted for the
    preregistered endpoint.
    """
    answer, method, meta = extract_declared_answer(
        text,
        arm=arm,
        tokens_emitted=tokens_emitted,
        budget=budget,
        eos_emitted=eos_emitted,
    )
    correct, verify_method = verify_answer(answer, ground_truth)
    if arm == "C":
        format_valid, format_reason = validate_register_trace(
            text, arm=arm, expected_kind=expected_kind
        )
    else:
        format_valid, format_reason = bool(answer), "declared_answer"

    return {
        "answer": answer,
        "extraction_method": method,
        "answer_declared": meta["answer_declared"],
        "closure": meta["closure"],
        "hit_budget": meta["hit_budget"],
        "truncated": meta["truncated"],
        "format_valid": format_valid,
        "format_reason": format_reason,
        "expected_kind": (
            (
                "bypass"
                if (expected_kind or "").strip().lower() == "atomic"
                else (expected_kind or "").strip().lower() or None
            )
            if arm == "C"
            else None
        ),
        "is_correct": correct,
        "strict_correct": correct,
        "strict_correct_and_format": bool(correct and format_valid),
        "verification_method": verify_method,
    }


def load_ground_truth_overrides(path: str | Path) -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Ground-truth overrides must be a JSON object: {p}")
    return data


def apply_ground_truth_override(
    item: Dict[str, Any], overrides: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Return a copy of *item* with a documented ground-truth override applied."""
    row = dict(item)
    problem_id = row.get("id")
    # Preserve provenance if the row was already audited.  This makes the
    # function idempotent when v2 is passed through the evaluator again.
    row["ground_truth_original"] = row.get("ground_truth_original", row.get("ground_truth"))
    row["ground_truth_source"] = row.get("ground_truth_source", "benchmark")
    if problem_id in overrides:
        override = overrides[problem_id]
        row["ground_truth"] = str(override["ground_truth"])
        row["ground_truth_source"] = "override"
        row["ground_truth_override_reason"] = override.get("reason", "")
    return row
