"""
SymboLM Ground Truth Answer Verifier
===================================
Robustly checks whether an extracted model answer matches the ground truth.
Handles integers, floats, fractions, equations, latex expressions, and booleans.
"""

from __future__ import annotations

import math
import re
from typing import Any


def clean_math_string(s: str) -> str:
    """Removes standard LaTeX / formatting wrappers."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = re.sub(r"^\$+", "", s)
    s = re.sub(r"\$+$", "", s)
    s = s.replace(r"\text{", "").replace(r"\mathrm{", "")
    s = s.replace(r"\mathbf{", "")
    s = s.replace("$", "").replace("%", "").replace(",", "")
    s = s.strip()
    return s


def parse_numeric(val_str: str) -> float | None:
    """Attempts to parse standard numbers or basic fractions into float."""
    cleaned = clean_math_string(val_str)
    # Check simple float / int
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Fraction: a/b or \frac{a}{b}
    frac_match = re.search(r"\\frac\{([^\}]+)\}\{([^\}]+)\}", cleaned)
    if frac_match:
        try:
            num = float(frac_match.group(1).strip())
            denom = float(frac_match.group(2).strip())
            if denom != 0:
                return num / denom
        except ValueError:
            pass

    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0].strip())
                denom = float(parts[1].strip())
                if denom != 0:
                    return num / denom
            except ValueError:
                pass

    return None


def verify_answer(predicted: Any, expected: Any, tolerance: float = 1e-4) -> bool:
    """
    Returns True if predicted matches expected.
    Works across numeric types, bools, and canonicalized strings.
    """
    if predicted is None or expected is None:
        return False

    pred_str = str(predicted).strip()
    exp_str = str(expected).strip()

    if pred_str.lower() == exp_str.lower():
        return True

    # Check boolean representations
    bool_map = {
        "true": True, "yes": True, "1": True, "correct": True,
        "false": False, "no": False, "0": False, "incorrect": False,
    }
    if pred_str.lower() in bool_map and exp_str.lower() in bool_map:
        return bool_map[pred_str.lower()] == bool_map[exp_str.lower()]

    # Numeric comparison
    num_pred = parse_numeric(pred_str)
    num_exp = parse_numeric(exp_str)

    if num_pred is not None and num_exp is not None:
        if math.isclose(num_pred, num_exp, rel_tol=tolerance, abs_tol=tolerance):
            return True

    # Strip whitespace & symbols comparison
    norm_p = clean_math_string(pred_str).lower()
    norm_e = clean_math_string(exp_str).lower()
    return norm_p == norm_e


if __name__ == "__main__":
    assert verify_answer("42", "42")
    assert verify_answer("42.0", "42")
    assert verify_answer("1/2", "0.5")
    assert verify_answer(r"\frac{3}{4}", "0.75")
    assert verify_answer("True", "true")
    assert not verify_answer("42", "43")
    print("All answer verifier checks passed successfully.")
