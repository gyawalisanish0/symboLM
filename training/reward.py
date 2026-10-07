"""
SymboLM Reward Function
========================
Multi-component reward for GRPO training.

Components:
  r_correctness  — is the final answer correct? (dominant signal)
  r_efficiency   — how many tokens were saved vs English baseline?
  r_format       — does the trace follow symbolic grammar?
  r_length       — mild penalty for verbosity (discourages padding)
  r_leak         — penalty for English filler leaking into the trace

Total: weighted sum, normalized to roughly [-1, +1] range.

Design notes:
  - Correctness dominates (weight 1.0) — don't sacrifice accuracy for compression
  - Efficiency is a bonus (weight 0.4) — reward compact correct traces more
  - Format is a guide (weight 0.2) — nudge toward grammar without hard constraint
  - Length is a whisper (weight 0.001) — very mild, just breaks ties
  - Leak is a warning (weight 0.3) — penalize English filler meaningfully
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from symbolic.grammar import validate_trace, extract_answer, _estimate_tokens
from symbolic.parser import SRLParser, evaluate_invariants


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------

def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.
    Strips whitespace, commas in numbers, trailing zeros, True/False casing.
    """
    ans = answer.strip()
    # Numeric: remove commas, normalize floats
    try:
        # Handle fractions
        if "/" in ans:
            parts = ans.split("/")
            val = float(parts[0]) / float(parts[1])
            return str(round(val, 6))
        val = float(ans.replace(",", ""))
        # Remove trailing zeros for display consistency
        if val == int(val):
            return str(int(val))
        return str(round(val, 6))
    except ValueError:
        pass
    # Boolean
    lower = ans.lower()
    if lower in ("true", "yes", "correct", "1"):
        return "true"
    if lower in ("false", "no", "incorrect", "0"):
        return "false"
    return lower


def answers_match(predicted: str | None, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth after normalization."""
    if predicted is None:
        return False
    try:
        return normalize_answer(predicted) == normalize_answer(ground_truth)
    except Exception:
        return predicted.strip().lower() == ground_truth.strip().lower()


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """Detailed reward breakdown for logging / analysis."""
    total: float
    correctness: float
    efficiency:  float
    format_score: float
    invariant_score: float
    length_penalty: float
    leak_penalty: float

    def __str__(self) -> str:
        return (
            f"total={self.total:+.3f} | "
            f"correct={self.correctness:+.2f} | "
            f"eff={self.efficiency:+.2f} | "
            f"fmt={self.format_score:+.2f} | "
            f"inv={self.invariant_score:+.2f} | "
            f"len={self.length_penalty:+.3f} | "
            f"leak={self.leak_penalty:+.2f}"
        )


# Weights — adjust these to change training behavior
WEIGHTS = {
    "correctness":    1.0,    # Binary: +1 correct, -0.5 wrong
    "efficiency":     0.4,    # Fraction of tokens saved (0 to 0.5 bonus)
    "format":         0.2,    # Grammar adherence (+0.2 max)
    "invariant":      0.3,    # Deterministic SymPy step verification (+0.3 / -0.3)
    "length":         0.001,  # Per-token penalty (very mild)
    "leak":           0.3,    # English leak penalty (0 to -0.3)
}

# Baseline: average English CoT token count (from dataset stats)
# Used to compute relative efficiency when we don't have the English version
BASELINE_ENGLISH_TOKENS = 55  # rough GSM8K average


def compute_reward(
    prediction: str,
    ground_truth: str,
    english_reference: str | None = None,
    return_breakdown: bool = False,
) -> float | RewardBreakdown:
    """
    Compute the total GRPO reward for a model prediction.

    Args:
        prediction:        Full model output (including <think> and <ans> tags).
        ground_truth:      The correct answer string.
        english_reference: If available, the English CoT for efficiency comparison.
                           If None, uses BASELINE_ENGLISH_TOKENS.
        return_breakdown:  If True, return RewardBreakdown instead of float.

    Returns:
        Scalar reward (float) or RewardBreakdown if requested.
    """
    # ----------------------------------------------------------------
    # 1. Extract symbolic trace and answer
    # ----------------------------------------------------------------
    think_match = re.search(r"<think>(.*?)</think>", prediction, re.DOTALL)
    symbolic_trace = think_match.group(1).strip() if think_match else prediction.strip()

    predicted_answer = extract_answer(prediction)

    # ----------------------------------------------------------------
    # 2. Correctness reward (dominant signal)
    # ----------------------------------------------------------------
    is_correct = answers_match(predicted_answer, ground_truth)
    r_correctness = 1.0 if is_correct else -0.5

    # ----------------------------------------------------------------
    # 3. Efficiency reward (token compression bonus)
    # ----------------------------------------------------------------
    sym_tokens = _estimate_tokens(symbolic_trace)

    if english_reference:
        eng_tokens = _estimate_tokens(english_reference)
    else:
        eng_tokens = BASELINE_ENGLISH_TOKENS

    if eng_tokens > 0:
        raw_compression = 1.0 - (sym_tokens / eng_tokens)
        # Clip to [0, 0.5] — we reward compression, but not at the expense of everything
        # Also no reward for making it longer than English (negative compression)
        r_efficiency = max(0.0, min(0.5, raw_compression))
    else:
        r_efficiency = 0.0

    # Only give efficiency bonus if the answer is correct
    # (prevents model from learning to be compact but wrong)
    if not is_correct:
        r_efficiency = 0.0

    # ----------------------------------------------------------------
    # 4. Format & Invariant Reward (SRL v1.0 AST Compiler)
    # ----------------------------------------------------------------
    srl_parser = SRLParser()
    parse_res = srl_parser.parse(prediction)

    r_format = 0.0
    r_invariant = 0.0

    if parse_res.is_valid:
        r_format += 0.20
    else:
        r_format -= 0.15

    # Deterministic SymPy Step Invariant Verification
    if parse_res.ast and not parse_res.ast.is_empty:
        invariant_diags = evaluate_invariants(parse_res.ast)
        if not invariant_diags:
            r_invariant = 0.30  # All intermediate math steps strictly preserved truth
        else:
            r_invariant = -0.30  # Hallucinated or broken intermediate arithmetic

    # ----------------------------------------------------------------
    # 5. Length penalty (very mild — just breaks ties)
    # ----------------------------------------------------------------
    r_length = -WEIGHTS["length"] * sym_tokens

    # ----------------------------------------------------------------
    # 6. English leak penalty
    # ----------------------------------------------------------------
    validation = validate_trace(symbolic_trace)
    leak_score = validation.english_leak_score  # 0 = pure symbolic, 1 = pure English
    r_leak = -WEIGHTS["leak"] * leak_score

    # ----------------------------------------------------------------
    # 7. Combine
    # ----------------------------------------------------------------
    total = (
        WEIGHTS["correctness"] * r_correctness
        + WEIGHTS["efficiency"] * r_efficiency
        + WEIGHTS["format"]    * r_format
        + WEIGHTS["invariant"] * r_invariant
        + r_length               # already weighted (per-token)
        + r_leak                 # already weighted
    )

    breakdown = RewardBreakdown(
        total=total,
        correctness=WEIGHTS["correctness"] * r_correctness,
        efficiency= WEIGHTS["efficiency"]  * r_efficiency,
        format_score=WEIGHTS["format"]     * r_format,
        invariant_score=WEIGHTS["invariant"] * r_invariant,
        length_penalty=r_length,
        leak_penalty=r_leak,
    )

    return breakdown if return_breakdown else total


# ---------------------------------------------------------------------------
# Batch reward function (interface expected by TRL GRPO trainer)
# ---------------------------------------------------------------------------

def batch_reward_fn(
    predictions: list[str],
    ground_truths: list[str],
    english_refs: list[str] | None = None,
) -> list[float]:
    """
    Compute rewards for a batch of predictions.
    This is the function signature expected by trl.GRPOTrainer.

    Args:
        predictions:   List of model outputs (full text with <think> and <ans>).
        ground_truths: List of correct answers.
        english_refs:  Optional list of English CoT references for efficiency computation.
    """
    refs = english_refs or [None] * len(predictions)
    return [
        compute_reward(pred, gt, eng_ref)
        for pred, gt, eng_ref in zip(predictions, ground_truths, refs)
    ]


# ---------------------------------------------------------------------------
# Reward factory for TRL (returns a closure)
# ---------------------------------------------------------------------------

def make_reward_fn(include_english_refs: bool = False) -> Callable:
    """
    Factory that returns a reward function compatible with TRL GRPOTrainer.

    The returned function signature matches:
        reward_fn(prompts, completions, **kwargs) -> list[float]
    """
    def reward_fn(prompts: list, completions: list, **kwargs) -> list[float]:
        ground_truths = kwargs.get("answers", [""] * len(completions))
        english_refs  = kwargs.get("english_cots", None) if include_english_refs else None

        # completions may be list of strings or list of dicts (chat format)
        pred_texts = []
        for comp in completions:
            if isinstance(comp, str):
                pred_texts.append(comp)
            elif isinstance(comp, list):
                # Chat completion format: [{role, content}, ...]
                content = " ".join(m.get("content", "") for m in comp if m.get("role") == "assistant")
                pred_texts.append(content)
            else:
                pred_texts.append(str(comp))

        return batch_reward_fn(pred_texts, ground_truths, english_refs)

    return reward_fn


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("SymboLM Reward Function Demo\n" + "=" * 50)

    test_cases = [
        {
            "name": "Correct + compact symbolic",
            "prediction": "<think>\nlet(M=5) | M-=2→M=3 | M+=3→M=6 ∴ ans=6\n</think>\n<ans>6</ans>",
            "ground_truth": "6",
            "english": "Mary starts with 5 apples. She gives away 2, leaving 3. Then buys 3 more for a total of 6.",
        },
        {
            "name": "Correct but English-heavy (verbose)",
            "prediction": "<think>\nLet me think step by step. First, Mary has 5 apples. She gives 2 to John, so she has 3. Then she buys 3 more, so she has 6 apples total.\n</think>\n<ans>6</ans>",
            "ground_truth": "6",
            "english": "Mary starts with 5 apples. She gives away 2, leaving 3. Then buys 3 more for a total of 6.",
        },
        {
            "name": "Wrong answer",
            "prediction": "<think>\nlet(M=5) | M-=2→M=3 | M+=3→M=7 ∴ ans=7\n</think>\n<ans>7</ans>",
            "ground_truth": "6",
        },
        {
            "name": "Correct + very compact",
            "prediction": "<think>\n2^10=1024 | verify(1024>1000)✓ ∴ ans=True\n</think>\n<ans>True</ans>",
            "ground_truth": "True",
            "english": "I need to compute 2 to the power of 10. Starting with 2 and doubling: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024. Since 1024 is greater than 1000, the answer is yes.",
        },
    ]

    for tc in test_cases:
        breakdown = compute_reward(
            tc["prediction"],
            tc["ground_truth"],
            tc.get("english"),
            return_breakdown=True,
        )
        print(f"\n[{tc['name']}]")
        print(f"  {breakdown}")
