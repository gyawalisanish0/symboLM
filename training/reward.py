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
    concept_score: float = 0.0
    anti_guess_penalty: float = 0.0
    length_penalty: float = 0.0
    leak_penalty: float = 0.0

    def __str__(self) -> str:
        return (
            f"total={self.total:+.3f} | "
            f"correct={self.correctness:+.2f} | "
            f"eff={self.efficiency:+.2f} | "
            f"fmt={self.format_score:+.2f} | "
            f"inv={self.invariant_score:+.2f} | "
            f"cpt={self.concept_score:+.2f} | "
            f"noguess={self.anti_guess_penalty:+.2f} | "
            f"len={self.length_penalty:+.3f} | "
            f"leak={self.leak_penalty:+.2f}"
        )


# Weights — adjust these to change training behavior
WEIGHTS = {
    "correctness":    1.0,    # Binary: +1 correct, -0.5 wrong
    "efficiency":     0.4,    # Fraction of tokens saved (0 to 0.5 bonus)
    "format":         0.2,    # Grammar adherence (+0.2 max)
    "invariant":      0.3,    # Deterministic SymPy step verification (+0.3 / -0.3)
    "concept":        0.4,    # Register v2.0 concept alignment bonus (+0.4 max)
    "length":         0.001,  # Per-token penalty (very mild)
    "leak":           0.3,    # English leak penalty (0 to -0.3)
}

# Baseline: average English CoT token count (from dataset stats)
# Used to compute relative efficiency when we don't have the English version
BASELINE_ENGLISH_TOKENS = 55  # rough GSM8K average


NOISE_PATTERNS = [
    re.compile(r"=\{`?\$\{?"),                       # e.g. ={`${105}`}>
    re.compile(r"\?>[\"']"),                           # e.g. ?>"></
    re.compile(r"<\s*/\s*<\s*\|"),                     # broken tags
    re.compile(r"[<{]{3,}"),                           # repeated bracket soup
    re.compile(r"\$\{[^}]*\}"),                        # template string leftovers
    re.compile(r"<\/?[a-zA-Z0-9_\-]+(?!\w)[^>]*$"),    # unclosed trailing html/xml tag
]


def detect_trailing_noise(text: str) -> float:
    """Detect unparsed trailing syntax garbage and template residue."""
    for pat in NOISE_PATTERNS:
        if pat.search(text):
            return -1.0
    # Also check if text ends with invalid dangling punctuation after numbers
    if re.search(r"\d+\s*[\?=>\"'\}\{]{2,}", text):
        return -1.0
    return 0.0


def detect_self_correction(text: str, is_correct: bool) -> float:
    """Award bonus if model used compact verification or backtrack to arrive at truth."""
    if not is_correct:
        return 0.0
    has_backtrack = ("✗" in text) or bool(re.search(r"\bbacktrack\b", text)) or bool(re.search(r"\btry\(", text))
    has_verify = bool(re.search(r"verify\(", text)) or ("✓" in text)
    if (has_backtrack or has_verify) and ("∴" in text or "<ans>" in text):
        return 0.4
    return 0.0


def compute_reward(
    prediction: str,
    ground_truth: str,
    english_reference: str | None = None,
    expected_concept: str | None = None,
    is_multi_step: bool = False,
    return_breakdown: bool = False,
) -> float | RewardBreakdown:
    """
    Compute the total GRPO reward for a model prediction.

    Args:
        prediction:        Full model output (including <think> and <ans> tags).
        ground_truth:      The correct answer string.
        english_reference: If available, the English CoT for efficiency comparison.
                           If None, uses BASELINE_ENGLISH_TOKENS.
        expected_concept:  Optional canonical mathematical concept (e.g. 'chinese_remainder').
        is_multi_step:     If True, penalizes blind guesses without macro-planning / steps.
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
        # Clip to [0, 0.5] — reward compression, but not at the expense of everything
        r_efficiency = max(0.0, min(0.5, raw_compression))
    else:
        r_efficiency = 0.0

    # Only give efficiency bonus if the answer is correct
    if not is_correct:
        r_efficiency = 0.0

    # ----------------------------------------------------------------
    # 4. Format & Invariant Reward (SRL v1.0 AST Compiler)
    # ----------------------------------------------------------------
    srl_parser = SRLParser()
    parse_res = srl_parser.parse(prediction)

    r_format = 0.0
    r_invariant = 0.0
    r_concept = 0.0

    if parse_res.is_valid:
        r_format += 0.20
    else:
        r_format -= 0.15

    # Register v2.0 Concept Grounding Reward
    if parse_res.ast and parse_res.ast.plan:
        r_concept += 0.25  # Valid macro-plan declared
        if expected_concept:
            cand = parse_res.ast.plan.concept.lower()
            exp = expected_concept.lower()
            if exp in cand or cand in exp:
                r_concept += 0.20  # Matched domain invariant

    # Deterministic SymPy Step Invariant Verification
    if parse_res.ast and not parse_res.ast.is_empty:
        invariant_diags = evaluate_invariants(parse_res.ast)
        if not invariant_diags:
            r_invariant = 0.30  # All intermediate math steps strictly preserved truth
        else:
            r_invariant = -0.30  # Hallucinated or broken intermediate arithmetic

    # ----------------------------------------------------------------
    # 5. Anti-Guessing Penalty (kills the Binary Trap)
    # ----------------------------------------------------------------
    r_anti_guess = 0.0
    if is_multi_step:
        has_plan = bool(parse_res.ast and parse_res.ast.plan)
        has_multi_steps = bool(parse_res.ast and len(parse_res.ast.steps) > 1)
        has_bypass = "<reg:bypass>" in prediction or (parse_res.ast and parse_res.ast.register_type == "bypass")
        # If model attempted blind 1-token guess on complex problem
        if (sym_tokens <= 5 and not has_plan and not has_multi_steps) or has_bypass:
            r_anti_guess = -0.80

    # ----------------------------------------------------------------
    # 6. Length penalty (very mild — just breaks ties)
    # ----------------------------------------------------------------
    r_length = -WEIGHTS["length"] * sym_tokens

    # ----------------------------------------------------------------
    # 7. English leak penalty
    # ----------------------------------------------------------------
    validation = validate_trace(symbolic_trace)
    leak_score = validation.english_leak_score  # 0 = pure symbolic, 1 = pure English
    r_leak = -WEIGHTS["leak"] * leak_score

    # ----------------------------------------------------------------
    # 8. Trailing Noise & Syntax Hallucination Penalty
    # ----------------------------------------------------------------
    r_noise = detect_trailing_noise(prediction)

    # ----------------------------------------------------------------
    # 9. Compact Self-Correction Bonus
    # ----------------------------------------------------------------
    r_self_correct = detect_self_correction(prediction, is_correct)

    # ----------------------------------------------------------------
    # 10. Combine
    # ----------------------------------------------------------------
    total = (
        WEIGHTS["correctness"] * r_correctness
        + WEIGHTS["efficiency"] * r_efficiency
        + WEIGHTS["format"]    * r_format
        + WEIGHTS["invariant"] * r_invariant
        + WEIGHTS["concept"]   * r_concept
        + r_anti_guess           # heavy penalty if blind guess
        + r_length               # already weighted (per-token)
        + r_leak                 # already weighted
        + r_noise                # heavy penalty if noisy (-1.0)
        + r_self_correct         # bonus if verified self-correcting (+0.4)
    )

    breakdown = RewardBreakdown(
        total=total,
        correctness=WEIGHTS["correctness"] * r_correctness,
        efficiency= WEIGHTS["efficiency"]  * r_efficiency,
        format_score=WEIGHTS["format"]     * r_format,
        invariant_score=WEIGHTS["invariant"] * r_invariant,
        concept_score=WEIGHTS["concept"]   * r_concept,
        anti_guess_penalty=r_anti_guess,
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
# Modular Reward Functions for TRL GRPOTrainer
# ---------------------------------------------------------------------------

def correctness_reward_func(prompts, completions, answers=None, **kwargs) -> list[float]:
    """Evaluates final answer accuracy against ground-truth normalized answer."""
    ans_list = answers if answers is not None else kwargs.get("answer", kwargs.get("answers", [""] * len(completions)))
    rewards = []
    for comp, ans in zip(completions, ans_list):
        extracted = extract_answer(comp)
        if answers_match(extracted, ans):
            rewards.append(1.0)
        else:
            rewards.append(-0.6)
    return rewards


def syntax_and_noise_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Penalizes trailing noise/template debris and rewards clean SRL grammar."""
    rewards = []
    srl_parser = SRLParser()
    for comp in completions:
        score = 0.0
        # 1. Trailing noise / template leftover penalty
        score += detect_trailing_noise(comp)

        # 2. SRL Parser AST validation
        res = srl_parser.parse(comp)
        if res.is_valid:
            score += 0.5
        else:
            score -= 0.3

        # 3. Soft trace validator
        val = validate_trace(comp)
        if val.has_conclusion:
            score += 0.2
        if val.english_leak_score > 0.25:
            score -= 0.3
        rewards.append(score)
    return rewards


def self_correction_reward_func(prompts, completions, answers=None, **kwargs) -> list[float]:
    """Awards a +0.4 bonus for verified hypothesis testing / backtracking."""
    ans_list = answers if answers is not None else kwargs.get("answer", kwargs.get("answers", [""] * len(completions)))
    rewards = []
    for comp, ans in zip(completions, ans_list):
        extracted = extract_answer(comp)
        is_corr = answers_match(extracted, ans)
        bonus = detect_self_correction(comp, is_corr)
        rewards.append(bonus)
    return rewards


def efficiency_and_bypass_reward_func(prompts, completions, answers=None, registers=None, **kwargs) -> list[float]:
    """Rewards token compression, zero-shot factual bypass, and intent scratchpads."""
    ans_list = answers if answers is not None else kwargs.get("answer", kwargs.get("answers", [""] * len(completions)))
    regs = registers if registers is not None else kwargs.get("register", kwargs.get("registers", ["symbolic_deductive"] * len(completions)))
    rewards = []
    for comp, ans, reg in zip(completions, ans_list, regs):
        extracted = extract_answer(comp)
        is_corr = answers_match(extracted, ans)

        # Register: Factual Bypass
        if reg in ("factual_bypass", "bypass"):
            has_bypass_tag = bool(re.search(r"<reg:bypass>(.*?)</reg:bypass>", comp, re.DOTALL))
            think_match = re.search(r"<think>(.*?)</think>", comp, re.DOTALL)
            think_content = think_match.group(1).strip() if think_match else ""
            if has_bypass_tag or len(think_content) == 0:
                rewards.append(0.9 if is_corr else -0.3)
            elif len(think_content) < 20:
                rewards.append(0.2 if is_corr else -0.2)
            else:
                rewards.append(-0.8)  # Overthinking penalty
            continue

        # Register: Intent Dialogue
        if reg in ("intent_dialogue", "intent"):
            has_intent_tag = bool(re.search(r"(?:<reg:intent>|intent:)\s*[^|\n<]+", comp))
            if has_intent_tag:
                rewards.append(0.6)
            else:
                rewards.append(-0.3)
            continue

        # Register: Concept Math / Symbolic Deductive
        # Penalize attempted zero-shot bypass on complex problems
        has_bypass_tag = bool(re.search(r"<reg:bypass>", comp))
        think_match = re.search(r"<think>(.*?)</think>", comp, re.DOTALL)
        think_content = think_match.group(1).strip() if think_match else ""
        if has_bypass_tag or (len(think_content) == 0 and not is_corr):
            rewards.append(-0.8)  # Anti-guess penalty
            continue

        if not is_corr:
            rewards.append(0.0)
            continue

        # Reward presence of valid <reg:plan>
        plan_bonus = 0.25 if ("<reg:plan>" in comp) else 0.0

        toks = _estimate_tokens(comp)
        baseline = 60
        comp_ratio = max(0.0, min(0.6, (baseline - toks) / baseline))
        rewards.append(float(comp_ratio * 0.5 + plan_bonus))
    return rewards


def concept_alignment_reward_func(prompts, completions, answers=None, concepts=None, **kwargs) -> list[float]:
    """Awards bonus for declaring valid Register v2.0 concept macro-plans."""
    rewards = []
    srl_parser = SRLParser()
    exp_concepts = concepts if concepts is not None else kwargs.get("concept", kwargs.get("concepts", [None] * len(completions)))
    for comp, exp_c in zip(completions, exp_concepts):
        res = srl_parser.parse(comp)
        if res.ast and res.ast.plan:
            bonus = 0.30
            if exp_c:
                cand = res.ast.plan.concept.lower()
                exp = str(exp_c).lower()
                if exp in cand or cand in exp:
                    bonus += 0.20
            rewards.append(bonus)
        else:
            rewards.append(0.0)
    return rewards


def detect_ngram_repetition(text: str) -> bool:
    """Detects looping repetitive n-grams (e.g. '527 527 527...' or 'potassium potassium...')."""
    words = re.findall(r"[\w.-]+", text)
    if len(words) >= 6:
        for n in (1, 2, 3):
            for i in range(len(words) - 3 * n):
                chunk = words[i:i + n]
                if words[i + n:i + 2 * n] == chunk and words[i + 2 * n:i + 3 * n] == chunk:
                    return True
    return False


def detect_web_debris(text: str) -> bool:
    """Detects base-model web vocabulary artifacts (PHP, HTML, random code scrap tokens)."""
    debris_patterns = [
        r"\bendphp\b", r"\bIconModule\b", r"\btatto\b", r"<\?php",
        r"\bimport\s+react\b", r"<!DOCTYPE", r"<\s*/?div\s*>",
        r"__init__", r"typedef\s+", r"public\s+static\s+void"
    ]
    for pat in debris_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def hard_eos_and_economy_reward_func(prompts, completions, answers=None, registers=None, **kwargs) -> list[float]:
    """
    Stage 4 Alignment: Rewards strict token economy, immediate EOS termination,
    and imposes heavy penalties on answer loops, debris, and excessive length.
    """
    rewards = []
    ans_list = answers if answers is not None else kwargs.get("answer", kwargs.get("answers", [""] * len(completions)))
    regs = registers if registers is not None else kwargs.get("register", kwargs.get("registers", ["symbolic_deductive"] * len(completions)))

    for comp, ans, reg in zip(completions, ans_list, regs):
        score = 0.0
        extracted = extract_answer(comp)
        is_corr = answers_match(extracted, ans)

        has_repeat = detect_ngram_repetition(comp)
        has_debris = detect_web_debris(comp)

        # 1. Heavy Repetition Penalty (-2.5)
        if has_repeat:
            score -= 2.5

        # 2. Base Web Debris Penalty (-2.0)
        if has_debris:
            score -= 2.0

        # 3. Clean EOS & Closure Bonus (+1.2)
        tail_after_ans = ""
        if "</ans>" in comp:
            tail_after_ans = comp.split("</ans>")[-1].strip()
        elif "</reg:deduce>" in comp:
            tail_after_ans = comp.split("</reg:deduce>")[-1].strip()
        elif "</reg:bypass>" in comp:
            tail_after_ans = comp.split("</reg:bypass>")[-1].strip()
        elif "</reg:intent>" in comp:
            tail_after_ans = comp.split("</reg:intent>")[-1].strip()
        elif "</think>" in comp:
            tail_after_ans = comp.split("</think>")[-1].strip()

        if is_corr and not has_debris and not has_repeat:
            tail_tokens = len(tail_after_ans.split())
            if tail_tokens <= 4:
                score += 1.2  # Immediate clean EOS cutoff
            elif tail_tokens > 20:
                score -= 1.0  # Trailing unpruned monologue

        # 4. Monotonic Length Economy Penalty / Bonus
        toks = _estimate_tokens(comp)
        if reg in ("factual_bypass", "bypass"):
            target_len = 15
        elif reg in ("intent_dialogue", "intent"):
            target_len = 75
        else:
            target_len = 50  # Concept Math target

        if toks > target_len:
            excess = toks - target_len
            score -= min(1.5, 0.03 * excess)
        elif is_corr and not has_debris and not has_repeat:
            savings = max(0.0, (target_len - toks) / target_len)
            score += 0.5 * savings

        rewards.append(float(score))

    return rewards


# ---------------------------------------------------------------------------
# Stage 5 Alignment: Gradient Registers & Scratchpad 2.0 Reward Mechanism
# ---------------------------------------------------------------------------

def extract_cognitive_depth(comp: str, default: str = "Δ1") -> str:
    """
    Extracts cognitive depth tier (Δ0, Δ1, Δ2, Δ3) from Scratchpad 2.0 header,
    registers, or reasoning structure.
    """
    # 1. Direct explicit Scratchpad 2.0 attribute: <reg:scratchpad depth="2">
    m = re.search(r'depth=["\']?([0-3]|Δ[0-3])["\']?', comp, re.IGNORECASE)
    if m:
        val = m.group(1)
        return val if val.startswith("Δ") else f"Δ{val}"

    # 2. Structural inference from register markers:
    if "<reg:bypass>" in comp:
        return "Δ0"
    if "hyp [" in comp or "<reg:verify>" in comp:
        return "Δ2"
    if "<reg:plan>" in comp or "<reg:deduce>" in comp:
        return "Δ1"
    if "<reg:intent>" in comp:
        return "Δ1"

    return default


def gradient_complexity_reward_func(
    prompts,
    completions,
    answers=None,
    cognitive_depths=None,
    **kwargs
) -> list[float]:
    """
    Stage 5 Alignment: Continuous Gradient-Based Complexity-Proportional Reward
    (Sanish Gyawali's Scratchpad 2.0 / Gradient Register Architecture).

    Formula:
      R(tau, y*, C) = R_correct(y*) * [1 + alpha * Phi(tau, C)] - beta * Psi(tau, C) - gamma * Omega(tau) + R_closure

    Where:
      - C in {Δ0, Δ1, Δ2, Δ3}
        * Δ0: L_opt = 10,  L_max = 15,  lambda = 0.08 (Strict atomic recall)
        * Δ1: L_opt = 30,  L_max = 45,  lambda = 0.04 (Linear invariant)
        * Δ2: L_opt = 55,  L_max = 80,  lambda = 0.03 (Branching verification)
        * Δ3: L_opt = 90,  L_max = 150, lambda = 0.02 (Deep Scratchpad 2.0 unrolling)
      - Phi(tau, C) = max(0.0, (L_opt(C) - |tau|) / L_opt(C))
      - Psi(tau, C) = ReLU(|tau| - L_max(C)) * lambda(C)
      - Omega(tau) = 2.5 (if repeat) + 2.0 (if debris)
    """
    rewards = []
    ans_list = answers if answers is not None else kwargs.get("answer", kwargs.get("answers", [""] * len(completions)))

    # Cognitive depths can be provided in kwargs or extracted from completion
    depth_kwargs = cognitive_depths if cognitive_depths is not None else kwargs.get(
        "cognitive_depth", kwargs.get("cognitive_depths", kwargs.get("depth", [None] * len(completions)))
    )
    if not isinstance(depth_kwargs, list):
        depth_kwargs = [depth_kwargs] * len(completions)

    DEPTH_CONFIG = {
        "Δ0": {"L_opt": 10, "L_max": 15, "lambda": 0.08},
        "Δ1": {"L_opt": 30, "L_max": 45, "lambda": 0.04},
        "Δ2": {"L_opt": 55, "L_max": 80, "lambda": 0.03},
        "Δ3": {"L_opt": 90, "L_max": 150, "lambda": 0.02},
    }

    for idx, (comp, ans) in enumerate(zip(completions, ans_list)):
        # 1. Resolve Cognitive Depth Tier
        c_depth = depth_kwargs[idx] if idx < len(depth_kwargs) and depth_kwargs[idx] else None
        if not c_depth:
            c_depth = extract_cognitive_depth(comp, default="Δ1")
        if not str(c_depth).startswith("Δ"):
            c_depth = f"Δ{c_depth}"

        cfg = DEPTH_CONFIG.get(str(c_depth), DEPTH_CONFIG["Δ1"])
        l_opt = cfg["L_opt"]
        l_max = cfg["L_max"]
        l_lambda = cfg["lambda"]

        # 2. Extract and match answer
        extracted = extract_answer(comp)
        is_corr = answers_match(extracted, ans)

        # 3. Base Correctness Reward
        r_correct = 2.0 if is_corr else -0.8

        # 4. Token Counting
        toks = _estimate_tokens(comp)

        # 5. Gradient Economy Scaling (Phi)
        if is_corr:
            phi = max(0.0, (l_opt - toks) / l_opt)
            r_economy = r_correct * (1.0 + 0.5 * phi)
        else:
            r_economy = r_correct

        # 6. Elastic Length Punishment (Psi) - Zero penalty if within L_max!
        excess = max(0, toks - l_max)
        psi = excess * l_lambda

        # 7. Repetition and Web Debris Penalty (Omega)
        has_repeat = detect_ngram_repetition(comp)
        has_debris = detect_web_debris(comp)
        omega = 0.0
        if has_repeat:
            omega += 2.5
        if has_debris:
            omega += 2.0

        # 8. Clean Closure Bonus (+0.5 if clean EOS within 4 tokens of answer)
        r_closure = 0.0
        tail_after_ans = ""
        for tag in ("</ans>", "</reg:scratchpad>", "</reg:deduce>", "</reg:bypass>", "</reg:intent>", "</think>"):
            if tag in comp:
                tail_after_ans = comp.split(tag)[-1].strip()
                break

        if is_corr and not has_repeat and not has_debris:
            tail_toks = len(tail_after_ans.split())
            if tail_toks <= 4:
                r_closure = 0.5
            elif tail_toks > 25:
                r_closure = -0.5

        # Total continuous gradient reward
        score = r_economy - psi - omega + r_closure
        rewards.append(float(score))

    return rewards


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
