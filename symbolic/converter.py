"""
English CoT → Symbolic CoT Converter
======================================
Converts natural language reasoning chains into the SymboLM symbolic DSL.

Two modes:
  1. Rule-based (offline, no API) — pattern matching for common structures
  2. LLM-assisted (uses a small local model or API) — for complex conversions

The rule-based converter handles:
  - GSM8K style word problems (arithmetic chains)
  - Simple algebra (isolate variable)
  - Syllogisms / logic
  - Verification patterns

For the training dataset, we need ~5000 high-quality symbolic traces.
Strategy: rule-based for bulk, manual review for ~200 quality samples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from symbolic.grammar import (
    THINK_CLOSE,
    THINK_OPEN,
    ANS_CLOSE,
    ANS_OPEN,
    validate_trace,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConversionResult:
    symbolic_cot: str
    full_output: str          # <think>...</think><ans>...</ans>
    answer: str
    confidence: float         # 0–1, how confident we are in the conversion
    method: str               # "rule_based" | "llm_assisted" | "manual"
    validation: ValidationResult | None = None


# ---------------------------------------------------------------------------
# Rule-based patterns for GSM8K-style problems
# ---------------------------------------------------------------------------

# Each rule is (description, regex_on_english_cot, converter_fn)
# converter_fn(match, full_cot) -> symbolic_steps (list of str)

Rule = tuple[str, re.Pattern, Callable]


def _extract_numbers(text: str) -> list[str]:
    """Extract all numbers (int/float) from text."""
    return re.findall(r"-?\d+(?:\.\d+)?", text)


def _extract_final_answer(text: str) -> str | None:
    """Try to find the final answer in English CoT text."""
    patterns = [
        r"answer is[:\s]+(-?\d+(?:\.\d+)?)",
        r"= (-?\d+(?:\.\d+)?)[\.\s]*$",
        r"total.*?(-?\d+(?:\.\d+)?)[\.\s]*$",
        r"result.*?(-?\d+(?:\.\d+)?)[\.\s]*$",
        r"so.*?(-?\d+(?:\.\d+)?)[\.\s]*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    # fallback: last number in text
    nums = _extract_numbers(text)
    return nums[-1] if nums else None


# ---------------------------------------------------------------------------
# Arithmetic chain converter (GSM8K main pattern)
# ---------------------------------------------------------------------------

def convert_arithmetic_chain(english_cot: str, problem: str = "") -> ConversionResult | None:
    """
    Convert a word problem with arithmetic steps.
    Handles: starts with X, add/subtract Y, multiply/divide by Z, etc.

    Example English:
      "Mary starts with 5 apples. She gives 2 to John, so she has 3.
       Then buys 3 more, so 6. Answer is 6."

    Output: let(M=5) | M-=2→M=3 | M+=3→M=6 ∴ ans=6
    """
    text = english_cot.strip()

    # Try to extract variable assignments and operations
    steps: list[str] = []

    # Pattern: "starts with N" or "has N" or "there are N"
    init_patterns = [
        (r"(?:starts? with|has|there (?:are|is)|begins? with)\s+(\w+)\s+(-?\d+(?:\.\d+)?)", "init"),
        (r"(-?\d+(?:\.\d+)?)\s+(?:\w+)\s+(?:to begin|at start|initially)", "init_rev"),
    ]

    # Pattern: "adds N" / "gains N" / "buys N more"
    add_patterns = [
        r"(?:add|gain|buy|get|earn|receive|pick up)[s\s]+(?:up\s+)?(-?\d+(?:\.\d+)?)",
        r"\+\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s+more",
    ]

    # Pattern: "loses N" / "gives N" / "spends N"
    sub_patterns = [
        r"(?:lose[s]?|give[s]?|spend[s]?|use[s]?|remove[s]?|sell[s]?)\s+(-?\d+(?:\.\d+)?)",
        r"-\s*(-?\d+(?:\.\d+)?)",
    ]

    # Pattern: "multiplied by N" / "times N"
    mul_patterns = [
        r"(?:multipl(?:y|ied)\s+by|times)\s+(-?\d+(?:\.\d+)?)",
        r"\*\s*(-?\d+(?:\.\d+)?)",
    ]

    # Pattern: "divided by N"
    div_patterns = [
        r"(?:divid(?:e[ds]?|ing)\s+by|divided\s+by)\s+(-?\d+(?:\.\d+)?)",
        r"/\s*(-?\d+(?:\.\d+)?)",
    ]

    # Simplified: extract all numbers and operations, build a chain
    sentences = re.split(r"[.!?]+", text)
    var = "x"  # generic variable name
    current_val: float | None = None
    operations: list[str] = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        nums = _extract_numbers(sent)
        if not nums:
            continue
        num = float(nums[-1])  # most relevant number usually last

        if current_val is None:
            # First sentence — initialize variable
            current_val = num
            operations.append(f"let({var}={_fmt(num)})")
            continue

        # Detect operation type
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in ["add", "gain", "buy", "more", "earn", "receive"]):
            current_val += num
            operations.append(f"{var}+={_fmt(num)}→{var}={_fmt(current_val)}")
        elif any(kw in sent_lower for kw in ["lose", "give", "spend", "remove", "sell", "less"]):
            current_val -= num
            operations.append(f"{var}-={_fmt(num)}→{var}={_fmt(current_val)}")
        elif any(kw in sent_lower for kw in ["multipl", "times", "double", "triple"]):
            current_val *= num
            operations.append(f"{var}*={_fmt(num)}→{var}={_fmt(current_val)}")
        elif any(kw in sent_lower for kw in ["divid", "half", "split"]):
            current_val /= num
            operations.append(f"{var}/={_fmt(num)}→{var}={_fmt(current_val)}")

    if not operations or current_val is None:
        return None

    answer = _extract_final_answer(text) or _fmt(current_val)
    symbolic = " | ".join(operations) + f" ∴ ans={answer}"
    full_out = f"{THINK_OPEN}\n{symbolic}\n{THINK_CLOSE}\n{ANS_OPEN}{answer}{ANS_CLOSE}"

    result = ConversionResult(
        symbolic_cot=symbolic,
        full_output=full_out,
        answer=answer,
        confidence=0.75,
        method="rule_based",
    )
    result.validation = validate_trace(symbolic)
    return result


def _fmt(n: float) -> str:
    """Format number: show as int if it's whole, else float."""
    return str(int(n)) if n == int(n) else str(round(n, 4))


# ---------------------------------------------------------------------------
# Algebra converter (simple linear equations)
# ---------------------------------------------------------------------------

def convert_algebra(english_cot: str, problem: str = "") -> ConversionResult | None:
    """
    Handle simple algebraic reasoning: isolate variable, solve.
    Example: "3x + 6 = 21 | 3x=15 | x=5 | verify✓ ∴ ans=5"
    """
    # Look for equation pattern in problem or CoT
    eq_match = re.search(r"(\d*[a-z])\s*([+\-])\s*(\d+)\s*=\s*(\d+)", problem + " " + english_cot)
    if not eq_match:
        return None

    var, op, const, rhs = eq_match.groups()
    coeff_str = re.match(r"(\d+)[a-z]", var)
    coeff = float(coeff_str.group(1)) if coeff_str else 1.0
    var_name = var[-1]  # single letter
    rhs_val = float(rhs)
    const_val = float(const)

    if op == "+":
        isolated = rhs_val - const_val
    else:
        isolated = rhs_val + const_val

    solve_val = isolated / coeff

    # Build symbolic steps
    steps = [
        f"{var}{op}{const}={rhs}",
        f"{coeff}{var_name}={_fmt(rhs_val)}-{_fmt(const_val)}→{coeff}{var_name}={_fmt(isolated)}" if op == "+" else
        f"{coeff}{var_name}={_fmt(rhs_val)}+{_fmt(const_val)}→{coeff}{var_name}={_fmt(isolated)}",
        f"{var_name}={_fmt(isolated)}/{_fmt(coeff)}→{var_name}={_fmt(solve_val)}",
        f"verify({var}{op}{const}={rhs})✓" if op == "+" else f"verify({var}{op}{const}={rhs})✓",
    ]
    symbolic = " | ".join(steps) + f" ∴ ans={_fmt(solve_val)}"
    answer = _fmt(solve_val)
    full_out = f"{THINK_OPEN}\n{symbolic}\n{THINK_CLOSE}\n{ANS_OPEN}{answer}{ANS_CLOSE}"

    result = ConversionResult(
        symbolic_cot=symbolic,
        full_output=full_out,
        answer=answer,
        confidence=0.85,
        method="rule_based",
    )
    result.validation = validate_trace(symbolic)
    return result


# ---------------------------------------------------------------------------
# Syllogism / Logic converter
# ---------------------------------------------------------------------------

def convert_syllogism(english_cot: str, problem: str = "") -> ConversionResult | None:
    """
    Handle simple syllogisms (modus ponens, modus tollens).
    Example: "All cats are animals. Whiskers is a cat. → Whiskers is an animal."
    Output: "hyp(∀x: cat(x)→animal(x)) | hyp(cat(W)) | cat(W)→animal(W) ∴ ans=True"
    """
    text = (problem + " " + english_cot).lower()

    # Detect "all X are Y" major premise
    all_match = re.search(r"all\s+(\w+)s?\s+are\s+(\w+)s?", text)
    if not all_match:
        return None

    category, parent = all_match.groups()

    # Detect "Z is a X" minor premise
    is_a_match = re.search(rf"(\w+)\s+is\s+a(?:n)?\s+{category}", text)
    if not is_a_match:
        return None

    subject = is_a_match.group(1)
    subj_abbr = subject[0].upper()  # e.g. Whiskers → W

    symbolic = (
        f"hyp(∀x: {category}(x)→{parent}(x)) | "
        f"hyp({category}({subj_abbr})) | "
        f"{category}({subj_abbr})→{parent}({subj_abbr}) "
        f"∴ ans=True"
    )
    answer = "True"
    full_out = f"{THINK_OPEN}\n{symbolic}\n{THINK_CLOSE}\n{ANS_OPEN}{answer}{ANS_CLOSE}"

    result = ConversionResult(
        symbolic_cot=symbolic,
        full_output=full_out,
        answer=answer,
        confidence=0.80,
        method="rule_based",
    )
    result.validation = validate_trace(symbolic)
    return result


# ---------------------------------------------------------------------------
# LLM-assisted converter (uses a prompted model)
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """You are a symbolic reasoning converter for the SymboLM project.
Convert the given English chain-of-thought reasoning into a compact symbolic format.

Rules:
1. Use these operators: → (implies), ∴ (therefore/answer), | (step separator),
   ∵ (because), ∧ (and), ∨ (or), ¬ (not), ∀ (for all), ∃ (exists),
   verify()✓ or verify()✗, let(), hyp(), conclude()
2. Eliminate ALL filler words: "let me think", "I need to", "first", "then", etc.
3. Each step should be one logical operation
4. End with: ∴ ans=<value>
5. Keep variable names to 1-2 characters

Output ONLY the symbolic trace, no explanations.

Example:
English: "Mary has 5 apples. Gives 2 away so 3 left. Buys 3 more so 6 total."
Symbolic: let(M=5) | M-=2→M=3 | M+=3→M=6 ∴ ans=6
"""

def convert_with_llm(
    problem: str,
    english_cot: str,
    answer: str,
    model_fn: Callable[[str, str], str] | None = None,
) -> ConversionResult | None:
    """
    Use a language model to convert English CoT to symbolic.
    model_fn(system_prompt, user_message) -> response_str

    If model_fn is None, this returns None (LLM not available).
    """
    if model_fn is None:
        return None

    user_msg = f"Problem: {problem}\nEnglish CoT: {english_cot}\nAnswer: {answer}"
    try:
        symbolic = model_fn(LLM_SYSTEM_PROMPT, user_msg).strip()
    except Exception as e:
        print(f"LLM conversion failed: {e}")
        return None

    full_out = f"{THINK_OPEN}\n{symbolic}\n{THINK_CLOSE}\n{ANS_OPEN}{answer}{ANS_CLOSE}"
    result = ConversionResult(
        symbolic_cot=symbolic,
        full_output=full_out,
        answer=answer,
        confidence=0.70,  # LLM can hallucinate structure
        method="llm_assisted",
    )
    result.validation = validate_trace(symbolic)
    return result


# ---------------------------------------------------------------------------
# Master converter — tries rules in order, falls back to LLM
# ---------------------------------------------------------------------------

RULE_CONVERTERS: list[tuple[str, Callable]] = [
    ("algebra",    convert_algebra),
    ("syllogism",  convert_syllogism),
    ("arithmetic", convert_arithmetic_chain),
]


def convert(
    problem: str,
    english_cot: str,
    answer: str,
    model_fn: Callable | None = None,
    min_confidence: float = 0.5,
) -> ConversionResult | None:
    """
    Convert English CoT to symbolic. Tries rule-based first, then LLM.

    Returns None if no method succeeds with sufficient confidence.
    """
    # Try rules
    for rule_name, rule_fn in RULE_CONVERTERS:
        try:
            result = rule_fn(english_cot, problem)
            if result and result.confidence >= min_confidence:
                return result
        except Exception:
            continue

    # Fall back to LLM if available
    return convert_with_llm(problem, english_cot, answer, model_fn)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        {
            "problem": "Mary has 5 apples. She gives 2 to John. Then buys 3 more. How many?",
            "english_cot": "Mary starts with 5 apples. She gives 2 to John so she has 3. Then she buys 3 more so she has 6.",
            "answer": "6",
        },
        {
            "problem": "Solve 3x + 6 = 21",
            "english_cot": "Subtract 6: 3x = 15. Divide by 3: x = 5. Verify: 15+6=21 ✓",
            "answer": "5",
        },
        {
            "problem": "All cats are animals. Whiskers is a cat. Is Whiskers an animal?",
            "english_cot": "All cats are animals (major premise). Whiskers is a cat (minor premise). By modus ponens, Whiskers is an animal.",
            "answer": "True",
        },
    ]

    print("SymboLM Converter Demo\n" + "=" * 50)
    for tc in test_cases:
        result = convert(tc["problem"], tc["english_cot"], tc["answer"])
        if result:
            valid = result.validation
            print(f"\nProblem  : {tc['problem']}")
            print(f"Method   : {result.method} (confidence={result.confidence:.0%})")
            print(f"Symbolic : {result.symbolic_cot}")
            print(f"Valid    : {valid.is_valid if valid else '?'}")
            print(f"Answer   : {result.answer}")
        else:
            print(f"\nFailed to convert: {tc['problem']}")
