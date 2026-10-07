"""
SymboLM Symbolic Reasoning Grammar
====================================
A compact DSL for expressing reasoning chains with maximum token efficiency.

Design principles:
  1. One operator = one token (unicode chars preferred)
  2. No filler words — every token carries logical content
  3. Composable — operations chain naturally with |
  4. Verifiable — answer can be extracted with a regex
  5. Readable — humans can learn it in ~10 minutes

Grammar overview:
  <trace>    ::= <step> ("|" <step>)* "∴" <conclusion>
  <step>     ::= <assignment> | <operation> | <branch> | <verify> | <quantify>
  <conclusion>::= "ans=" <value> | "ans∈" <set> | "conclude(" <expr> ")"

Token budget comparison vs English:
  English:  "Let me work through this. First I need to find X..."  ~20 tokens
  Symbolic: "hyp(X=?) | derive(X) | solve→X=val ∴ ans=val"       ~10 tokens
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Symbol vocabulary — these will be added as single tokens to the tokenizer
# ---------------------------------------------------------------------------

LOGICAL_OPS = {
    "→":  "implies / leads to",
    "∴":  "therefore (conclusion marker)",
    "∵":  "because (reason marker)",
    "∧":  "logical AND",
    "∨":  "logical OR",
    "¬":  "logical NOT",
    "⊕":  "exclusive OR",
    "≡":  "equivalent to",
    "≠":  "not equal",
    "≤":  "less than or equal",
    "≥":  "greater than or equal",
    "∈":  "element of",
    "∉":  "not element of",
    "⊂":  "subset of",
    "⊃":  "superset of",
    "∀":  "for all",
    "∃":  "there exists",
    "∄":  "there does not exist",
    "∩":  "intersection",
    "∪":  "union",
    "∅":  "empty set",
    "✓":  "verified correct",
    "✗":  "verified incorrect / backtrack",
    "↑":  "maximize / increase",
    "↓":  "minimize / decrease",
    "~":  "approximately",
    "∞":  "infinity",
}

KEYWORDS = {
    # Reasoning structure
    "hyp":       "hypothesis / given",
    "derive":    "derive / compute next",
    "conclude":  "final conclusion",
    "let":       "assign variable",
    "verify":    "verify a claim",
    "backtrack": "undo last step, try alternative",
    "case":      "branch into cases",
    "base":      "base case (induction)",
    "step":      "inductive step",

    # Math shortcuts
    "∂":         "derivative (d/dx)",
    "∫":         "integral",
    "Σ":         "sum",
    "Π":         "product",
    "√":         "square root",
    "ans":       "final answer (reserved)",

    # Control flow
    "if":        "conditional branch",
    "else":      "else branch",
    "loop":      "repeat until condition",
    "try":       "attempt (may fail)",

    # Common domain shorthands
    "gcd":       "greatest common divisor",
    "lcm":       "least common multiple",
    "mod":       "modulo",
    "abs":       "absolute value",
    "max":       "maximum",
    "min":       "minimum",
    "len":       "length / cardinality",
    "sort":      "sorted collection",
}

# Separator tokens
SEPARATOR     = "|"   # step separator within a trace
CONCLUSION    = "∴"   # marks the final answer step
REASON        = "∵"   # marks a justification
IMPLIES       = "→"   # causal / logical implication
CORRECT       = "✓"   # verification passed
WRONG         = "✗"   # verification failed → triggers backtrack

# Special tokens for training
THINK_OPEN    = "<think>"
THINK_CLOSE   = "</think>"
ANS_OPEN      = "<ans>"
ANS_CLOSE     = "</ans>"

ALL_SPECIAL_TOKENS = list(LOGICAL_OPS.keys()) + [
    THINK_OPEN, THINK_CLOSE, ANS_OPEN, ANS_CLOSE
]

# ---------------------------------------------------------------------------
# Pattern library — common reasoning idioms in the DSL
# ---------------------------------------------------------------------------

PATTERNS = {
    "arithmetic_chain": (
        "hyp(start={v}) | {v}+={delta}→{v}={r1} | {v}-={d2}→{v}={r2} ∴ ans={r2}",
        "Chain of arithmetic operations on a variable"
    ),
    "derivative_zero": (
        "f(x)={expr} | ∂f={deriv} | ∂f=0→x={xval} | ∂²f={d2}{'>'if d2>0 else '<'}0→{'min' if d2>0 else 'max'} ∴ ans=({xval},{fval})",
        "Find critical point via differentiation"
    ),
    "case_split": (
        "case(x>0)[{pos_path}]|case(x=0)[{zero_path}]|case(x<0)[{neg_path}] ∴ ans={result}",
        "Split into exhaustive cases"
    ),
    "verify_and_conclude": (
        "{reasoning} | verify({claim})✓ ∴ ans={val}",
        "Verify a derived claim before concluding"
    ),
    "backtrack_pattern": (
        "try({attempt}) | verify({check})✗ | backtrack | {alternative} ∴ ans={val}",
        "Attempt, fail verification, backtrack, retry"
    ),
    "induction": (
        "base(n=0): {base_case} | step(n→n+1): hyp(P(n))→derive(P(n+1)) ∴ ∀n: P(n)",
        "Mathematical induction"
    ),
    "word_problem": (
        "let({vars}) | {operations} ∴ ans={val}",
        "Generic word problem with variable assignment"
    ),
    "counting": (
        "let(S={set}) | len(S)={n} | filter(S,{cond})→S'={filtered} | len(S')={k} ∴ ans={k}",
        "Count elements satisfying a condition"
    ),
}

# ---------------------------------------------------------------------------
# Symbolic trace format
# ---------------------------------------------------------------------------

TRACE_FORMAT = """\
{think_open}
{symbolic_steps}
{think_close}
{ans_open}{answer}{ans_close}\
"""

EXAMPLE_TRACES = [
    {
        "problem": "Mary has 5 apples. She gives 2 to John. Then buys 3 more. How many?",
        "english_cot": (
            "Let me think step by step. Mary starts with 5 apples. She gives 2 to John, "
            "so she has 5 - 2 = 3 apples. Then she buys 3 more, so 3 + 3 = 6. The answer is 6."
        ),
        "symbolic_cot": "let(M=5) | M-=2→M=3 | M+=3→M=6 ∴ ans=6",
        "answer": "6",
        "english_tokens": 52,
        "symbolic_tokens": 16,
        "compression_ratio": 0.69,
    },
    {
        "problem": "Find the maximum of f(x) = -x² + 4x - 1.",
        "english_cot": (
            "To find the maximum, I take the derivative: f'(x) = -2x + 4. "
            "Setting equal to zero: -2x + 4 = 0, x = 2. "
            "The second derivative is -2 < 0, confirming a maximum. "
            "f(2) = -4 + 8 - 1 = 3. The maximum is 3 at x = 2."
        ),
        "symbolic_cot": "f(x)=-x²+4x-1 | ∂f=-2x+4 | ∂f=0→x=2 | ∂²f=-2<0→max | f(2)=3 ∴ ans=(x=2,max=3)",
        "answer": "3",
        "english_tokens": 68,
        "symbolic_tokens": 24,
        "compression_ratio": 0.65,
    },
    {
        "problem": "Is 2^10 > 1000?",
        "english_cot": (
            "I need to compute 2 to the power of 10. "
            "2^1=2, 2^2=4, 2^4=16, 2^8=256, 2^10=1024. "
            "1024 > 1000, so yes."
        ),
        "symbolic_cot": "2^10=1024 | verify(1024>1000)✓ ∴ ans=True",
        "answer": "True",
        "english_tokens": 38,
        "symbolic_tokens": 11,
        "compression_ratio": 0.71,
    },
    {
        "problem": "Solve for x: 3x + 6 = 21",
        "english_cot": (
            "I need to isolate x. Subtract 6 from both sides: 3x = 15. "
            "Divide both sides by 3: x = 5. Let me verify: 3(5)+6=21 ✓"
        ),
        "symbolic_cot": "3x+6=21 | 3x=21-6→3x=15 | x=15/3→x=5 | verify(3·5+6=21)✓ ∴ ans=5",
        "answer": "5",
        "english_tokens": 44,
        "symbolic_tokens": 21,
        "compression_ratio": 0.52,
    },
    {
        "problem": "If all cats are animals and Whiskers is a cat, is Whiskers an animal?",
        "english_cot": (
            "This is a syllogism. The major premise is: all cats are animals. "
            "The minor premise is: Whiskers is a cat. "
            "By modus ponens, Whiskers is an animal."
        ),
        "symbolic_cot": "hyp(∀x: cat(x)→animal(x)) | hyp(cat(Whiskers)) | cat(Whiskers)→animal(Whiskers) ∴ ans=True",
        "answer": "True",
        "english_tokens": 46,
        "symbolic_tokens": 18,
        "compression_ratio": 0.61,
    },
]

# ---------------------------------------------------------------------------
# Validator — soft grammar checker (does not enforce strict grammar,
# just checks for key structural markers)
# ---------------------------------------------------------------------------

# Regex to extract final answer from a symbolic trace
ANSWER_PATTERN = re.compile(
    r"ans\s*=\s*([^\s|✓✗\n]+)"  # ans=<value>
    r"|"
    r"<ans>(.*?)</ans>",          # <ans>value</ans>
    re.DOTALL,
)


@dataclass
class ValidationResult:
    is_valid: bool
    issues: list[str] = field(default_factory=list)
    extracted_answer: str | None = None
    token_count_estimate: int = 0
    has_conclusion: bool = False
    has_steps: bool = False
    english_leak_score: float = 0.0  # 0=pure symbolic, 1=pure English


def validate_trace(trace: str) -> ValidationResult:
    """
    Soft-validate a symbolic reasoning trace.
    Returns a ValidationResult with any issues found.
    """
    issues: list[str] = []

    # Check for conclusion marker
    has_conclusion = "∴" in trace or "<ans>" in trace
    if not has_conclusion:
        issues.append("Missing conclusion marker (∴ or <ans>)")

    # Check for step separator (multi-step reasoning)
    has_steps = "|" in trace
    if not has_steps:
        issues.append("No step separators (|) found — trace may be too short")

    # Extract answer
    extracted_answer: str | None = None
    m = ANSWER_PATTERN.search(trace)
    if m:
        extracted_answer = (m.group(1) or m.group(2) or "").strip()
    else:
        issues.append("Could not extract answer from trace")

    # Estimate English leak — count common filler words
    english_fillers = [
        r"\blet me\b", r"\bi need to\b", r"\bfirst\b.*\bthen\b",
        r"\bstep by step\b", r"\btherefore\b", r"\bthus\b",
        r"\bwe can see\b", r"\bnotice that\b", r"\bsince\b.*\bwe\b",
    ]
    leak_hits = sum(
        1 for pat in english_fillers
        if re.search(pat, trace.lower())
    )
    english_leak_score = min(leak_hits / max(len(english_fillers), 1), 1.0)
    if english_leak_score > 0.3:
        issues.append(
            f"High English leak score ({english_leak_score:.2f}) — "
            "model is using too much natural language in the trace"
        )

    # Rough token count (split on whitespace + individual unicode chars)
    token_count_estimate = _estimate_tokens(trace)

    return ValidationResult(
        is_valid=len(issues) == 0,
        issues=issues,
        extracted_answer=extracted_answer,
        token_count_estimate=token_count_estimate,
        has_conclusion=has_conclusion,
        has_steps=has_steps,
        english_leak_score=english_leak_score,
    )


def extract_answer(text: str) -> str | None:
    """Extract the final answer from a full model output (including <ans> tags or ∴ ans=)."""
    # Try <ans> tags first (most reliable)
    m = re.search(r"<ans>(.*?)</ans>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try ∴ ans= pattern
    m = re.search(r"∴\s*ans\s*=\s*([^\s|✓✗\n<]+)", text)
    if m:
        return m.group(1).strip()
    return None


def _estimate_tokens(text: str) -> int:
    """
    Rough token count estimate.
    Splits on whitespace and counts individual unicode symbol chars separately.
    Not exact — real count depends on the tokenizer vocabulary.
    """
    count = 0
    for word in text.split():
        # Each unicode operator likely maps to 1 token after tokenizer extension
        symbol_chars = sum(1 for c in word if c in LOGICAL_OPS)
        regular_chars = len(word) - symbol_chars
        count += symbol_chars + max(1, regular_chars // 4)  # ~4 chars per token
    return count


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------

def compression_ratio(english_cot: str, symbolic_cot: str) -> float:
    """
    Returns what fraction of tokens were saved by using symbolic reasoning.
    0.0 = no saving, 1.0 = 100% saving (impossible), 0.5 = 50% fewer tokens.
    """
    eng_tokens = _estimate_tokens(english_cot)
    sym_tokens = _estimate_tokens(symbolic_cot)
    if eng_tokens == 0:
        return 0.0
    return max(0.0, 1.0 - sym_tokens / eng_tokens)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

def demo():
    print("=" * 60)
    print("SymboLM Symbolic Grammar — Example Traces")
    print("=" * 60)

    avg_compression = 0.0
    for ex in EXAMPLE_TRACES:
        ratio = compression_ratio(ex["english_cot"], ex["symbolic_cot"])
        avg_compression += ratio
        result = validate_trace(ex["symbolic_cot"])
        print(f"\nProblem : {ex['problem']}")
        print(f"Symbolic: {ex['symbolic_cot']}")
        print(f"Answer  : {result.extracted_answer}")
        print(f"Valid   : {result.is_valid}  |  Compression: {ratio:.0%}  |  Tokens: {result.token_count_estimate}")
        if result.issues:
            for issue in result.issues:
                print(f"  ⚠ {issue}")

    avg_compression /= len(EXAMPLE_TRACES)
    print(f"\n{'='*60}")
    print(f"Average token compression across examples: {avg_compression:.0%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    demo()
