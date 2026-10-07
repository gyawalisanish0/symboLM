"""
SymboLM Symbolic-to-English Translator
======================================
Decompresses compact symbolic traces into readable natural English prose.
Useful when an end-user needs to inspect the full explanation in plain words.
"""

from __future__ import annotations

import re
from typing import List


def translate_symbolic_step(step: str) -> str:
    """Translates a single symbolic reasoning step into English."""
    step = step.strip()
    if not step:
        return ""

    # Variable assignment: let(var=val)
    m = re.match(r"let\(([^=]+)=([^\)]+)\)", step)
    if m:
        return f"Initialize variable {m.group(1)} to {m.group(2)}."

    # Hypothesis: hyp(expr)
    m = re.match(r"hyp\(([^\)]+)\)", step)
    if m:
        return f"Given hypothesis: {m.group(1)}."

    # Verification: verify(claim)✓
    if "verify(" in step and "✓" in step:
        m = re.search(r"verify\(([^\)]+)\)✓", step)
        if m:
            return f"Verified that {m.group(1)} holds true."

    # Operations with implication: X+=Y→X=Z
    if "→" in step:
        parts = step.split("→")
        return f"Compute {parts[0].strip()}, yielding {parts[1].strip()}."

    # Derivative
    if "∂f" in step or "∂" in step:
        return f"Differentiate: {step}."

    # Conclusion
    if "∴" in step:
        clean = step.replace("∴", "").strip()
        return f"Therefore, conclude {clean}."

    return f"Step: {step}."


def decompress_trace(trace: str) -> str:
    """Translates an entire symbolic trace into an English explanation."""
    # Remove <think> tags if present
    trace_clean = re.sub(r"</?think>", "", trace).strip()

    # Split steps by '|'
    steps = trace_clean.split("|")
    english_steps: List[str] = []

    for i, step in enumerate(steps, 1):
        tr = translate_symbolic_step(step)
        if tr:
            english_steps.append(f"{i}. {tr}")

    return "\n".join(english_steps)


if __name__ == "__main__":
    sample = "let(M=5) | M-=2→M=3 | M+=3→M=6 | verify(M=6)✓ ∴ ans=6"
    print("Original Symbolic Trace:")
    print(sample)
    print("\nDecompressed English Explanation:")
    print(decompress_trace(sample))
