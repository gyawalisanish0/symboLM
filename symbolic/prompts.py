"""
Canonical prompts and manual chat formatting for SymboLM.
=========================================================

The published Qwen/DeepSeek chat template in ``checkpoints/symbolm_v2_adapter``
drops everything before ``</think>`` from an assistant message.  Using that
template for SFT silently trains the model on the final answer only, not on the
reasoning trace.  This module therefore builds the exact prompt/target pair with
the tokenizer's real special tokens and never relies on the chat template for
training data.

Contract
--------
Reasoning target::

    <reg:plan>...</reg:plan>
    <reg:deduce>...</reg:deduce>
    <reg:verify>...</reg:verify>          # optional
    </think>
    <ans>...</ans>
    <EOS>

Atomic target::

    <reg:bypass>...</reg:bypass>
    <EOS>

The prompt always ends with ``<|Assistant|><think>\\n`` for the register model,
so the generated continuation begins at ``<reg:plan>`` or ``<reg:bypass>``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# The exact system prompt used for Arm C in both training and evaluation.
REGISTER_SYSTEM_PROMPT = (
    "You are SymboLM. Solve using typed registers: <reg:plan> for invariants, "
    "<reg:deduce> for deductive reduction (conclude with ∴ ans = <value>), "
    "<reg:verify> for invariant check, then close with </think> and emit "
    "<ans><value></ans> for the final answer. If the problem is an atomic fact, "
    "emit <reg:bypass><value></reg:bypass> and stop."
)

COT_SYSTEM_PROMPT = (
    "You are a mathematical assistant. Solve the problem step-by-step using "
    "clear English reasoning. Put your final answer inside \\boxed{}."
)

ANSWER_ONLY_SYSTEM_PROMPT = (
    "You are a mathematical assistant. Provide ONLY the final numeric or "
    "algebraic answer with no explanation. Put your final answer inside \\boxed{}."
)

SANDBOX_SYSTEM_PROMPT = (
    "You are a mathematical assistant. Solve step-by-step. You may write small "
    "Python expressions in ```python ... ``` blocks to calculate arithmetic. "
    "Put your final answer inside \\boxed{}."
)

BYPASS_SYSTEM_PROMPT = (
    "You are SymboLM. Emit <reg:bypass>ans</reg:bypass> for atomic facts."
)

QWEN_USER_TOKEN = "<｜User｜>"
QWEN_ASSISTANT_TOKEN = "<｜Assistant｜>"
QWEN_THINK_OPEN = "<think>\n"

KNOWN_LOOP_MARKERS = (
    "bands:",
    "letterSpacing",
    "tatto",
    "@endsection",
    "zbollah",
    "kube2023",
)

EOS_MARKERS = (
    "<｜end▁of▁sentence｜>",
    "<|endoftext|>",
    "<｜end of sentence｜>",
)


def _token_available(tokenizer: Any, token: str) -> bool:
    """Return True when *token* is a known special token in the tokenizer."""
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            return False
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if unk_id is not None and token_id == unk_id:
            return False
        return isinstance(token_id, int) and token_id >= 0
    except Exception:
        return False


def uses_qwen_chat_tokens(tokenizer: Any) -> bool:
    """True for the DeepSeek-R1-Distill-Qwen tokenizer family."""
    return _token_available(tokenizer, QWEN_USER_TOKEN) and _token_available(
        tokenizer, QWEN_ASSISTANT_TOKEN
    )


def build_chat_prompt(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    open_think: bool = True,
) -> str:
    """
    Build the exact prompt string used for SFT and evaluation.

    For Qwen/DeepSeek tokenizers this is constructed manually so the result is
    deterministic and is a strict prefix of the training string.  For other
    tokenizers we fall back to ``apply_chat_template`` and append ``<think>``
    only when the tokenizer did not already add it.
    """
    if uses_qwen_chat_tokens(tokenizer):
        bos = tokenizer.bos_token or ""
        text = f"{bos}{system_prompt}{QWEN_USER_TOKEN}{user_prompt}{QWEN_ASSISTANT_TOKEN}"
        if open_think:
            text += QWEN_THINK_OPEN
        return text

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as exc:
        raise ValueError(
            "Tokenizer has no usable chat template and no Qwen chat tokens. "
            "Cannot build a prompt without guessing the chat format."
        ) from exc
    if open_think and "<think>" not in text:
        text += QWEN_THINK_OPEN
    return text


def normalize_target_body(target: str) -> str:
    """
    Normalize an assistant target for the manual prompt.

    - Remove the optional opening ``<think>`` line (the prompt already has it).
    - Remove a trailing EOS marker if the source data included one.
    """
    body = (target or "").strip()
    for prefix in ("<think>\n", "<think>"):
        if body.startswith(prefix):
            body = body[len(prefix):]
            break
    for eos in EOS_MARKERS:
        if body.endswith(eos):
            body = body[: -len(eos)].rstrip()
            break
    return body


def build_full_text(tokenizer: Any, prompt: str, target_body: str) -> str:
    """Append the target body and exactly one real EOS token to the prompt."""
    body = normalize_target_body(target_body)
    eos = tokenizer.eos_token or ""
    if eos and body.endswith(eos):
        body = body[: -len(eos)].rstrip()
    return f"{prompt}{body}{eos}"


def classify_target_body(target_body: str) -> Tuple[str, str]:
    """
    Classify a normalized target body.

    Returns ``(kind, reason)`` where kind is one of:
      reasoning, bypass, dialogue, contrastive, invalid_reasoning,
      invalid_bypass, invalid
    """
    body = normalize_target_body(target_body)
    if not body:
        return "invalid", "empty_target"

    if "<checkpoint:" in body or "<backtrack:" in body:
        return "contrastive", "contains_contrastive_branch"

    if "</reg:intent>" in body and "</ans>" not in body:
        return "dialogue", "dialogue_target_out_of_scope"

    if "</ans>" in body:
        if not body.startswith("<reg:plan>"):
            return "invalid_reasoning", "reasoning_target_prefix"
        if "<reg:bypass>" in body or "</reg:bypass>" in body:
            return "invalid_reasoning", "reasoning_target_contains_bypass"
        required = (
            "<reg:plan>",
            "</reg:plan>",
            "<reg:deduce>",
            "</reg:deduce>",
            "</think>",
            "<ans>",
            "</ans>",
        )
        missing = [tag for tag in required if tag not in body]
        if missing:
            return "invalid_reasoning", "missing:" + ",".join(missing)
        for tag in (
            "<reg:plan>",
            "</reg:plan>",
            "<reg:deduce>",
            "</reg:deduce>",
            "<ans>",
            "</ans>",
        ):
            if body.count(tag) != 1:
                return "invalid_reasoning", f"duplicate_tag:{tag}"
        # The declared answer must be the last thing in the training target.
        # A trailing number or a second answer block is not a valid contract.
        if not body.rstrip().endswith("</ans>"):
            return "invalid_reasoning", "answer_not_terminal"
        positions = [
            body.find("<reg:plan>"),
            body.find("</reg:plan>"),
            body.find("<reg:deduce>"),
            body.find("</reg:deduce>"),
            body.find("</think>"),
            body.find("<ans>"),
            body.find("</ans>"),
        ]
        if positions != sorted(positions):
            return "invalid_reasoning", "tag_order"
        return "reasoning", "canonical_reasoning_target"

    if "</reg:bypass>" in body:
        if "<reg:bypass>" not in body:
            return "invalid_bypass", "missing:<reg:bypass>"
        if "<ans>" in body or "</ans>" in body:
            return "invalid_bypass", "bypass_target_contains_ans"
        if "<reg:plan>" in body or "<reg:deduce>" in body:
            return "invalid_bypass", "bypass_target_contains_reasoning"
        if "<think>" in body or "</think>" in body:
            return "invalid_bypass", "bypass_target_contains_think"
        if body.count("<reg:bypass>") != 1 or body.count("</reg:bypass>") != 1:
            return "invalid_bypass", "bypass_tag_count"
        if not body.rstrip().endswith("</reg:bypass>"):
            return "invalid_bypass", "bypass_not_terminal"
        if not re.search(r"^<reg:bypass>\s*[^<]+\s*</reg:bypass>\s*$", body, re.DOTALL):
            return "invalid_bypass", "empty_or_extra_bypass_content"
        return "bypass", "canonical_bypass_target"

    if "<ans>" in body or "<reg:bypass>" in body:
        return "invalid", "missing_closing_terminal_tag"

    return "invalid", "no_terminal_answer_tag"


def is_valid_target_body(target_body: str) -> bool:
    return classify_target_body(target_body)[0] in {"reasoning", "bypass"}


def extract_target_answer(target_body: str) -> Optional[str]:
    """Extract the declared answer from a training target, for diagnostics."""
    body = normalize_target_body(target_body)
    match = re.search(r"<ans>\s*([^<]+?)\s*</ans>", body, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"<reg:bypass>\s*([^<]+?)\s*</reg:bypass>", body, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def contains_loop_marker(text: str) -> Optional[str]:
    """Return the first known degenerate-loop marker found, if any."""
    for marker in KNOWN_LOOP_MARKERS:
        if marker in (text or ""):
            return marker
    return None


def system_prompt_for_arm(arm: str, is_atomic: bool = False) -> str:
    """Single source of truth for the system prompt of each eval arm."""
    if arm == "A" or arm == "F":
        return COT_SYSTEM_PROMPT
    if arm == "B":
        return ANSWER_ONLY_SYSTEM_PROMPT
    if arm == "C":
        return REGISTER_SYSTEM_PROMPT
    if arm == "D":
        return BYPASS_SYSTEM_PROMPT if is_atomic else COT_SYSTEM_PROMPT
    if arm == "E":
        return SANDBOX_SYSTEM_PROMPT
    raise ValueError(f"Unknown arm: {arm}")
