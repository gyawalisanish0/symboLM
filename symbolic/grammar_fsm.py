"""
SymboLM grammar helpers and opening-token bias.
================================================

Two different mechanisms are often confused in this repository:

1. ``SymboLogitsProcessor`` is an *opening-token bias*.  It suppresses common
   conversational openers for the first few generated tokens.  It is **not** a
   complete grammar-constrained decoder.
2. ``symbolm.gbnf`` is the real llama.cpp/Ollama grammar.  It is what should be
   used for deployment when hard structural enforcement is required.

The rescue experiment measures format validity with ``validate_register_trace``
rather than pretending the HF logits processor enforces the whole grammar.

Canonical generated continuation (the prompt already ends with ``<think>\\n``)::

    <reg:plan>...</reg:plan>
    <reg:deduce>...</reg:deduce>
    <reg:verify>...</reg:verify>          # optional
    </think>
    <ans>...</ans>

or, for atomic facts::

    <reg:bypass>...</reg:bypass>
"""

from typing import List, Set, Optional, Dict, Any, Tuple
import re
import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase


# Common conversational English filler words to ban at the start of generation.
ENGLISH_PROSE_FILLERS = [
    "Alright", "alright", "So", "so", "Let", "let", "First", "first",
    "To", "to", "We", "we", "In", "in", "The", "the", "Since", "since",
    "Notice", "notice", "Recall", "recall", "Okay", "okay", "Hmm", "hmm",
    "I", "i", "Here", "here", "Now", "now", "Well", "well",
]


class SymboLogitsProcessor(LogitsProcessor):
    """
    Bias the first few generated tokens away from conversational English.

    This is deliberately *not* called a grammar enforcer.  It:
      1. suppresses common English opener tokens immediately after ``<think>``;
      2. boosts the three legal register openers;
      3. continues suppressing openers for the first four generated tokens.

    Use ``symbolm.gbnf`` in llama.cpp/Ollama or ``validate_register_trace`` in
    evaluation when you need actual structural enforcement.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_len: int,
        allow_dialogue: bool = False,
        early_token_limit: int = 4,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.allow_dialogue = allow_dialogue
        self.early_token_limit = max(1, int(early_token_limit))

        self.register_openers: Set[int] = set()
        for tok_str in ["<reg:plan>", "<reg:bypass>", "<reg:intent>"]:
            tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
            if tok_ids:
                self.register_openers.add(tok_ids[0])

        self.filler_token_ids: Set[int] = set()
        for word in ENGLISH_PROSE_FILLERS:
            for variant in [word, f" {word}"]:
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if ids:
                    self.filler_token_ids.add(ids[0])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        tokens_generated = cur_len - self.prompt_len

        for b in range(batch_size):
            if 0 <= tokens_generated < self.early_token_limit:
                for filler_id in self.filler_token_ids:
                    if filler_id < scores.shape[1]:
                        scores[b, filler_id] = -float("inf")

            if tokens_generated == 0:
                for op_id in self.register_openers:
                    if op_id < scores.shape[1]:
                        scores[b, op_id] += 5.0

        return scores


def validate_register_trace(text: str, arm: str = "C") -> Tuple[bool, str]:
    """
    Validate a generated trace against the canonical register contract.

    This delegates to the strict scorer so there is exactly one definition of
    "valid format" in the repository.
    """
    from eval.strict_scorer import validate_register_trace as _validate

    return _validate(text, arm=arm)


def export_gbnf_grammar(filepath: str = "symbolm.gbnf") -> str:
    """
    Write the real GBNF grammar for llama.cpp / Ollama.

    The grammar is written for the *continuation* after the prompt's
    ``<think>\\n``.  An optional ``<think>\\n`` is allowed so the same grammar
    also works when the prompt does not pre-open the trace.
    """
    gbnf_content = r'''# SymboLM v3 Grammar Specification (GBNF)
# Canonical continuation after the prompt's <think> newline.
# Reasoning: plan -> deduce -> optional verify -> </think> -> <ans>.
# Atomic:    <reg:bypass>...</reg:bypass> and stop.
# Author: Sanish Gyawali (Rupandehi, Nepal)

root ::= think_open? body

body ::= reasoning | bypass

reasoning ::= plan "\n" deduce ("\n" verify)? "\n</think>\n" answer

bypass ::= "<reg:bypass>" [^<]+ "</reg:bypass>"

plan ::= "<reg:plan>" [^<]+ "</reg:plan>"

deduce ::= "<reg:deduce>" [^<]+ "</reg:deduce>"

verify ::= "<reg:verify>" [^<]+ "</reg:verify>"

answer ::= "<ans>" [^<]+ "</ans>"

think_open ::= "<think>\n"
'''
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(gbnf_content)
    return gbnf_content
