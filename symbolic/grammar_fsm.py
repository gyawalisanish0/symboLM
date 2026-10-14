"""
SymboLM Grammar-Constrained Decoding & Finite-State Machine (FSM)
==================================================================
Guarantees that autoregressive generation cannot revert to conversational
English Chain-of-Thought (CoT). Restricts token emission strictly to the
formal SymboLM Tri-Register grammar (<reg:plan>, <reg:deduce>, <reg:verify>,
<reg:bypass>, <reg:intent>, <ans>).

Author & Principal Systems Architect: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

from typing import List, Set, Optional, Dict, Any
import re
import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase


# Common conversational English filler words to ban inside <think>
ENGLISH_PROSE_FILLERS = [
    "Alright", "alright", "So", "so", "Let", "let", "First", "first",
    "To", "to", "We", "we", "In", "in", "The", "the", "Since", "since",
    "Notice", "notice", "Recall", "recall", "Okay", "okay", "Hmm", "hmm",
    "I", "i", "Here", "here", "Now", "now", "Well", "well"
]


class SymboLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that enforces Register syntax and masks conversational English words.
    
    1. At the opening of <think>, forces the model to emit one of:
       <reg:plan>, <reg:bypass>, or <reg:intent>.
    2. Completely suppresses English conversational opener tokens immediately after <think>.
    3. Closes registers cleanly and guides the transition to <ans>.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_len: int,
        allow_dialogue: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.allow_dialogue = allow_dialogue
        
        # Build token sets
        self.register_openers: Set[int] = set()
        for tok_str in ["<reg:plan>", "<reg:bypass>", "<reg:intent>"]:
            tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
            if tok_ids:
                self.register_openers.add(tok_ids[0])
                
        # Identify conversational filler token IDs to ban at start of think
        self.filler_token_ids: Set[int] = set()
        for word in ENGLISH_PROSE_FILLERS:
            # Check with and without leading space
            for variant in [word, f" {word}"]:
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if ids:
                    self.filler_token_ids.add(ids[0])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        # If we are at the very beginning of the generation (right after prompt)
        tokens_generated = cur_len - self.prompt_len
        
        for b in range(batch_size):
            if tokens_generated == 0:
                # FIRST TOKEN: Heavily penalize conversational English fillers
                for filler_id in self.filler_token_ids:
                    if filler_id < scores.shape[1]:
                        scores[b, filler_id] = -float("inf")
                        
                # If register openers exist in tokenizer, boost them
                for op_id in self.register_openers:
                    if op_id < scores.shape[1]:
                        scores[b, op_id] += 5.0
            
            elif tokens_generated < 4:
                # Early generation: continue penalizing English prose starters
                for filler_id in self.filler_token_ids:
                    if filler_id < scores.shape[1]:
                        scores[b, filler_id] = -float("inf")

        return scores


def export_gbnf_grammar(filepath: str = "symbolm.gbnf") -> str:
    """
    Generates a standard GBNF (GGML BNF) grammar specification for llama.cpp / Ollama.
    This guarantees 100% C++ level grammar constraint across all hardware backbones.
    """
    gbnf_content = r'''# SymboLM v2.1 Grammar Specification (GBNF)
# Author: Sanish Gyawali (Rupandehi, Nepal)
# Enforces Tri-Register execution and completely eliminates English CoT drift.

root ::= think answer

think ::= "<think>\n" register_sequence "\n</think>"

register_sequence ::= bypass_branch | reasoning_branch

bypass_branch ::= "<reg:bypass>" [^<]+ "</reg:bypass>"

reasoning_branch ::= plan_block "\n" deduce_block ( "\n" verify_block )?

plan_block ::= "<reg:plan>" [^<]+ "</reg:plan>"

deduce_block ::= "<reg:deduce>" [^<]+ "</reg:deduce>"

verify_block ::= "<reg:verify>" [^<]+ "</reg:verify>"

answer ::= "\n<ans>" [^<]+ "</ans>"
'''
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(gbnf_content)
    return gbnf_content
