"""
Unit Tests for SymboLM Grammar FSM & LogitsProcessor
====================================================
Author & Principal Systems Architect: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import unittest
import os
import torch
from transformers import AutoTokenizer
from symbolic.grammar_fsm import (
    SymboLogitsProcessor,
    export_gbnf_grammar,
    validate_register_trace,
    ENGLISH_PROSE_FILLERS,
)


class TestGrammarFSM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use local adapter tokenizer if exists, else fast fallback
        cls.tok_path = "checkpoints/symbolm_v2_adapter"
        if not os.path.exists(cls.tok_path):
            cls.tok_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.tok_path)

    def test_export_gbnf_grammar(self):
        out_path = "scratch/test_symbolm.gbnf"
        os.makedirs("scratch", exist_ok=True)
        content = export_gbnf_grammar(out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertIn("root ::= think_open? body", content)
        self.assertIn("<reg:plan>", content)
        self.assertIn("<reg:deduce>", content)
        self.assertIn("<reg:bypass>", content)
        self.assertIn("<ans>", content)

    def test_register_trace_validator(self):
        trace = (
            "<reg:plan>concept: arithmetic</reg:plan>\n"
            "<reg:deduce>2+2=4 ∴ ans=4</reg:deduce>\n"
            "<reg:verify>verify(4==4)</reg:verify>\n"
            "</think>\n<ans>4</ans>"
        )
        ok, reason = validate_register_trace(trace, arm="C")
        self.assertTrue(ok, reason)

        bad = trace.replace("</think>\n", "")
        ok, reason = validate_register_trace(bad, arm="C")
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_think_close")

    def test_logits_processor_masks_english_fillers(self):
        prompt = "<｜User｜>Find the value of x.<｜Assistant｜><think>\n"
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = input_ids.shape[1]
        
        proc = SymboLogitsProcessor(self.tokenizer, prompt_len=prompt_len)
        vocab_size = len(self.tokenizer)
        scores = torch.zeros((1, vocab_size), dtype=torch.float32)
        
        # Step 0: Exactly at start of think
        modified_scores = proc(input_ids, scores)
        
        # Check that common conversational fillers have -inf
        for word in ["Alright", "So", "Let", "First"]:
            tok_id = self.tokenizer.encode(word, add_special_tokens=False)
            if tok_id:
                self.assertEqual(
                    modified_scores[0, tok_id[0]].item(), 
                    -float("inf"),
                    f"Word '{word}' (id {tok_id[0]}) should be masked to -inf!"
                )


if __name__ == "__main__":
    unittest.main()
