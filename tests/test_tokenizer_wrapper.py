"""
Unit Tests for SymboLM Dedicated Register Tokenizer
===================================================
Author: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import unittest
from transformers import AutoTokenizer
from symbolm.tokenizer import setup_register_tokenizer, measure_wrapper_token_count, REGISTER_SPECIAL_TOKENS


class TestRegisterTokenizer(unittest.TestCase):
    def setUp(self):
        # Test on SmolLM2-1.7B-Instruct (LLaMA architecture, 49k BPE vocab)
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

    def test_wrapper_compression(self):
        # Baseline before setup: fragmented subwords
        cost_before = measure_wrapper_token_count(self.tokenizer, "bypass", "K")
        self.assertGreater(cost_before, 3, "Baseline tokenizer should fragment XML tags")

        # After setup: atomic register delimiters
        tok_after, _ = setup_register_tokenizer(self.tokenizer)
        cost_after = measure_wrapper_token_count(tok_after, "bypass", "K")

        # The opening <reg:bypass> is 1 token, K is 1 token, </reg:bypass> is 1 token -> Exactly 3 tokens!
        self.assertEqual(cost_after, 3, f"Expected 3 tokens after special token registration, got {cost_after}")

        # Check all 10 tokens exist in vocabulary
        for tok in REGISTER_SPECIAL_TOKENS:
            token_id = tok_after.convert_tokens_to_ids(tok)
            self.assertIsNotNone(token_id)
            self.assertNotEqual(token_id, tok_after.unk_token_id)


if __name__ == "__main__":
    unittest.main()
