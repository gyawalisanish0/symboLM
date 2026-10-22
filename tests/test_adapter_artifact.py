"""
Regression tests for the extended-tokenizer adapter artifact.

The tokenizer adds <ans> and </ans>, which forces a resized embedding matrix and
LM head.  A plain LoRA state dict does not contain those rows, so evaluation
would reserve different vectors for the terminal-answer markers.  These tests
verify that the training helper initializes genuinely added rows, does not
overwrite existing base tokens, and persists the exact rows for PeftModel.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen2Config, Qwen2ForCausalLM

from symbolic.tokenizer_ext import (
    SEMANTIC_ANCHORS,
    initialize_symbol_embeddings,
)
from training.sft_train import attach_lora_adapter


class FakeAnchorTokenizer:
    """Minimal tokenizer surface for the semantic-anchor initializer."""

    def __init__(self):
        self.added = {"<ans>": 10, "→": 11}
        self.ids = {
            "<ans>": 10,
            "→": 11,
            "answer": 1,
            "implies": 2,
            "ans": 5,
            "correct": 3,
        }
        self.unk_token_id = 0

    def get_added_vocab(self):
        return dict(self.added)

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, 0)

    def encode(self, text, add_special_tokens=False):
        return [self.ids.get(text, 0)]


class TestAdapterArtifact(unittest.TestCase):
    def test_terminal_markers_have_semantic_anchors(self):
        self.assertIn("<ans>", SEMANTIC_ANCHORS)
        self.assertIn("</ans>", SEMANTIC_ANCHORS)

    def test_initialize_added_input_and_output_rows_without_touching_base_tokens(self):
        cfg = Qwen2Config(
            vocab_size=20,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
        model = Qwen2ForCausalLM(cfg)
        with torch.no_grad():
            model.get_input_embeddings().weight[1] = 0.2
            model.get_input_embeddings().weight[2] = 0.3
            model.get_input_embeddings().weight[5] = 0.9
            model.get_output_embeddings().weight[1] = 1.2
            model.get_output_embeddings().weight[2] = 1.3
            model.get_output_embeddings().weight[5] = 1.9

        count = initialize_symbol_embeddings(
            model, FakeAnchorTokenizer(), verbose=False
        )
        self.assertEqual(count, 2)

        with torch.no_grad():
            # <ans> and → are genuinely added; ans is an existing base token.
            self.assertTrue(torch.allclose(model.get_input_embeddings().weight[10], torch.full((8,), 0.2)))
            self.assertTrue(torch.allclose(model.get_input_embeddings().weight[11], torch.full((8,), 0.3)))
            self.assertTrue(torch.allclose(model.get_input_embeddings().weight[5], torch.full((8,), 0.9)))
            self.assertTrue(torch.allclose(model.get_output_embeddings().weight[10], torch.full((8,), 1.2)))
            self.assertTrue(torch.allclose(model.get_output_embeddings().weight[11], torch.full((8,), 1.3)))
            self.assertTrue(torch.allclose(model.get_output_embeddings().weight[5], torch.full((8,), 1.9)))

    def test_resized_embedding_and_lm_head_rows_are_saved_at_real_geometry(self):
        # Real Qwen2.5/DeepSeek geometry: base model vocab 151,936, extended
        # tokenizer vocab 151,712.  This is a shrink, not the old 100 -> 120
        # expansion that could not catch accidental reallocation.
        base_vocab = 151936
        extended_vocab = 151712
        start = 151665
        cfg = Qwen2Config(
            vocab_size=base_vocab,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
        model = Qwen2ForCausalLM(cfg)
        model.resize_token_embeddings(extended_vocab)
        with torch.no_grad():
            model.get_input_embeddings().weight[start:] = 0.125
            model.lm_head.weight[start:] = 0.375

        peft_model = attach_lora_adapter(model)
        with tempfile.TemporaryDirectory() as tmp:
            peft_model.save_pretrained(tmp)
            adapter_config = json.loads(
                (Path(tmp) / "adapter_config.json").read_text(encoding="utf-8")
            )
            self.assertIn("embed_tokens", adapter_config["modules_to_save"])
            self.assertIn("lm_head", adapter_config["modules_to_save"])

            base = Qwen2ForCausalLM(cfg)
            base.resize_token_embeddings(extended_vocab)
            restored = PeftModel.from_pretrained(base, tmp)

            emb = restored.get_input_embeddings().weight[start:]
            head = restored.get_output_embeddings().weight[start:]
            self.assertTrue(torch.allclose(emb, torch.full_like(emb, 0.125)))
            self.assertTrue(torch.allclose(head, torch.full_like(head, 0.375)))


if __name__ == "__main__":
    unittest.main()
