"""
Regression test for the extended-tokenizer adapter artifact.

The tokenizer adds <ans> and </ans>, which forces a resized embedding matrix and
LM head.  A plain LoRA state dict does not contain those rows, so evaluation
would reinitialize them randomly.  This test verifies that the training helper
persists the exact rows and that PeftModel restores them.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen2Config, Qwen2ForCausalLM

from symbolic.tokenizer_ext import SEMANTIC_ANCHORS
from training.sft_train import attach_lora_adapter


class TestAdapterArtifact(unittest.TestCase):
    def test_terminal_markers_have_semantic_anchors(self):
        self.assertIn("<ans>", SEMANTIC_ANCHORS)
        self.assertIn("</ans>", SEMANTIC_ANCHORS)

    def test_resized_embedding_and_lm_head_rows_are_saved(self):
        cfg = Qwen2Config(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
        model = Qwen2ForCausalLM(cfg)
        model.resize_token_embeddings(120)
        with torch.no_grad():
            model.get_input_embeddings().weight[100:] = 0.125
            model.lm_head.weight[100:] = 0.375

        peft_model = attach_lora_adapter(model)
        with tempfile.TemporaryDirectory() as tmp:
            peft_model.save_pretrained(tmp)
            adapter_config = json.loads(
                (Path(tmp) / "adapter_config.json").read_text(encoding="utf-8")
            )
            self.assertIn("embed_tokens", adapter_config["modules_to_save"])
            self.assertIn("lm_head", adapter_config["modules_to_save"])

            base = Qwen2ForCausalLM(cfg)
            base.resize_token_embeddings(120)
            restored = PeftModel.from_pretrained(base, tmp)

            emb = restored.get_input_embeddings().weight[100:]
            head = restored.get_output_embeddings().weight[100:]
            self.assertTrue(torch.allclose(emb, torch.full_like(emb, 0.125)))
            self.assertTrue(torch.allclose(head, torch.full_like(head, 0.375)))


if __name__ == "__main__":
    unittest.main()
