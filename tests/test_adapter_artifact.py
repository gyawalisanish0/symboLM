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
from training.token_delta import (
    assert_terminal_marker_rows_distinct,
    merge_token_deltas,
)


class FakeAnchorTokenizer:
    """Minimal tokenizer surface for the semantic-anchor initializer."""

    def __init__(self):
        self.added = {"<ans>": 10, "</ans>": 11, "→": 12}
        self.ids = {
            "<ans>": 10,
            "</ans>": 11,
            "→": 12,
            "answer": 1,
            "final": 2,
            " answer": 3,
            " end": 4,
            "ans": 5,
            "implies": 6,
            "correct": 7,
        }
        self.phrase_ids = {
            "final answer": [2, 3],
            "answer end": [1, 4],
        }
        self.unk_token_id = 0

    def get_added_vocab(self):
        return dict(self.added)

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, 0)

    def encode(self, text, add_special_tokens=False):
        return self.phrase_ids.get(text, [self.ids.get(text, 0)])


class TestAdapterArtifact(unittest.TestCase):
    def test_terminal_markers_have_distinct_multi_token_anchors(self):
        self.assertEqual(SEMANTIC_ANCHORS["<ans>"], "final answer")
        self.assertEqual(SEMANTIC_ANCHORS["</ans>"], "answer end")

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
            # Rows 1-4 are the anchor vocabulary; 5 is an existing base token.
            for row, value in {1: 0.1, 2: 0.2, 3: 0.5, 4: 1.0, 5: 0.9, 6: 0.3}.items():
                model.get_input_embeddings().weight[row] = value
            for row, value in {1: 1.0, 2: 2.0, 3: 4.0, 4: 8.0, 5: 1.9, 6: 1.3}.items():
                model.get_output_embeddings().weight[row] = value

        count = initialize_symbol_embeddings(
            model, FakeAnchorTokenizer(), verbose=False
        )
        self.assertEqual(count, 3)

        with torch.no_grad():
            # <ans>, </ans>, and → are genuinely added; ans is a base token.
            # <ans> averages "final" + " answer"; </ans> averages "answer" +
            # " end", so neither is an exact copy of the base answer row.
            self.assertTrue(torch.allclose(model.get_input_embeddings().weight[10], torch.full((8,), 0.35)))
            self.assertTrue(torch.allclose(model.get_input_embeddings().weight[11], torch.full((8,), 0.55)))
            self.assertTrue(torch.allclose(model.get_input_embeddings().weight[12], torch.full((8,), 0.3)))
            self.assertTrue(torch.allclose(model.get_input_embeddings().weight[5], torch.full((8,), 0.9)))
            self.assertTrue(torch.allclose(model.get_output_embeddings().weight[10], torch.full((8,), 3.0)))
            self.assertTrue(torch.allclose(model.get_output_embeddings().weight[11], torch.full((8,), 4.5)))
            self.assertTrue(torch.allclose(model.get_output_embeddings().weight[12], torch.full((8,), 1.3)))
            self.assertTrue(torch.allclose(model.get_output_embeddings().weight[5], torch.full((8,), 1.9)))
            self.assertFalse(torch.equal(model.get_output_embeddings().weight[10], model.get_output_embeddings().weight[1]))
            self.assertFalse(torch.equal(model.get_output_embeddings().weight[11], model.get_output_embeddings().weight[1]))
            self.assertFalse(torch.equal(model.get_output_embeddings().weight[10], model.get_output_embeddings().weight[11]))

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


def _tiny_qwen(vocab_size=32, hidden_size=8):
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return cfg, Qwen2ForCausalLM(cfg)


class FakeMarkerTokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<ans>": 20, "</ans>": 21, "answer": 1}[token]


class TestTokenDeltaAdapter(unittest.TestCase):
    def setUp(self):
        self.cfg, model = _tiny_qwen()
        with torch.no_grad():
            # Simulate the r5 failure: the added rows and the English answer
            # row are exactly identical before training.
            for row in (1, 20, 21):
                model.get_input_embeddings().weight[row] = 0.25
                model.get_output_embeddings().weight[row] = 1.5
        self.model = attach_lora_adapter(model, added_token_ids=[20, 21])
        self.delta = self.model.symbol_token_delta

    def test_zero_delta_preserves_the_frozen_row_tie(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([[5, 6]])).logits[0, -1]
        self.assertTrue(torch.allclose(logits[20], logits[1]))
        self.assertTrue(torch.allclose(logits[21], logits[1]))

    def test_deltas_break_ties_and_receive_gradients(self):
        self.model.eval()
        with torch.no_grad():
            before = self.model(input_ids=torch.tensor([[5, 6]])).logits[0, -1]
            self.delta.head_delta[0].fill_(0.5)
            self.delta.head_delta[1].fill_(-0.5)
            after = self.model(input_ids=torch.tensor([[5, 6]])).logits[0, -1]
        self.assertTrue(torch.allclose(before[20], before[1]))
        self.assertGreater((after[20] - after[1]).abs().item(), 1e-4)
        self.assertGreater((after[21] - after[1]).abs().item(), 1e-4)
        self.assertGreater((after[20] - after[21]).abs().item(), 1e-4)

        self.model.train()
        self.model.zero_grad(set_to_none=True)
        loss = self.model(
            input_ids=torch.tensor([[20, 5]]),
            labels=torch.tensor([[20, 21]]),
        ).loss
        loss.backward()
        self.assertIsNotNone(self.delta.embed_delta.grad)
        self.assertIsNotNone(self.delta.head_delta.grad)
        self.assertGreater(self.delta.embed_delta.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.delta.head_delta.grad.abs().sum().item(), 0.0)
        for parameter in self.model.get_output_embeddings().parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        lora_grads = [
            parameter.grad for name, parameter in self.model.named_parameters()
            if "lora_A" in name or "lora_B" in name
        ]
        self.assertTrue(any(grad is not None and grad.abs().sum().item() > 0 for grad in lora_grads))

    def test_merge_is_exact_and_survives_save_load(self):
        self.model.eval()
        with torch.no_grad():
            base_input = self.model.get_input_embeddings().weight.detach().clone()
            base_output = self.model.get_output_embeddings().weight.detach().clone()
            self.delta.embed_delta[0].fill_(0.125)
            self.delta.embed_delta[1].fill_(-0.25)
            self.delta.head_delta[0].fill_(0.375)
            self.delta.head_delta[1].fill_(-0.625)
            before = self.model(input_ids=torch.tensor([[5, 20, 21]])).logits.detach().clone()

        summary = merge_token_deltas(self.model)
        self.assertTrue(summary["present"])
        self.assertTrue(summary["merged"])
        self.assertEqual(len(self.delta._handles), 0)

        with torch.no_grad():
            after = self.model(input_ids=torch.tensor([[5, 20, 21]])).logits
            merged_input = self.model.get_input_embeddings().weight[20]
            merged_output = self.model.get_output_embeddings().weight[20]
            expected_input = base_input[20] + self.delta.embed_delta[0]
            expected_output = base_output[20] + self.delta.head_delta[0]
        self.assertTrue(torch.allclose(before, after, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(merged_input, expected_input))
        self.assertTrue(torch.allclose(merged_output, expected_output))

        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp)
            adapter_config = json.loads(
                (Path(tmp) / "adapter_config.json").read_text(encoding="utf-8")
            )
            self.assertIn("embed_tokens", adapter_config["modules_to_save"])
            self.assertIn("lm_head", adapter_config["modules_to_save"])

            base = Qwen2ForCausalLM(self.cfg)
            base.resize_token_embeddings(32)
            restored = PeftModel.from_pretrained(base, tmp)
            self.assertTrue(
                torch.allclose(
                    restored.get_input_embeddings().weight[20], merged_input
                )
            )
            self.assertTrue(
                torch.allclose(
                    restored.get_output_embeddings().weight[20], merged_output
                )
            )

    def test_terminal_marker_check_rejects_exact_ties(self):
        cfg, model = _tiny_qwen()
        with torch.no_grad():
            for row in (1, 20, 21):
                model.get_input_embeddings().weight[row] = 0.25
                model.get_output_embeddings().weight[row] = 1.5
        with self.assertRaises(RuntimeError):
            assert_terminal_marker_rows_distinct(model, FakeMarkerTokenizer())

        with torch.no_grad():
            model.get_output_embeddings().weight[20] = 1.6
            model.get_output_embeddings().weight[21] = 1.7
            model.get_input_embeddings().weight[20] = 0.3
            model.get_input_embeddings().weight[21] = 0.4
        result = assert_terminal_marker_rows_distinct(model, FakeMarkerTokenizer())
        self.assertTrue(result["distinct"])
        self.assertEqual(result["token_ids"]["<ans>"], 20)


if __name__ == "__main__":
    unittest.main()
