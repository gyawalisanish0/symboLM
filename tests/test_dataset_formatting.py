"""
Integration tests for the manual SFT chat builder and label masking.

These tests use the local adapter tokenizer when available.  They are the
regression guard for the old literal-BOS/EOS and chat-template-stripping bugs.
"""

import json
import tempfile
import unittest
from pathlib import Path

from data.symbolic_dataset import SymbolicSFTDataset
from symbolic.prompts import REGISTER_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = ROOT / "checkpoints" / "symbolm_v2_adapter"


@unittest.skipUnless(
    (TOKENIZER_DIR / "tokenizer_config.json").exists(),
    "local adapter tokenizer is required for this integration test",
)
class TestDatasetFormatting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(TOKENIZER_DIR), trust_remote_code=True
        )

    def _make_dataset(self, rows):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "rows.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        ds = SymbolicSFTDataset(
            path, self.tokenizer, max_length=512, strict_format=True
        )
        # The dataset eagerly loads records, so the temporary file can be
        # removed immediately without keeping a ResourceWarning alive.
        tmp.cleanup()
        return ds

    def test_prompt_is_token_prefix_and_eos_is_real(self):
        rows = [
            {
                "id": "reason-1",
                "prompt": "What is 2+2?",
                "full_target": (
                    "<think>\n<reg:plan>arithmetic</reg:plan>\n"
                    "<reg:deduce>2+2=4 ∴ ans=4</reg:deduce>\n"
                    "<reg:verify>verify(4==4)</reg:verify>\n"
                    "</think>\n<ans>4</ans>"
                ),
                "system_prompt": REGISTER_SYSTEM_PROMPT,
            },
            {
                "id": "bypass-1",
                "prompt": "Who proposed the uncertainty principle?",
                "full_target": "<reg:bypass>Werner Heisenberg</reg:bypass>",
                "system_prompt": REGISTER_SYSTEM_PROMPT,
            },
        ]
        ds = self._make_dataset(rows)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.summary()["kinds"], {"reasoning": 1, "bypass": 1})

        for idx in range(len(ds)):
            item = ds[idx]
            ids = item["input_ids"].tolist()
            labels = item["labels"].tolist()
            prompt = ds.get_prompt(idx)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

            self.assertTrue(prompt.endswith("<think>\n"))
            self.assertEqual(ids[: len(prompt_ids)], prompt_ids)
            self.assertEqual(sum(1 for x in labels if x == -100), len(prompt_ids))
            self.assertEqual(ids[-1], self.tokenizer.eos_token_id)
            self.assertNotEqual(labels[-1], -100)

        self.assertIn("<ans>", ds.get_target_body(0))
        self.assertIn("</reg:bypass>", ds.get_target_body(1))

    def test_dialogue_and_contrastive_are_skipped(self):
        rows = [
            {
                "id": "dialogue-1",
                "prompt": "A colleague took credit for my idea.",
                "full_target": "<reg:intent>assert-boundaries</reg:intent>\nTalk to them.",
            },
            {
                "id": "contrastive-1",
                "prompt": "What is 1+1?",
                "full_target": (
                    "<think>\n<reg:plan>arithmetic</reg:plan>\n"
                    "<checkpoint:CP1>\n<reg:deduce>wrong</reg:deduce>\n"
                    "<reg:verify>bad</reg:verify>\n<backtrack:CP1>\n"
                    "<reg:deduce>1+1=2</reg:deduce>\n</think>\n<ans>2</ans>"
                ),
            },
            {
                "id": "canonical-1",
                "prompt": "What is 3+3?",
                "full_target": (
                    "<think>\n<reg:plan>arithmetic</reg:plan>\n"
                    "<reg:deduce>3+3=6</reg:deduce>\n</think>\n<ans>6</ans>"
                ),
            },
        ]
        ds = self._make_dataset(rows)
        self.assertEqual(len(ds), 1)
        skipped = ds.summary()["skipped"]
        self.assertEqual(skipped.get("dialogue_out_of_scope"), 1)
        self.assertEqual(skipped.get("contrastive_out_of_scope"), 1)
        self.assertEqual(ds.records[0]["id"], "canonical-1")


if __name__ == "__main__":
    unittest.main()
