"""Regression checks for the r4 LF-enforced canonical artifacts."""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CANONICAL_HASHES = {
    "data/concept_math_contrastive_v2_1.jsonl": (
        "724581AD5F3DEE4A1A4D2C321C167E13A016F64E023492AF6460D2BF04A3B7A7"
    ),
    "data/research_sft/train.jsonl": (
        "BB50D9CBD56B0D2B515D7B922DF10578F587AA60358B0FF87A11821C4D3D94DC"
    ),
    "data/research_sft/val.jsonl": (
        "3FCFCC4141522062087D0F35FDF2C54226D454A1A71FE59BEA26F42636FDEDCD"
    ),
    "data/held_out_eval_200_v2.jsonl": (
        "F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707"
    ),
    "data/held_out_eval_200_clean.jsonl": (
        "575B247066B65E5B8514990280C534BAA9448014E100ADCFE67D385E43B9AA8E"
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


class TestCanonicalProvenance(unittest.TestCase):
    def test_canonical_data_is_lf_and_has_the_declared_hash(self):
        for name, expected in CANONICAL_HASHES.items():
            with self.subTest(path=name):
                raw = (ROOT / name).read_bytes()
                self.assertNotIn(b"\r", raw, f"{name} contains a carriage return")
                self.assertEqual(sha256_bytes(raw), expected)

    def test_sft_manifest_matches_the_actual_lf_files(self):
        manifest = json.loads(
            (ROOT / "data" / "research_sft" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["line_endings"], "lf")
        self.assertEqual(manifest["source"], "data/concept_math_contrastive_v2_1.jsonl")
        self.assertEqual(
            manifest["source_sha256"], CANONICAL_HASHES[manifest["source"]]
        )
        self.assertEqual(
            manifest["train_sha256"],
            CANONICAL_HASHES["data/research_sft/train.jsonl"],
        )
        self.assertEqual(
            manifest["val_sha256"],
            CANONICAL_HASHES["data/research_sft/val.jsonl"],
        )
        self.assertEqual((manifest["train"], manifest["val"]), (4140, 460))
        self.assertEqual(
            manifest["target_kinds"], {"reasoning": 4200, "bypass": 400}
        )

    def test_benchmark_manifest_matches_the_actual_lf_files(self):
        manifest = json.loads(
            (ROOT / "data" / "held_out_eval_200_v2_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["line_endings"], "lf")
        self.assertEqual(
            manifest["input_sha256"],
            CANONICAL_HASHES["data/held_out_eval_200_clean.jsonl"],
        )
        self.assertEqual(
            manifest["output_sha256"],
            CANONICAL_HASHES["data/held_out_eval_200_v2.jsonl"],
        )
        self.assertEqual(manifest["n_rows"], 200)
        self.assertEqual(manifest["n_unique_prompts"], 173)

    def test_evaluator_hash_constants_match_the_lf_files(self):
        from eval.robust_eval import DATASET_SHA256, LEGACY_DATASET_SHA256

        self.assertEqual(
            DATASET_SHA256,
            CANONICAL_HASHES["data/held_out_eval_200_v2.jsonl"],
        )
        self.assertEqual(
            LEGACY_DATASET_SHA256,
            CANONICAL_HASHES["data/held_out_eval_200_clean.jsonl"],
        )


if __name__ == "__main__":
    unittest.main()
