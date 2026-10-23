"""Regression checks for the r6 Kaggle package and token-delta fix."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SFT = ROOT / "experiments" / "kaggle" / "sft_v3"


class TestKaggleSftPackageFix(unittest.TestCase):
    def test_generator_pins_r6_and_removes_torchao(self):
        source = (SFT / "make_notebook.py").read_text(encoding="utf-8")
        self.assertIn('GIT_TAG = "sft-v3-20261023-r6"', source)
        self.assertIn('FORBIDDEN_PACKAGES = ("torchao",)', source)
        self.assertIn('"pip", "uninstall", "-y", "-q", name', source)
        self.assertIn('if actual != "absent":', source)

    def test_generated_notebook_contains_the_package_fix(self):
        notebook = json.loads(
            (SFT / "symbolm_v3_sft.ipynb").read_text(encoding="utf-8")
        )
        text = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("sft-v3-20261023-r6", text)
        self.assertIn("FORBIDDEN_PACKAGES", text)
        self.assertIn('"pip", "uninstall", "-y", "-q", name', text)
        self.assertIn("torchao", text)
        self.assertIn("token", text.lower())

    def test_runner_gate_requires_torchao_absence(self):
        source = (SFT / "run_sft_kernel.py").read_text(encoding="utf-8")
        self.assertIn('EXPECTED_ABSENT_PACKAGES = ("torchao",)', source)
        self.assertIn('package_versions.get(name) == "absent"', source)
        self.assertIn("terminal_marker_rows", source)

    def test_runner_gate_requires_token_delta_evidence(self):
        source = (SFT / "run_sft_kernel.py").read_text(encoding="utf-8")
        self.assertIn('token_delta.get("present") is True', source)
        self.assertIn('marker_check.get("distinct") is True', source)
        self.assertIn("EXPECTED_ANS_OPEN_ID", source)
        self.assertIn("for head_key in head_keys", source)
        self.assertIn('"original_module" not in key', source)

    def test_training_manifest_records_torchao(self):
        source = (ROOT / "training" / "sft_train.py").read_text(encoding="utf-8")
        self.assertIn('"torchao": package_version("torchao")', source)

        from training.sft_train import package_version

        self.assertEqual(
            package_version("definitely-not-a-real-distribution-symbolm"),
            "absent",
        )


if __name__ == "__main__":
    unittest.main()
