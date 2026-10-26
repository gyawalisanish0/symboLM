"""Model-free tests for the fail-closed evaluation contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.run_contract import (
    EvalContractError,
    adapter_artifact_sha256,
    expected_cell_keys,
    load_benchmark_rows,
    select_stratified_subset,
    sha256_file,
    sha256_text,
    validate_adapter_config,
    validate_resume_manifest,
    verify_completed_cells,
)
from eval.robust_eval import build_prompt
from eval.strict_scorer import score_declared_answer
from symbolic.prompts import (
    QWEN_ASSISTANT_TOKEN,
    QWEN_USER_TOKEN,
    REGISTER_SYSTEM_PROMPT,
    build_chat_prompt,
    extract_target_answer,
)

TRACE = (
    "<reg:plan>concept: percentage</reg:plan>\n"
    "<reg:deduce>123*1.5=184.5 | 184.5-15=169.5 ∴ ans=169.5</reg:deduce>\n"
    "<reg:verify>verify(169.5>0)✓</reg:verify>\n"
    "</think>\n"
    "<ans>169.5</ans>"
)


class _FakeQwenTokenizer:
    """Minimal tokenizer surface needed by symbolic.prompts."""

    bos_token = "<bos>"
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = None

    def __init__(self):
        self._ids = {
            QWEN_USER_TOKEN: 10,
            QWEN_ASSISTANT_TOKEN: 11,
        }

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "CHAT:" + messages[-1]["content"]


def _row(problem_id: str, category: str, prompt: str = "p") -> dict:
    return {
        "id": problem_id,
        "prompt": f"{prompt}-{problem_id}",
        "ground_truth": "1",
        "category": category,
    }


class TestRunContract(unittest.TestCase):
    def test_adapter_hash_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adapter_model.safetensors").write_bytes(b"adapter-bytes")
            expected = sha256_file(root / "adapter_model.safetensors")
            self.assertEqual(adapter_artifact_sha256(root), expected)
            self.assertEqual(len(expected), 64)

            (root / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "base/model"}),
                encoding="utf-8",
            )
            self.assertIsNotNone(
                validate_adapter_config(root, "base/model", "rev")
            )
            with self.assertRaises(EvalContractError):
                validate_adapter_config(root, "other/model", "rev")

    def test_adapter_hash_rejects_missing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(EvalContractError):
                adapter_artifact_sha256(Path(tmp))

    def test_benchmark_loader_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.jsonl"
            path.write_text('{"id": "1", "prompt": "p", "ground_truth": "1"}\n', encoding="utf-8")
            with self.assertRaises(EvalContractError):
                load_benchmark_rows(path)

            path.write_text('not-json\n', encoding="utf-8")
            with self.assertRaises(EvalContractError):
                load_benchmark_rows(path)

            path.write_text(
                json.dumps(_row("1", "cat")) + "\n" + json.dumps(_row("1", "cat")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(EvalContractError):
                load_benchmark_rows(path)

    def test_benchmark_loader_hash_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.jsonl"
            rows = [_row(str(i), "cat") for i in range(3)]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            expected = sha256_file(path)
            loaded = load_benchmark_rows(path, expected_sha256=expected, expected_rows=3)
            self.assertEqual(len(loaded), 3)
            with self.assertRaises(EvalContractError):
                load_benchmark_rows(path, expected_sha256="0" * 64)

    def test_subset_is_deterministic_and_stratified(self):
        rows = []
        for category in ("a", "b", "c", "d", "e"):
            for index in range(20):
                rows.append(_row(f"{category}{index:02d}", category))

        first, info = select_stratified_subset(rows, 20, "20261025")
        second, _ = select_stratified_subset(rows, 20, "20261025")
        self.assertEqual(
            [row["id"] for row in first],
            [row["id"] for row in second],
        )
        self.assertEqual(len(first), 20)
        self.assertEqual(len({row["id"] for row in first}), 20)
        self.assertEqual(info["n"], 20)
        self.assertEqual(set(info["quotas"]), {"a", "b", "c", "d", "e"})
        self.assertTrue(all(count >= 1 for count in info["quotas"].values()))

        other, _ = select_stratified_subset(rows, 20, "different-seed")
        self.assertNotEqual(
            [row["id"] for row in first],
            [row["id"] for row in other],
        )

    def test_subset_full_and_invalid(self):
        rows = [_row(str(i), "cat") for i in range(5)]
        selected, info = select_stratified_subset(rows, 5, 42)
        self.assertEqual([row["id"] for row in selected], [row["id"] for row in rows])
        self.assertEqual(info["method"], "all_rows")
        with self.assertRaises(EvalContractError):
            select_stratified_subset(rows, 6, 42)
        with self.assertRaises(EvalContractError):
            select_stratified_subset(rows, 0, 42)

    def test_completion_contract(self):
        expected = expected_cell_keys(
            [_row("1", "cat"), _row("2", "cat")], ["A", "B"], [256, 512]
        )
        self.assertEqual(len(expected), 8)
        verify_completed_cells(expected, set(expected))
        with self.assertRaises(EvalContractError):
            verify_completed_cells(expected, set(list(expected)[:-1]))
        with self.assertRaises(EvalContractError):
            verify_completed_cells(expected, set(expected) | {"extra|A|256"})

    def test_resume_manifest_rejects_contract_changes(self):
        manifest = {
            "dataset_sha256": "A" * 64,
            "adapter_sha256": "B" * 64,
            "base_model": "base/model",
            "base_revision": "rev",
            "run_kind": "diagnostic",
            "truncation_rule": "hit_budget_without_eos_is_fail",
            "strict_scoring": True,
            "subset": {
                "method": "proportional_stratified_sha256",
                "seed": "42",
                "n": 2,
                "selected_ids": ["1", "2"],
            },
            "arms": ["C"],
            "budgets": [256],
            "contracts": {
                "evaluator_sha256": "C" * 64,
                "run_contract_sha256": "D" * 64,
                "strict_scorer_sha256": "E" * 64,
                "prompts_sha256": "F" * 64,
                "ground_truth_overrides_sha256": None,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            manifest_path = out_dir / "EVAL_RUN_MANIFEST.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (out_dir / "robust_eval_results.jsonl").write_text("", encoding="utf-8")
            validate_resume_manifest(out_dir, manifest)

            orphan = out_dir / "orphan"
            orphan.mkdir()
            (orphan / "robust_eval_results.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(EvalContractError):
                validate_resume_manifest(orphan, manifest)

            changed_adapter = json.loads(json.dumps(manifest))
            changed_adapter["adapter_sha256"] = "0" * 64
            with self.assertRaises(EvalContractError):
                validate_resume_manifest(out_dir, changed_adapter)

            changed_subset = json.loads(json.dumps(manifest))
            changed_subset["subset"]["selected_ids"] = ["1", "3"]
            with self.assertRaises(EvalContractError):
                validate_resume_manifest(out_dir, changed_subset)

            changed_scorer = json.loads(json.dumps(manifest))
            changed_scorer["contracts"]["strict_scorer_sha256"] = "0" * 64
            with self.assertRaises(EvalContractError):
                validate_resume_manifest(out_dir, changed_scorer)

            no_contracts = json.loads(json.dumps(manifest))
            no_contracts["contracts"] = {}
            with self.assertRaises(EvalContractError):
                validate_resume_manifest(out_dir, no_contracts)

            changed_arms = json.loads(json.dumps(manifest))
            changed_arms["arms"] = ["C", "D"]
            with self.assertRaises(EvalContractError):
                validate_resume_manifest(out_dir, changed_arms)
            validate_resume_manifest(out_dir, changed_arms, replace_arms=["D"])


class TestPromptScorerParity(unittest.TestCase):
    def test_arm_c_prompt_matches_training_builder(self):
        tokenizer = _FakeQwenTokenizer()
        item = {"id": "x", "prompt": "What is 2+2?", "category": "arithmetic_word_problem"}
        prompt, is_atomic = build_prompt(item, "C", tokenizer)
        expected = build_chat_prompt(
            tokenizer, REGISTER_SYSTEM_PROMPT, item["prompt"], open_think=True
        )
        self.assertEqual(prompt, expected)
        self.assertFalse(is_atomic)
        self.assertIn(item["prompt"], prompt)
        self.assertNotIn("Problem: ", prompt)
        self.assertTrue(prompt.endswith("<think>\n"))

    def test_arm_c_atomic_prompt_stays_register_prompt(self):
        tokenizer = _FakeQwenTokenizer()
        item = {
            "id": "x",
            "prompt": "What is the boiling point of water?",
            "category": "atomic_scientific_fact",
        }
        prompt, is_atomic = build_prompt(item, "C", tokenizer)
        self.assertTrue(is_atomic)
        self.assertIn(REGISTER_SYSTEM_PROMPT, prompt)
        self.assertIn(item["prompt"], prompt)

    def test_target_extraction_matches_strict_scorer(self):
        target_answer = extract_target_answer(TRACE)
        self.assertEqual(target_answer, "169.5")
        score = score_declared_answer(
            TRACE,
            arm="C",
            ground_truth=target_answer,
            tokens_emitted=70,
            budget=256,
            eos_emitted=True,
            expected_kind="reasoning",
        )
        self.assertEqual(score["answer"], target_answer)
        self.assertTrue(score["strict_correct"])
        self.assertTrue(score["format_valid"])

    def test_atomic_target_parity(self):
        target = "<reg:bypass>Heisenberg</reg:bypass>"
        target_answer = extract_target_answer(target)
        score = score_declared_answer(
            target,
            arm="C",
            ground_truth=target_answer,
            tokens_emitted=8,
            budget=256,
            eos_emitted=True,
            expected_kind="bypass",
        )
        self.assertEqual(score["answer"], target_answer)
        self.assertTrue(score["strict_correct"])
        self.assertTrue(score["format_valid"])

    def test_prompt_hash_is_deterministic(self):
        tokenizer = _FakeQwenTokenizer()
        item = {"id": "x", "prompt": "p", "category": "arithmetic_word_problem"}
        prompt_a, _ = build_prompt(item, "C", tokenizer)
        prompt_b, _ = build_prompt(item, "C", tokenizer)
        self.assertEqual(sha256_text(prompt_a), sha256_text(prompt_b))


if __name__ == "__main__":
    unittest.main()
