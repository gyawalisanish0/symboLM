"""
Tests for strict declared-answer scoring and legacy extraction.
"""

import unittest

from eval.robust_eval import extract_honest_answer, clean_math_string, extract_boxed
from eval.strict_scorer import (
    score_declared_answer,
    validate_register_trace,
    verify_answer,
)

TRACE = (
    "<reg:plan>concept: percentage</reg:plan>\n"
    "<reg:deduce>123*1.5=184.5 | 184.5-15=169.5 ∴ ans=169.5</reg:deduce>\n"
    "<reg:verify>verify(169.5>0)</reg:verify>\n"
    "</think>\n"
    "<ans>169.5</ans>"
)


class TestStrictScorer(unittest.TestCase):
    def test_declared_ans_is_accepted(self):
        ans, method = extract_honest_answer(
            TRACE, arm="C", tokens_emitted=70, max_tokens=256, eos_emitted=False
        )
        self.assertEqual(ans, "169.5")
        self.assertEqual(method, "closed_ans_tag")

    def test_declared_bypass_is_accepted(self):
        text = "<reg:bypass>Heisenberg</reg:bypass>"
        ans, method = extract_honest_answer(
            text, arm="C", tokens_emitted=5, max_tokens=256, eos_emitted=False
        )
        self.assertEqual(ans, "Heisenberg")
        self.assertEqual(method, "closed_bypass_tag")

    def test_truncated_is_fail_even_with_ans(self):
        result = score_declared_answer(
            TRACE, arm="C", ground_truth="169.5",
            tokens_emitted=256, budget=256, eos_emitted=False,
        )
        self.assertTrue(result["truncated"])
        self.assertFalse(result["is_correct"])
        self.assertIsNone(result["answer"])

    def test_eos_at_budget_is_not_truncated(self):
        result = score_declared_answer(
            TRACE, arm="C", ground_truth="169.5",
            tokens_emitted=256, budget=256, eos_emitted=True,
        )
        self.assertFalse(result["truncated"])
        self.assertTrue(result["is_correct"])

    def test_deduce_equality_fallback_is_rejected_in_strict_mode(self):
        text = (
            "<reg:plan>concept: markdown</reg:plan>\n"
            "<reg:deduce>selling_price=123*(1+50%)-15=184-15=169 zbollah169</reg:deduce>\n"
            "<reg:verify>verify(169>0)✓</reg:verify>"
        )
        ans, method = extract_honest_answer(
            text, arm="C", tokens_emitted=80, max_tokens=256, eos_emitted=False
        )
        self.assertIsNone(ans)
        self.assertEqual(method, "no_declared_answer")

    def test_legacy_fallback_is_explicit_only(self):
        text = "<reg:plan>invariants</reg:plan>\n<reg:deduce>step 1 | ∴ ans = 169</reg:deduce>"
        ans, method = extract_honest_answer(
            text, arm="C", tokens_emitted=45, max_tokens=256, strict=False
        )
        self.assertEqual(ans, "169")
        self.assertEqual(method, "symbolic_therefore")

    def test_boxed_is_not_valid_for_arm_c(self):
        text = "The answer is \\boxed{169.5}."
        ans, method = extract_honest_answer(
            text, arm="C", tokens_emitted=20, max_tokens=256, eos_emitted=False
        )
        self.assertIsNone(ans)
        self.assertEqual(method, "no_declared_answer")

    def test_boxed_is_valid_for_baseline_arm(self):
        text = "The answer is \\boxed{295}."
        ans, method = extract_honest_answer(
            text, arm="A", tokens_emitted=20, max_tokens=256, eos_emitted=False
        )
        self.assertEqual(ans, "295")
        self.assertEqual(method, "latex_boxed")

    def test_boxed_parser_nested_braces(self):
        self.assertEqual(extract_boxed(r"\boxed{\frac{1}{2}}"), r"\frac{1}{2}")

    def test_math_cleaner(self):
        self.assertEqual(clean_math_string("$\\mathbf{169.5}$"), "169.5")

    def test_ground_truth_override_value(self):
        ok, method = verify_answer("169.5", "169.5")
        self.assertTrue(ok)
        bad, _ = verify_answer("169", "169.5")
        self.assertFalse(bad)

    def test_format_validator_accepts_canonical_trace(self):
        ok, reason = validate_register_trace(TRACE, arm="C")
        self.assertTrue(ok, reason)

    def test_format_validator_rejects_missing_think_close(self):
        text = TRACE.replace("</think>\n", "")
        ok, reason = validate_register_trace(text, arm="C")
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_think_close")

    def test_format_validator_rejects_loop_marker(self):
        text = TRACE.replace("<ans>169.5</ans>", "<ans>169.5</ans> bands:")
        ok, reason = validate_register_trace(text, arm="C")
        self.assertFalse(ok)
        self.assertIn("loop_marker", reason)

    def test_format_validator_accepts_bypass(self):
        ok, reason = validate_register_trace("<reg:bypass>6</reg:bypass>", arm="C")
        self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()
