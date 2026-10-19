"""
Unit tests for robust_eval.py answer extraction and closure recognition.
Tests:
  - Exact <ans> and <reg:bypass> closures.
  - Boxed extraction.
  - Closed <reg:deduce> and <reg:verify> extraction.
  - Rejection of truncated incomplete derivations without closure.
"""

import unittest
from eval.robust_eval import extract_honest_answer, clean_math_string, extract_boxed


class TestRobustEvalExtractor(unittest.TestCase):
    def test_ans_tag_extraction(self):
        text = "<think>some thoughts</think>\n<ans>42</ans>"
        ans, method = extract_honest_answer(text, arm="C", tokens_emitted=50, max_tokens=256)
        self.assertEqual(ans, "42")
        self.assertEqual(method, "closed_ans_tag")

    def test_bypass_tag_extraction(self):
        text = "<reg:bypass>6</reg:bypass>"
        ans, method = extract_honest_answer(text, arm="B", tokens_emitted=3, max_tokens=256)
        self.assertEqual(ans, "6")
        self.assertEqual(method, "closed_bypass_tag")

    def test_boxed_extraction(self):
        text = "Therefore the result is \\boxed{295}."
        ans, method = extract_honest_answer(text, arm="A", tokens_emitted=80, max_tokens=256)
        self.assertEqual(ans, "295")
        self.assertEqual(method, "latex_boxed")

    def test_closed_deduce_therefore(self):
        text = "<reg:plan>invariants</reg:plan>\n<reg:deduce>step 1 | ∴ ans = 169</reg:deduce>"
        ans, method = extract_honest_answer(text, arm="C", tokens_emitted=45, max_tokens=256)
        self.assertEqual(ans, "169")
        self.assertEqual(method, "symbolic_therefore")

    def test_closed_deduce_equality_rhs(self):
        text = (
            "<reg:plan>concept: markdown</reg:plan>\n"
            "<reg:deduce>selling_price=123*(1+50%)-15=184-15=169 zbollah169</reg:deduce>\n"
            "<reg:verify>verify(169>0)✓</reg:verify>"
        )
        # Even if tokens_emitted hit max_tokens, closure was achieved!
        ans, method = extract_honest_answer(text, arm="C", tokens_emitted=256, max_tokens=256)
        self.assertEqual(ans, "169")
        self.assertIn(method, ("symbolic_therefore", "deduce_equality_rhs"))

    def test_truncated_unclosed_rejected(self):
        text = "<reg:plan>concept: markdown</reg:plan>\n<reg:deduce>selling_price=123*5"
        ans, method = extract_honest_answer(text, arm="C", tokens_emitted=256, max_tokens=256)
        self.assertIsNone(ans)
        self.assertEqual(method, "truncated_incomplete_derivation")

    def test_empty_text_rejected(self):
        ans, method = extract_honest_answer("", arm="C", tokens_emitted=0, max_tokens=256)
        self.assertIsNone(ans)
        self.assertEqual(method, "empty_text")


if __name__ == "__main__":
    unittest.main()
