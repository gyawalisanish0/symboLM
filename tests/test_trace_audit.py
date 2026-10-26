"""Tests for the r7 trace-audit and verification-repair contracts."""

from __future__ import annotations

import unittest

from scripts.audit_training_traces import (
    audit_row,
    classify_verify,
    normalize_expression,
    repair_verify_from_deduce,
    summarize,
)

TOTIENT_TARGET = (
    "\u003cthink\u003e\n"
    "<reg:plan>concept: modular_totient | base: 7 | modulus: 17 | "
    "theorem: φ(17)=16</reg:plan>\n"
    "<reg:deduce>7^2027 mod 17 | φ(17)=16 | 2027=16·126+11→"
    "7^2027≡7^11 mod 17 | 7^11 mod 17=14 ∴ ans=14</reg:deduce>\n"
    "<reg:verify>verify(14<17 ∧ 14≥0)✓</reg:verify>\n"
    "</think>\n"
    "<ans>14</ans>"
)


def _row(target: str, answer: str, kind: str = "reasoning") -> dict:
    return {
        "id": "test_row",
        "prompt": "Compute the remainder when 7^2027 is divided by 17.",
        "full_target": target,
        "ground_truth": answer,
        "target_kind": kind,
        "split": "train",
        "source_index": 1,
    }


class TestVerificationParser(unittest.TestCase):
    def test_unicode_normalization(self):
        value = normalize_expression("verify(14<17 ∧ 14≥0)✓")
        self.assertIn("14<17", value)
        self.assertIn("and", value)
        self.assertIn("14 >= 0", value)

    def test_valid_check(self):
        result = classify_verify("verify(60>0 ∧ 60==P(5,3))✓", "60")
        self.assertEqual(result["status"], "valid")
        self.assertTrue(result["independent_arithmetic"])

    def test_mod_check_is_valid(self):
        result = classify_verify(
            "verify(91 mod 9=1 ∧ 91 mod 11=3)✓", "91"
        )
        self.assertEqual(result["status"], "valid")

    def test_weak_check(self):
        result = classify_verify("verify(60>0)✓", "60")
        self.assertEqual(result["status"], "weak")

    def test_tautology(self):
        result = classify_verify("verify(ans == 60)✓", "60")
        self.assertEqual(result["status"], "tautology")

    def test_alias_tautology(self):
        result = classify_verify("verify(det==5)✓", "5")
        self.assertEqual(result["status"], "tautology")

    def test_false_check(self):
        result = classify_verify("verify(ans == 59)✓", "60")
        self.assertEqual(result["status"], "false")

    def test_unrelated_check(self):
        result = classify_verify("verify(2+2==4)✓", "60")
        self.assertEqual(result["status"], "unrelated")

    def test_repair_modular_check(self):
        deduce = (
            "7^2027 mod 17 | φ(17)=16 | 2027=16·126+11→"
            "7^2027≡7^11 mod 17 | 7^11 mod 17=14"
        )
        repaired = repair_verify_from_deduce(deduce, "14")
        self.assertIsNotNone(repaired)
        self.assertIn("7**11 % 17 == 14", repaired[0])

    def test_repair_superscript_check(self):
        deduce = "x²+1/x²=u²-2=8²-2=62"
        repaired = repair_verify_from_deduce(deduce, "62")
        self.assertIsNotNone(repaired)
        self.assertIn("8**2-2 == 62", repaired[0])

    def test_no_repair_for_variable_chain(self):
        deduce = "t_total=1+t_series=3.86"
        self.assertIsNone(repair_verify_from_deduce(deduce, "3.86"))


class TestTraceAudit(unittest.TestCase):
    def test_weak_verify_is_repaired_and_verified(self):
        coverage, row, detail = audit_row(
            _row(TOTIENT_TARGET, "14"), "train", "source-hash"
        )
        self.assertEqual(coverage, "verified_reasoning")
        self.assertEqual(detail["verification_status"], "repaired")
        self.assertIn("7**11 % 17 == 14", row["full_target"])
        self.assertTrue(row["audit"]["verification_valid"])
        self.assertTrue(row["audit"]["answer_checked"])

    def test_bypass_contract_is_separate(self):
        target = "<reg:bypass>Newton</reg:bypass>"
        coverage, row, detail = audit_row(
            _row(target, "Newton", kind="bypass"), "train", "source-hash"
        )
        self.assertEqual(coverage, "verified_bypass_contract")
        self.assertEqual(detail["verification_status"], "not_applicable")
        self.assertFalse(row["audit"]["verification_valid"])
        self.assertTrue(row["audit"]["bypass_contract_valid"])

    def test_wrong_answer_is_rejected(self):
        coverage, row, detail = audit_row(
            _row(TOTIENT_TARGET, "15"), "train", "source-hash"
        )
        self.assertEqual(coverage, "rejected")
        self.assertIn("answer_mismatch", detail["reasons"])

    def test_arithmetic_mismatch_is_rejected(self):
        target = (
            "\u003cthink\u003e\n"
            "<reg:plan>concept: successive_discounts | original_price: 80 | "
            "discounts: [20%, 10%]</reg:plan>\n"
            "<reg:deduce>p₁=80·(1-20/100)=64 | "
            "p₂=64·(1-10/100)=57 ∴ ans=57</reg:deduce>\n"
            "<reg:verify>verify(ans == 57)✓</reg:verify>\n"
            "</think>\n"
            "<ans>57</ans>"
        )
        coverage, row_out, detail = audit_row(
            _row(target, "57"), "train", "source-hash"
        )
        self.assertEqual(row_out["ground_truth"], "57")
        self.assertEqual(coverage, "rejected")
        self.assertIn("arithmetic_mismatch", detail["reasons"])

    def test_unrepairable_tautology_is_removed(self):
        target = (
            "<think>\n"
            "<reg:plan>concept: rate | W_A: 1/6 | W_B: 1/8</reg:plan>\n"
            "<reg:deduce>W_A=1/6 | W_B=1/8 | "
            "t_total=1+t_series=3.86 ∴ ans=3.86</reg:deduce>\n"
            "<reg:verify>verify(ans == 3.86)✓</reg:verify>\n"
            "</think>\n"
            "<ans>3.86</ans>"
        )
        coverage, row, detail = audit_row(
            _row(target, "3.86"), "train", "source-hash"
        )
        self.assertEqual(coverage, "answer_checked_only")
        self.assertEqual(detail["verification_status"], "removed")
        self.assertNotIn("<reg:verify>", row["full_target"])
        self.assertFalse(row["audit"]["verification_valid"])

    def test_summary_counts_are_consistent(self):
        records = []
        for coverage, target, answer, kind in (
            ("verified_reasoning", TOTIENT_TARGET, "14", "reasoning"),
            (
                "verified_bypass_contract",
                "<reg:bypass>Newton</reg:bypass>",
                "Newton",
                "bypass",
            ),
        ):
            actual_coverage, row, detail = audit_row(
                _row(target, answer, kind=kind), "train", "source-hash"
            )
            self.assertEqual(actual_coverage, coverage)
            records.append(
                {
                    "split": "train",
                    "coverage": actual_coverage,
                    "row": row,
                    "detail": detail,
                }
            )
        summary = summarize(records)
        self.assertEqual(summary["audited"], 2)
        self.assertEqual(summary["verified_reasoning"], 1)
        self.assertEqual(summary["verified_bypass_contract"], 1)
        self.assertEqual(summary["retained_total"], 2)
        self.assertEqual(summary["rejected"], 0)


if __name__ == "__main__":
    unittest.main()
