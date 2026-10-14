"""
Unit Tests for Built-in Tiny Terminal & AST Interceptor
======================================================
Author & Principal Systems Architect: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import unittest
from symbolm.tiny_terminal import TinyTerminal, StepResult, TerminalTraceAudit


class TestTinyTerminal(unittest.TestCase):
    def setUp(self):
        self.terminal = TinyTerminal()

    def test_arithmetic_equality(self):
        # 15 * 24 = 360
        res = self.terminal.verify_equality("15 * 24", "360")
        self.assertTrue(res.valid)
        self.assertEqual(res.computed_value, 360)

    def test_arithmetic_contradiction(self):
        # 15 * 24 = 350 (slip)
        res = self.terminal.verify_equality("15 * 24", "350")
        self.assertFalse(res.valid)
        self.assertIn("Contradiction", res.error)
        self.assertIn("hyp [15 * 24 = 360]", res.repaired_hypothesis)

    def test_audit_deduction_steps_valid(self):
        deduce_trace = (
            "<reg:deduce>\n"
            "k = 1\n"
            "n = 3 * k + 2\n"
            "n = 5\n"
            "∴ ans = 5\n"
            "</reg:deduce>"
        )
        audit = self.terminal.audit_deduction_steps(deduce_trace)
        self.assertTrue(audit.is_valid)
        self.assertEqual(audit.certified_answer, "5")
        self.assertLess(audit.execution_time_ms, 50.0)  # Must run in sub-50ms

    def test_audit_deduction_steps_with_contradiction(self):
        deduce_trace = (
            "<reg:deduce>\n"
            "k = 1\n"
            "n = 3 * k + 2\n"
            "n = 8\n"  # Contradiction: 3*1+2 = 5, not 8!
            "∴ ans = 8\n"
            "</reg:deduce>"
        )
        audit = self.terminal.audit_deduction_steps(deduce_trace)
        self.assertFalse(audit.is_valid)
        self.assertIn("[TINY_TERMINAL_ERR]", audit.compact_error_context)
        self.assertIn("Contradiction", audit.compact_error_context)


if __name__ == "__main__":
    unittest.main()
