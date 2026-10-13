"""
Unit Tests for SymboLM Neuro-Symbolic Evaluator & Verifier
===========================================================
Author: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import unittest
from symbolm.symbolic.evaluator import (
    inv_mod,
    solve_crt,
    legendre_valuation,
    evaluate_unit_step,
    verify_grounded_constraint,
)


class TestNeuroSymbolicEvaluator(unittest.TestCase):
    def test_modular_arithmetic_and_crt(self):
        # 1. Modular Inverse
        # 7 * 8 = 56 = 5 * 11 + 1 = 1 (mod 11)
        self.assertEqual(inv_mod(7, 11), 8)

        # 2. Chinese Remainder Theorem: x = 4 (mod 7) and x = 9 (mod 11)
        # 7k + 4 = 9 (mod 11) -> 7k = 5 (mod 11) -> k = (8 * 5) % 11 = 40 % 11 = 7
        k = (inv_mod(7, 11) * 5) % 11
        self.assertEqual(k, 7)
        x = 7 * k + 4
        self.assertEqual(x, 53)

        # 3. Direct CRT Solver
        crt_sol = solve_crt([(4, 7), (9, 11)])
        self.assertEqual(crt_sol, 53)

    def test_legendre_formula(self):
        # Highest power of 3 dividing 200!
        # [200/3]=66, [200/9]=22, [200/27]=7, [200/81]=2 -> Total = 97
        total, terms = legendre_valuation(200, 3)
        self.assertEqual(terms, [66, 22, 7, 2])
        self.assertEqual(total, 97)

    def test_unit_step_evaluation(self):
        env = {}
        # let(u = 6)
        val, env = evaluate_unit_step("let(u = 6)", env)
        self.assertEqual(env["u"], 6)

        # u^2 - 2
        val, env = evaluate_unit_step("u^2 - 2", env)
        self.assertEqual(val, 34)

    def test_grounded_vs_ceremonial_verification(self):
        # Grounded check passes for correct CRT answer 53
        ok, msg = verify_grounded_constraint("x % 7 == 4 and x % 11 == 9", {"x": 53})
        self.assertTrue(ok)
        self.assertEqual(msg, "verified")

        # Grounded check fails for incorrect CRT answer 25 (k=3)
        ok, msg = verify_grounded_constraint("x % 7 == 4 and x % 11 == 9", {"x": 25})
        self.assertFalse(ok)

        # Ceremonial tautology rejected
        ok, msg = verify_grounded_constraint("66 > 0", {"x": 66})
        self.assertFalse(ok)
        self.assertEqual(msg, "rejected_trivial_tautology")


if __name__ == "__main__":
    unittest.main()
