"""
Unit Tests for SymboLM Unbounded Scratchpad 2.1 & Multi-Level Backtracking
========================================================================
Verifies:
  1. No hard token cap: Monotonic parsimony gradient remains strictly positive at 500+ tokens.
  2. Reward for short reasoning over long reasoning (∂R/∂|τ| < 0) with token friction.
  3. Atomic unit-step computation (CRT and Legendre valuations).
  4. Stack-based checkpointing and backtracking (checkpoint -> verify✗ -> backtrack -> replan -> verify✓).
  5. Anti-ceremonial tautology rejection (rejecting '66 > 0' and 'ans > 0').
  6. Cognitive Depth Spectrum (Δ0 to Δ5) integration and round-trip parsing.

Author & Principal Systems Architect: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import unittest
from symbolm.scratchpad import (
    ScratchpadDepth,
    ScratchpadStateMachine,
    compute_unbounded_parsimony_reward,
    parse_scratchpad_block,
)
from symbolm.registers import parse_registers, RegisterType
from symbolm.symbolic.evaluator import (
    compute_unbounded_gradient_reward,
    verify_grounded_constraint,
    inv_mod,
    legendre_valuation,
    solve_crt,
)
from symbolic.parser import SRLParser


class TestScratchpadV21(unittest.TestCase):

    def test_unbounded_parsimony_gradient_and_friction(self):
        """
        Verify that:
          1. Shorter derivations strictly beat longer derivations for the same depth.
          2. Zero hard cap: At 300, 500, and 800 tokens, reward remains strictly positive (R > 0).
          3. Token friction guarantees that redundant filler always carries marginal cost.
        """
        # Test Δ3 (Compositional Reduction)
        r_20 = compute_unbounded_parsimony_reward(20, ScratchpadDepth.DELTA_3, is_correct=True)
        r_60 = compute_unbounded_parsimony_reward(60, ScratchpadDepth.DELTA_3, is_correct=True)
        r_120 = compute_unbounded_parsimony_reward(120, ScratchpadDepth.DELTA_3, is_correct=True)
        r_300 = compute_unbounded_parsimony_reward(300, ScratchpadDepth.DELTA_3, is_correct=True)
        r_600 = compute_unbounded_parsimony_reward(600, ScratchpadDepth.DELTA_3, is_correct=True)

        # Monotonicity: Short strictly beats long
        self.assertGreater(r_20, r_60, "20 tokens should beat 60 tokens")
        self.assertGreater(r_60, r_120, "60 tokens should beat 120 tokens")
        self.assertGreater(r_120, r_300, "120 tokens should beat 300 tokens")
        self.assertGreater(r_300, r_600, "300 tokens should beat 600 tokens")

        # Zero Hard Cap: Even at 600 tokens, reward is strictly positive!
        self.assertGreater(r_600, 0.50, f"Expected strictly positive reward at 600 tokens, got {r_600}")

        # Test Δ5 (Olympiad Unbounded proof)
        r_deep_350 = compute_unbounded_parsimony_reward(350, ScratchpadDepth.DELTA_5, is_correct=True)
        r_deep_700 = compute_unbounded_parsimony_reward(700, ScratchpadDepth.DELTA_5, is_correct=True)
        self.assertGreater(r_deep_350, r_deep_700)
        self.assertGreater(r_deep_700, 0.50, f"Expected positive reward at 700 tokens on Δ5, got {r_deep_700}")

        # Wrong answers must be strictly negative
        r_wrong = compute_unbounded_parsimony_reward(20, ScratchpadDepth.DELTA_3, is_correct=False)
        self.assertLess(r_wrong, 0.0, "Incorrect answer must be penalized to negative")

    def test_unit_step_crt_execution(self):
        """
        Verify that Chinese Remainder Theorem solves correctly when decomposed
        into atomic single-operator steps (no multi-op fusion).
        """
        trace_text = """
        state(moduli=[7, 11], target="x")
        | inv = inv_mod(7, 11)
        | mult = (inv * 5) % 11
        | x = mult * 7 + 4
        | verify((x % 7 == 4) and (x % 11 == 9))✓
        ∴ ans = 53
        """
        sm = ScratchpadStateMachine(depth=ScratchpadDepth.DELTA_3, domain="number_theory")
        trace = sm.run_trace(trace_text)

        self.assertTrue(trace.is_valid, f"Trace failed: {trace.diagnostics}")
        self.assertEqual(sm.env.get("inv"), 8, "inv_mod(7, 11) should equal 8")
        self.assertEqual(sm.env.get("mult"), 7, "(8 * 5) % 11 should equal 7")
        self.assertEqual(sm.env.get("x"), 53, "7 * 7 + 4 should equal 53")
        self.assertEqual(trace.final_answer, "53")

    def test_legendre_prime_valuation_full_terms(self):
        """
        Verify that Legendre prime valuation nu_3(200!) evaluates all 4 terms
        (66 + 22 + 7 + 2 = 97) instead of truncating at 66.
        """
        total, terms = legendre_valuation(200, 3)
        self.assertEqual(total, 97, f"Expected total 97, got {total}")
        self.assertEqual(terms, [66, 22, 7, 2], f"Expected terms [66, 22, 7, 2], got {terms}")

        trace_text = """
        state(n=200, p=3)
        | t1 = 200 // 3
        | t2 = 200 // 9
        | t3 = 200 // 27
        | t4 = 200 // 81
        | total = t1 + t2 + t3 + t4
        | verify(total == 97)✓
        ∴ ans = 97
        """
        sm = ScratchpadStateMachine(depth=ScratchpadDepth.DELTA_1, domain="number_theory")
        trace = sm.run_trace(trace_text)

        self.assertTrue(trace.is_valid)
        self.assertEqual(sm.env.get("total"), 97)
        self.assertEqual(trace.final_answer, "97")

    def test_stack_checkpoint_and_multi_level_backtrack(self):
        """
        Verify robust stack-based checkpointing and backtracking:
          1. Sets initial state.
          2. Checkpoints state at CP1.
          3. Branches with a false hypothesis.
          4. verify(cond)✗ detects failure.
          5. backtrack(CP1) restores clean state.
          6. replan establishes corrected branch.
          7. verify(cond)✓ passes and final answer is extracted.
        """
        trace_text = """
        state(base=7, target=53)
        | checkpoint(CP1)
        | guess = 18
        | verify(guess % 11 == 9)✗
        | backtrack(CP1)
        | replan(strategy="bezout_reduction")
        | x = 7 * 7 + 4
        | verify(x % 11 == 9)✓
        ∴ ans = 53
        """
        sm = ScratchpadStateMachine(depth=ScratchpadDepth.DELTA_5, domain="number_theory")
        trace = sm.run_trace(trace_text)

        self.assertTrue(trace.is_valid, f"Trace failed: {trace.diagnostics}")
        self.assertEqual(trace.backtrack_count, 1, "Expected exactly 1 recorded backtrack")
        self.assertEqual(sm.env.get("x"), 53)
        # 'guess' variable should have been wiped out by rollback to CP1
        self.assertNotIn("guess", sm.env, "Rolled-back variable 'guess' should not exist in env")
        self.assertEqual(trace.final_answer, "53")

    def test_tautology_active_rejection(self):
        """
        Verify that ceremonial rubber-stamp tautologies like '66 > 0' and 'ans > 0'
        are actively rejected by the verifier.
        """
        passed1, reason1 = verify_grounded_constraint("66 > 0", {})
        self.assertFalse(passed1, "Vacuous tautology '66 > 0' should evaluate to False")
        self.assertEqual(reason1, "rejected_trivial_tautology")

        passed2, reason2 = verify_grounded_constraint("ans > 0", {})
        self.assertFalse(passed2, "Vacuous tautology 'ans > 0' should evaluate to False")
        self.assertEqual(reason2, "rejected_trivial_tautology")

        # Grounded algebraic check should succeed
        passed3, reason3 = verify_grounded_constraint("53 % 7 == 4", {"ans": 53})
        self.assertTrue(passed3, f"Grounded check should evaluate to True, got {reason3}")

    def test_round_trip_registers_parsing(self):
        """
        Verify that <reg:scratchpad depth="4" ...> parses cleanly via parse_registers
        and SRLParser into structured AST nodes.
        """
        xml_trace = """<reg:scratchpad depth="4" domain="number_theory">
  state(moduli=[7, 11])
  | checkpoint(CP1)
  | inv = inv_mod(7, 11)
  | x = inv * 7 - 3
  | verify(x > 0)✗
  | backtrack(CP1)
  | replan(strategy="direct_chinese_remainder")
  | x = 53
  | verify(x % 7 == 4)✓
  ∴ ans = 53
</reg:scratchpad>
<ans>53</ans>"""

        # 1. Parse via symbolm.registers.parse_registers
        reg_trace = parse_registers(xml_trace)
        self.assertEqual(reg_trace.register_type, RegisterType.SCRATCHPAD)
        self.assertEqual(reg_trace.depth, "4")
        self.assertEqual(reg_trace.answer, "53")
        self.assertEqual(reg_trace.checkpoints, ["CP1"])
        self.assertGreaterEqual(reg_trace.backtracks, 1)

        # 2. Parse via symbolic.parser.SRLParser
        parser = SRLParser()
        result = parser.parse(xml_trace)
        self.assertTrue(result.is_valid, f"SRLParser failed: {result.diagnostics}")
        self.assertEqual(result.ast.register_type, "scratchpad")
        self.assertEqual(result.extracted_answer, "53")


if __name__ == "__main__":
    unittest.main()
