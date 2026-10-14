"""
Unit Tests for Symbo State Machine & Counterfactual Branching
==============================================================
Author & Principal Systems Architect: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import unittest
from symbolm.state_machine import SymboStateMachine, EngineState


class TestStateMachine(unittest.TestCase):
    def setUp(self):
        self.sm = SymboStateMachine()

    def test_bypass_flow(self):
        self.sm.route_to_bypass("13")
        self.assertEqual(self.sm.current_state, EngineState.ROUTED_BYPASS)
        self.assertEqual(self.sm.answer_text, "13")

    def test_valid_deduction_flow(self):
        self.sm.enter_plan("concept: modular_reduction | modulus: 5")
        self.assertEqual(self.sm.current_state, EngineState.IN_PLAN)
        
        valid_deduce = "x = 4\ny = 2 * x + 1\ny = 9\n∴ ans = 9"
        ok = self.sm.process_deduction(valid_deduce)
        self.assertTrue(ok)
        self.assertEqual(self.sm.current_state, EngineState.IN_VERIFY)
        self.assertEqual(self.sm.answer_text, "9")

    def test_invalid_deduction_rollback(self):
        self.sm.enter_plan("concept: algebra")
        invalid_deduce = "x = 4\ny = 2 * x + 1\ny = 15\n∴ ans = 15"  # Slip: 2*4+1 = 9, not 15
        ok = self.sm.process_deduction(invalid_deduce)
        self.assertFalse(ok)
        self.assertEqual(self.sm.current_state, EngineState.COUNTERFACTUAL_ROLLBACK)
        self.assertEqual(self.sm.rollback_count, 1)
        self.assertIn("[TINY_TERMINAL_ERR]", self.sm.error_context)
        
        # Test compact 2-line recovery prompt
        rec_prompt = self.sm.build_counterfactual_prompt("Find y if x=4 and y=2x+1")
        self.assertIn("[TINY_TERMINAL_ERR]", rec_prompt)
        self.assertIn("<reg:plan>concept: algebra</reg:plan>", rec_prompt)
        self.assertIn("<reg:deduce>", rec_prompt)


if __name__ == "__main__":
    unittest.main()
