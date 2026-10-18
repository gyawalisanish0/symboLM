import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.reward import (
    detect_ngram_repetition,
    detect_web_debris,
    hard_eos_and_economy_reward_func,
    detect_self_correction,
)


class TestStage4Reward(unittest.TestCase):
    def test_repetition_detection(self):
        looping_output = "<think>\n<reg:deduce>x=527</reg:deduce>\n</think>\n527 527 527 527 527"
        self.assertTrue(detect_ngram_repetition(looping_output))

        potassium_loop = "<reg:bypass>K</reg:bypass> potassium potassium potassium potassium"
        self.assertTrue(detect_ngram_repetition(potassium_loop))

        clean_output = "<think>\n<reg:deduce>x=527 ∴ ans=527</reg:deduce>\n</think>\n<ans>527</ans>"
        self.assertFalse(detect_ngram_repetition(clean_output))

    def test_web_debris_detection(self):
        debris = "some reasoning endphp IconModule tatto"
        self.assertTrue(detect_web_debris(debris))

        clean = "<reg:deduce>Area=r*s=6*36=216 ∴ ans=216</reg:deduce>"
        self.assertFalse(detect_web_debris(clean))

    def test_hard_eos_and_economy_reward(self):
        # 1. Clean, compact proof with immediate EOS -> High Positive Reward
        clean_comp = "<think>\n<reg:deduce>x=527 ∴ ans=527</reg:deduce>\n</think>\n<ans>527</ans>"
        rewards = hard_eos_and_economy_reward_func(
            prompts=["prompt"],
            completions=[clean_comp],
            answers=["527"],
            registers=["concept_math"]
        )
        self.assertGreater(rewards[0], 1.0)
        print(f"[OK] Clean Compact Proof Reward: {rewards[0]:.2f}")

        # 2. Looping answer -> Severe Repetition Penalty
        loop_comp = "<think>\n<reg:deduce>x=527</reg:deduce>\n</think>\n527 527 527 527 527"
        rewards_loop = hard_eos_and_economy_reward_func(
            prompts=["prompt"],
            completions=[loop_comp],
            answers=["527"],
            registers=["concept_math"]
        )
        self.assertLess(rewards_loop[0], -1.5)
        print(f"[OK] Looping Repetition Penalty: {rewards_loop[0]:.2f}")

        # 3. Web debris -> Severe Debris Penalty
        debris_comp = "<think>\n<reg:deduce>x=527 endphp</reg:deduce>\n</think>\n<ans>527</ans>"
        rewards_debris = hard_eos_and_economy_reward_func(
            prompts=["prompt"],
            completions=[debris_comp],
            answers=["527"],
            registers=["concept_math"]
        )
        self.assertLess(rewards_debris[0], -0.5)
        print(f"[OK] Debris Penalty: {rewards_debris[0]:.2f}")

    def test_detect_self_correction_penalizes_false_checkmark(self):
        # False assertion 15*24=350 claimed with ✓ must be penalized with -0.40
        false_rollout = "<reg:deduce>verify(15 * 24 == 350)✓ ∴ ans=350</reg:deduce><ans>350</ans>"
        score = detect_self_correction(false_rollout, is_correct=False)
        self.assertEqual(score, -0.40)

    def test_detect_self_correction_rewards_valid_verification(self):
        # Valid assertion 15*24=360 with ✓ and correct answer gets +0.40
        valid_rollout = "<reg:deduce>verify(15 * 24 == 360)✓ ∴ ans=360</reg:deduce><ans>360</ans>"
        score = detect_self_correction(valid_rollout, is_correct=True)
        self.assertEqual(score, 0.40)

    def test_detect_self_correction_ignores_vacuous_tautology(self):
        # Vacuous tautology like 66 > 0 without genuine invariant does not get verification bonus
        tautology_rollout = "<reg:deduce>verify(66 > 0)✓</reg:deduce>"
        score = detect_self_correction(tautology_rollout, is_correct=True)
        self.assertEqual(score, 0.0)

    def test_detect_self_correction_rewards_backtrack_recovery(self):
        # Backtrack recovery with ✗ and correction gets +0.40
        recovery_rollout = "<reg:deduce>hyp [x=5]✗ backtrack replan let(x=7) ∴ ans=7</reg:deduce><ans>7</ans>"
        score = detect_self_correction(recovery_rollout, is_correct=True)
        self.assertEqual(score, 0.40)


if __name__ == "__main__":
    unittest.main()
