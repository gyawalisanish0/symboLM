import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from training.reward import (
    extract_cognitive_depth,
    gradient_complexity_reward_func,
)


class TestGradientReward(unittest.TestCase):
    def test_extract_cognitive_depth(self):
        # 1. Explicit Scratchpad 2.0 depth attribute
        s1 = '<reg:scratchpad depth="2" domain="physics">\nstate(v=10)\n</reg:scratchpad>'
        self.assertEqual(extract_cognitive_depth(s1), "Δ2")

        s2 = '<reg:scratchpad depth="3">\ntarget: 5x ≡ 9 (mod 11)\n</reg:scratchpad>'
        self.assertEqual(extract_cognitive_depth(s2), "Δ3")

        # 2. Structural tags
        s_bypass = '<reg:bypass>K</reg:bypass>'
        self.assertEqual(extract_cognitive_depth(s_bypass), "Δ0")

        s_branch = 'hyp [gcd(5,11)=1]✓ | verify(x=4)✓ ∴ ans=4'
        self.assertEqual(extract_cognitive_depth(s_branch), "Δ2")

        s_deduce = '<reg:deduce>x=5 ∴ ans=5</reg:deduce>'
        self.assertEqual(extract_cognitive_depth(s_deduce), "Δ1")

    def test_gradient_elastic_depth_scaling(self):
        # Case A: Deep problem (Δ3) with 75 tokens
        # In Stage 4 (35-token limit), this would be severely penalized.
        # In Scratchpad 2.0 (Δ3: L_max = 150), this incurs ZERO length penalty!
        deep_proof = (
            '<think>\n'
            '<reg:scratchpad depth="3">\n'
            'target: Chinese Remainder Theorem system with moduli 7, 11, 13\n'
            '| step 1: M = 7 * 11 * 13 = 1001\n'
            '| step 2: M1 = 143, M2 = 91, M3 = 77\n'
            '| step 3: Bézout coefficients compute modular inverses\n'
            '| hyp [gcd(Mi, mi) == 1]✓\n'
            '| inv1 = 5 because 143 * 5 ≡ 3 * 5 = 15 ≡ 1 (mod 7)\n'
            '| inv2 = 4 because 91 * 4 ≡ 3 * 4 = 12 ≡ 1 (mod 11)\n'
            '| inv3 = 12 because 77 * 12 ≡ (-1) * (-1) = 1 (mod 13)\n'
            '| sum = a1*M1*inv1 + a2*M2*inv2 + a3*M3*inv3\n'
            '| verify(sum % 1001)✓\n'
            '∴ ans=527\n'
            '</reg:scratchpad>\n'
            '</think>\n'
            '<ans>527</ans>'
        )

        rewards_deep = gradient_complexity_reward_func(
            prompts=["Solve CRT"],
            completions=[deep_proof],
            answers=["527"],
            cognitive_depths=["Δ3"]
        )
        # Should have strong positive reward (correct + within L_max=150 + clean closure)
        self.assertGreater(rewards_deep[0], 2.0)
        print(f"[OK] Scratchpad 2.0 (Δ3, 75 tokens) Reward: {rewards_deep[0]:.2f} (Zero Length Penalty)")

        # Case B: Atomic fact (Δ0) with verbose 75 tokens
        # For Δ0 (L_max = 15), 75 tokens should be heavily penalized by elastic punishment!
        verbose_fact = (
            '<think>\n'
            '<reg:bypass>Let me think carefully about the element potassium. '
            'Potassium is an alkali metal located in group 1 and period 4 of the periodic table. '
            'Its chemical symbol is derived from Kalium.</reg:bypass>\n'
            '</think>\n'
            '<ans>K</ans>'
        )
        rewards_fact = gradient_complexity_reward_func(
            prompts=["What is the symbol for potassium?"],
            completions=[verbose_fact],
            answers=["K"],
            cognitive_depths=["Δ0"]
        )
        # Should be penalized for excessive verbosity on an atomic retrieval task
        self.assertLess(rewards_fact[0], 0.5)
        print(f"[OK] Atomic Fact (Δ0, 75 tokens) Elastic Penalty Applied: {rewards_fact[0]:.2f}")

        # Case C: Atomic fact (Δ0) with compact 3 tokens
        compact_fact = '<reg:bypass>K</reg:bypass>\n<ans>K</ans>'
        rewards_compact = gradient_complexity_reward_func(
            prompts=["What is the symbol for potassium?"],
            completions=[compact_fact],
            answers=["K"],
            cognitive_depths=["Δ0"]
        )
        self.assertGreater(rewards_compact[0], 2.5)
        print(f"[OK] Atomic Fact (Δ0, 3 tokens) Gradient Economy Reward: {rewards_compact[0]:.2f}")


if __name__ == "__main__":
    unittest.main()
