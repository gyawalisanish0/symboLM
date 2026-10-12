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
    unbounded_parsimony_reward_func,
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

    def test_unbounded_parsimony_gradient(self):
        """
        Tests Sanish Gyawali's Unbounded Parsimony Gradient:
        1. Never locks or truncates deep proofs.
        2. Correct proofs of ANY length remain strictly positive (R > 0).
        3. Monotonically rewards using the least tokens possible.
        """
        prompt = "Find x"
        gt = "42"

        # Ultra-compact (5 tokens): model uses its latent knowledge directly
        trace_ultra = "<reg:bypass>42</reg:bypass>\n<ans>42</ans>"

        # Compact symbolic proof (25 tokens)
        trace_compact = "<think>\n<reg:deduce>2x=84 → x=42 ∴ ans=42</reg:deduce>\n</think>\n<ans>42</ans>"

        # Multi-step proof (60 tokens)
        trace_multistep = (
            "<think>\n"
            "<reg:plan>linear_equation | target: x</reg:plan>\n"
            "<reg:deduce>step1: 3x - x = 84 → 2x = 84 | step2: x = 84/2 = 42 ∴ ans=42</reg:deduce>\n"
            "<reg:verify>verify(2*42==84)✓</reg:verify>\n"
            "</think>\n"
            "<ans>42</ans>"
        )

        # Deep Olympiad derivation (120 tokens)
        trace_deep = (
            "<think>\n"
            "<reg:scratchpad depth=\"3\">\n"
            "state(system: 5x ≡ 21 mod 43)\n"
            "| hyp [gcd(5, 43) == 1]✓\n"
            "| Bézout identity: 43 = 8*5 + 3 → 5 = 1*3 + 2 → 3 = 1*2 + 1\n"
            "| backward substitution yields modular inverse 5^-1 ≡ 26 mod 43\n"
            "| x ≡ 26 * 21 = 546 ≡ 42 mod 43\n"
            "| verify(5*42 = 210 = 4*43 + 38 != 21? Recheck: 5*42 = 210 = 4*43 + 38)\n"
            "| verify(5*42 = 210 = 43*4 + 38, corrected: 43*5 - 5 = 210 ∴ 210 - 4*43 = 38)\n"
            "| final congruence verified: ans=42\n"
            "</reg:scratchpad>\n"
            "</think>\n"
            "<ans>42</ans>"
        )

        # Incorrect answer
        trace_wrong = "<think><reg:deduce>x=99</reg:deduce></think><ans>99</ans>"

        # Looping answer
        trace_loop = "<think><reg:deduce>42 42 42 42 42 42 42 42</reg:deduce></think><ans>42</ans>"

        rewards = unbounded_parsimony_reward_func(
            prompts=[prompt] * 6,
            completions=[trace_ultra, trace_compact, trace_multistep, trace_deep, trace_wrong, trace_loop],
            answers=[gt] * 6,
        )

        r_ultra, r_compact, r_multi, r_deep, r_wrong, r_loop = rewards

        # 1. Monotonicity: Fewer tokens -> Strictly higher reward!
        self.assertGreater(r_ultra, r_compact)
        self.assertGreater(r_compact, r_multi)
        self.assertGreater(r_multi, r_deep)

        # 2. Zero Token Lock: Deep proof (120 tokens) is STILL strongly POSITIVE!
        self.assertGreater(r_deep, 1.50)
        self.assertGreater(r_ultra, 2.50)

        # 3. Wrong answer is strictly negative
        self.assertLess(r_wrong, 0.0)

        # 4. Looping repetition is penalized
        self.assertLess(r_loop, 0.50)

        print("\n[VERIFIED] Sanish Gyawali's Unbounded Parsimony Gradient:")
        print(f"  • Ultra-compact (5 tokens)  : Reward = +{r_ultra:.2f} (Peak knowledge reward)")
        print(f"  • Compact proof (25 tokens) : Reward = +{r_compact:.2f}")
        print(f"  • Multi-step    (60 tokens) : Reward = +{r_multi:.2f}")
        print(f"  • Deep proof   (120 tokens) : Reward = +{r_deep:.2f} (STILL POSITIVE, NO LOCK!)")
        print(f"  • Wrong answer  (any)       : Reward = {r_wrong:.2f}")
        print(f"  • Looping trace             : Reward = {r_loop:.2f}")


if __name__ == "__main__":
    unittest.main()
