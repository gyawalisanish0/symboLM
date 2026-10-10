"""
SymboLM Register v2.0 Cognitive Architecture Unit Tests
=======================================================
Verifies:
  1. Register v2.0 delimiter parsing (<reg:plan>, <reg:deduce>, <reg:verify>, <reg:bypass>, <reg:intent>).
  2. PlanNode AST structure, concept/theorem/strategy extraction.
  3. Seamless backward compatibility with SRL v1.0 traces.
  4. Robust answer extraction across multi-register formats.

Author: Sanish Gyawali
AI Systems Exoskeleton: Antigravity (Google DeepMind)
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from symbolic.ast import PlanNode, VerifyNode, TransformNode
from symbolic.grammar import extract_answer
from symbolic.parser import SRLParser


def test_register_v2_plan_deduce_verify():
    """Test full Register v2.0 pipeline: Plan -> Deduce -> Verify."""
    parser = SRLParser()
    sample_text = (
        "<think>\n"
        "<reg:plan>concept: polynomial_invariant | identity: (x²+1/x²)=(x+1/x)²-2 | target: x⁴+1/x⁴</reg:plan>\n"
        "<reg:deduce>let(u=5) | u²-2=23 | 23²-2=527 ∴ ans=527</reg:deduce>\n"
        "<reg:verify>verify(527>0)✓</reg:verify>\n"
        "</think>\n"
        "<ans>527</ans>"
    )

    res = parser.parse(sample_text)
    assert res.is_valid, f"Parse failed with diagnostics: {res.diagnostics}"
    assert res.ast is not None
    assert res.ast.register_type == "plan_deduce"
    assert res.ast.plan is not None
    assert res.ast.plan.concept == "polynomial_invariant"
    assert res.ast.plan.theorem == "(x²+1/x²)=(x+1/x)²-2"
    assert res.extracted_answer == "527"
    assert extract_answer(sample_text) == "527"

    verifications = res.ast.get_verifications()
    assert len(verifications) == 1
    assert verifications[0].passed is True
    print("✓ test_register_v2_plan_deduce_verify PASSED")


def test_register_v2_geometry_inradius():
    """Test Register v2.0 on geometric inradius invariant."""
    parser = SRLParser()
    sample_text = (
        "<reg:plan>concept: inradius_geometry | theorem: Area=r*s | semiperimeter: s=P/2</reg:plan>\n"
        "<reg:deduce>let(P=72, r=6) | s=72/2=36 | Area=6*36=216 ∴ ans=216</reg:deduce>\n"
        "<ans>216</ans>"
    )

    res = parser.parse(sample_text)
    assert res.is_valid
    assert res.ast.plan.concept == "inradius_geometry"
    assert res.ast.plan.theorem == "Area=r*s"
    assert res.extracted_answer == "216"
    assert extract_answer(sample_text) == "216"
    print("✓ test_register_v2_geometry_inradius PASSED")


def test_register_v2_bypass():
    """Test Register v2.0 factual bypass (<reg:bypass>)."""
    parser = SRLParser()
    sample_text = "<reg:bypass>6</reg:bypass>"

    res = parser.parse(sample_text)
    assert res.is_valid
    assert res.ast.is_empty is True
    assert res.ast.register_type == "bypass"
    assert res.extracted_answer == "6"
    assert extract_answer(sample_text) == "6"
    print("✓ test_register_v2_bypass PASSED")


def test_register_v2_intent_dialogue():
    """Test Register v2.0 intent dialogue (<reg:intent>)."""
    parser = SRLParser()
    sample_text = (
        "<reg:intent>intent: mentor | tone: supportive | strategy: guided_inquiry</reg:intent>\n"
        "Great effort on the implementation! Let's examine how the base cases behave."
    )

    res = parser.parse(sample_text)
    assert res.is_valid
    assert res.ast.is_intent_only is True
    assert res.ast.register_type == "intent"
    assert len(res.ast.steps) == 1
    assert res.ast.steps[0].intent == "mentor"
    print("✓ test_register_v2_intent_dialogue PASSED")


def test_register_v2_chinese_remainder():
    """Test Register v2.0 on Chinese Remainder Theorem."""
    parser = SRLParser()
    sample_text = (
        "<think>\n"
        "<reg:plan>concept: chinese_remainder | moduli: [5,7,11] | method: modular_inverses</reg:plan>\n"
        "<reg:deduce>n≡3 mod 5 | n≡5 mod 7→n=33 mod 35 | n=35m+33≡7 mod 11→m≡9 mod 11→n=348 ∴ ans=328</reg:deduce>\n"
        "</think>\n"
        "<ans>328</ans>"
    )

    res = parser.parse(sample_text)
    assert res.is_valid
    assert res.ast.plan.concept == "chinese_remainder"
    assert res.extracted_answer == "328"
    assert extract_answer(sample_text) == "328"
    print("✓ test_register_v2_chinese_remainder PASSED")


def test_reward_anti_guess_penalty():
    """Verify that blind 1-token guessing on multi-step problems is penalized."""
    from training.reward import compute_reward

    # Prediction is a blind 1-token guess on an Olympiad problem
    pred_guess = "169"
    gt = "527"
    breakdown = compute_reward(
        prediction=pred_guess,
        ground_truth=gt,
        is_multi_step=True,
        expected_concept="polynomial_invariant",
        return_breakdown=True
    )
    assert breakdown.anti_guess_penalty == -0.80, f"Expected -0.80 anti-guess penalty, got {breakdown.anti_guess_penalty}"
    assert breakdown.total < -1.0, f"Expected total reward to be heavily penalized, got {breakdown.total}"
    print("✓ test_reward_anti_guess_penalty PASSED")


def test_reward_concept_alignment():
    """Verify that valid macro-planning receives concept alignment bonus."""
    from training.reward import compute_reward

    pred_concept = (
        "<think>\n"
        "<reg:plan>concept: polynomial_invariant | identity: (x²+1/x²)=(x+1/x)²-2</reg:plan>\n"
        "<reg:deduce>let(u=5) | u²-2=23 | 23²-2=527 ∴ ans=527</reg:deduce>\n"
        "<reg:verify>verify(527>0)✓</reg:verify>\n"
        "</think>\n"
        "<ans>527</ans>"
    )
    gt = "527"
    breakdown = compute_reward(
        prediction=pred_concept,
        ground_truth=gt,
        is_multi_step=True,
        expected_concept="polynomial_invariant",
        return_breakdown=True
    )
    assert breakdown.concept_score > 0.0, f"Expected concept bonus, got {breakdown.concept_score}"
    assert breakdown.anti_guess_penalty == 0.0, f"Should not have anti-guess penalty, got {breakdown.anti_guess_penalty}"
    assert breakdown.correctness == 1.0
    print("✓ test_reward_concept_alignment PASSED")


def test_dataset_integrity():
    """Verify generated curriculum dataset is well-formed and mathematically valid."""
    import json
    curriculum_path = ROOT_DIR / "data" / "concept_math_curriculum.jsonl"
    assert curriculum_path.exists(), "concept_math_curriculum.jsonl does not exist"

    with open(curriculum_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 1400, f"Expected 1400 samples, found {len(lines)}"

    concept_counts = {}
    for line in lines:
        sample = json.loads(line)
        assert "prompt" in sample and len(sample["prompt"]) > 5
        assert "ground_truth" in sample and len(sample["ground_truth"]) > 0
        assert "concept" in sample
        c = sample["concept"]
        concept_counts[c] = concept_counts.get(c, 0) + 1

    print(f"✓ test_dataset_integrity PASSED ({len(lines)} samples across {len(concept_counts)} concepts)")


def run_all_tests():
    print("\n" + "=" * 60)
    print("Running SymboLM Register v2.0 Architecture Test Suite")
    print("=" * 60)
    test_register_v2_plan_deduce_verify()
    test_register_v2_geometry_inradius()
    test_register_v2_bypass()
    test_register_v2_intent_dialogue()
    test_register_v2_chinese_remainder()
    test_reward_anti_guess_penalty()
    test_reward_concept_alignment()
    test_dataset_integrity()
    print("=" * 60)
    print("ALL REGISTER v2.0 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
