"""
Unit Tests for SymboLM Reasoning Language (SRL v1.0) Parser & Invariant Evaluator
==================================================================================
Author: Sanish Gyawali
"""

import sys
from pathlib import Path

# Configure utf-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from symbolic.ast import (
    DeclarationNode,
    IntentNode,
    StateFrameNode,
    TransformNode,
    VerifyNode,
)
from symbolic.parser import SRLParser, evaluate_invariants


def test_valid_gsm8k_trace():
    text = "<think>\nlet(tot=16) | eat-=3→13 | bake-=4→9 | sell(9*2)→18 ∴ ans=18\n</think>\n<ans>18</ans>"
    parser = SRLParser()
    res = parser.parse(text)

    assert res.is_valid, f"Parse failed with diagnostics: {res.diagnostics}"
    assert res.ast is not None
    assert res.ast.total_steps == 4
    assert res.extracted_answer == "18"

    # Step types
    assert isinstance(res.ast.steps[0], DeclarationNode)
    assert res.ast.steps[0].var_name == "tot"
    assert res.ast.steps[0].expression == "16"

    assert isinstance(res.ast.steps[1], TransformNode)
    assert res.ast.steps[1].rhs == "13"

    # Invariants
    diag = evaluate_invariants(res.ast)
    assert len(diag) == 0, f"Invariant failure: {diag}"
    print("✓ test_valid_gsm8k_trace PASSED")


def test_valid_math_algebra_trace():
    text = (
        "<think>\n"
        "5x-15=3x+25 | -3x→2x-15=25 | +15→2x=40 | /2→x=20 | verify(5*20-15==3*20+25)✓ ∴ ans=20\n"
        "</think>\n"
        "<ans>20</ans>"
    )
    parser = SRLParser()
    res = parser.parse(text)

    assert res.is_valid, f"Parse failed: {res.diagnostics}"
    assert res.ast is not None
    assert res.ast.total_steps == 5

    # Check verification node
    ver_nodes = res.ast.get_verifications()
    assert len(ver_nodes) == 1
    assert ver_nodes[0].passed is True

    # Invariant evaluation
    diag = evaluate_invariants(res.ast)
    assert len(diag) == 0, f"Invariant error: {diag}"
    print("✓ test_valid_math_algebra_trace PASSED")


def test_intent_directive_register():
    text = (
        "<think>\n"
        "intent: validate_frustration | strategy: document_evidence | tone: professional_calm\n"
        "</think>\n"
        "That is deeply frustrating. To handle this constructively..."
    )
    parser = SRLParser()
    res = parser.parse(text)

    assert res.is_valid
    assert res.ast is not None
    assert res.ast.is_intent_only is True
    assert isinstance(res.ast.steps[0], IntentNode)
    assert res.ast.steps[0].intent == "validate_frustration"
    assert res.ast.steps[0].strategy == "document_evidence"
    assert res.ast.steps[0].tone == "professional_calm"
    print("✓ test_intent_directive_register PASSED")


def test_questionnaire_state_frame_register():
    text = "<think>\nstate(fever=102, cough=5d, O2=96%) | urgency=Med ∴ ans=Med\n</think>\n<ans>Med</ans>"
    parser = SRLParser()
    res = parser.parse(text)

    assert res.is_valid
    assert res.ast is not None
    assert isinstance(res.ast.steps[0], StateFrameNode)
    assert res.ast.steps[0].state_dict["fever"] == "102"
    assert res.ast.steps[0].state_dict["O2"] == "96%"
    assert res.extracted_answer == "Med"
    print("✓ test_questionnaire_state_frame_register PASSED")


def test_zero_shot_factual_bypass():
    text = "<think></think>\nLima"
    parser = SRLParser()
    res = parser.parse(text)

    assert res.is_valid
    assert res.ast is not None
    assert res.ast.is_empty is True
    print("✓ test_zero_shot_factual_bypass PASSED")


def test_invariant_violation_caught():
    # Intentionally false arithmetic: 10 - 3 -> 8
    text = "<think>\nlet(x=10) | 10-3→8 ∴ ans=8\n</think>\n<ans>8</ans>"
    parser = SRLParser()
    res = parser.parse(text)
    assert res.ast is not None

    diag = evaluate_invariants(res.ast)
    assert len(diag) > 0, "Failed to catch arithmetic violation!"
    assert any(d.code == "E201" for d in diag)
    print("✓ test_invariant_violation_caught PASSED (Correctly flagged E201 arithmetic mismatch)")


def test_verification_mismatch_caught():
    # Intentionally false assertion: 5 == 10 marked as passed
    text = "<think>\nverify(5 == 10)✓ ∴ ans=Wrong\n</think>"
    parser = SRLParser()
    res = parser.parse(text)
    assert res.ast is not None

    diag = evaluate_invariants(res.ast)
    assert len(diag) > 0
    assert any(d.code == "E202" for d in diag)
    print("✓ test_verification_mismatch_caught PASSED (Correctly flagged E202 verification error)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running SymboLM SRL v1.0 Test Suite")
    print("=" * 60)

    test_valid_gsm8k_trace()
    test_valid_math_algebra_trace()
    test_intent_directive_register()
    test_questionnaire_state_frame_register()
    test_zero_shot_factual_bypass()
    test_invariant_violation_caught()
    test_verification_mismatch_caught()

    print("\n" + "=" * 60)
    print("ALL SRL v1.0 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
