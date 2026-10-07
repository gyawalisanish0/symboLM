"""
SymboLM Parser & Invariant Evaluator (SRL v1.0)
==============================================
Deterministic Recursive Descent Parser and SymPy AST Invariant Verifier.
Author: Sanish Gyawali
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import sympy as sp

from symbolic.ast import (
    AnswerNode,
    CaseNode,
    ConclusionNode,
    DeclarationNode,
    Diagnostic,
    ErrorLevel,
    IntentNode,
    StateFrameNode,
    StepNode,
    TraceNode,
    TransformNode,
    VerifyNode,
)


@dataclass
class ParseResult:
    """Encapsulates the result of parsing an SRL string."""
    ast: Optional[TraceNode]
    is_valid: bool
    diagnostics: List[Diagnostic] = field(default_factory=list)
    extracted_answer: Optional[str] = None

    @property
    def has_errors(self) -> bool:
        return any(d.level == ErrorLevel.ERROR for d in self.diagnostics)


class SRLParser:
    """Parser and validator for the SymboLM Reasoning Language (SRL v1.0)."""

    def __init__(self, strict_math: bool = True):
        self.strict_math = strict_math

    def parse(self, text: str) -> ParseResult:
        """Parses a full document containing <think>...</think> and optional <ans>...</ans>."""
        diagnostics: List[Diagnostic] = []
        text = text.strip()

        # 1. Extract <ans>...</ans>
        ans_val: Optional[str] = None
        ans_match = re.search(r"<ans>(.*?)</ans>", text, re.DOTALL)
        if ans_match:
            ans_val = ans_match.group(1).strip()
        ans_node = AnswerNode(raw_text=ans_val or "", value=ans_val or "") if ans_val else None

        # 2. Extract <think>...</think>
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if not think_match:
            # Check if text is just the raw trace without tags
            if "<think>" in text or "</think>" in text:
                diagnostics.append(Diagnostic("E101", "Malformed or unclosed <think> tag."))
                return ParseResult(ast=None, is_valid=False, diagnostics=diagnostics, extracted_answer=ans_val)
            trace_content = text
        else:
            trace_content = think_match.group(1).strip()

        # Handle empty thinking block (Factual zero-shot bypass)
        if not trace_content:
            ast = TraceNode(raw_text="", is_empty=True, answer=ans_node)
            return ParseResult(ast=ast, is_valid=True, diagnostics=diagnostics, extracted_answer=ans_val)

        # 3. Check for Intent Directive (Interpersonal dialogue register)
        if trace_content.startswith("intent:") or "intent:" in trace_content:
            intent_node = self._parse_intent_directive(trace_content)
            ast = TraceNode(
                raw_text=trace_content,
                steps=[intent_node],
                is_intent_only=True,
                answer=ans_node
            )
            return ParseResult(ast=ast, is_valid=True, diagnostics=diagnostics, extracted_answer=ans_val)

        # 4. Parse reasoning trace
        ast, trace_diag = self._parse_reasoning_trace(trace_content)
        diagnostics.extend(trace_diag)

        if ast:
            ast.answer = ans_node
            # Check answer alignment if both present
            if ast.conclusion and ans_node:
                if ast.conclusion.value.strip() != ans_node.value.strip():
                    diagnostics.append(
                        Diagnostic(
                            "E203",
                            f"Answer mismatch: Conclusion has '{ast.conclusion.value}' but <ans> has '{ans_node.value}'.",
                            level=ErrorLevel.WARNING
                        )
                    )

        # 5. Check for natural language leakage (Hygiene)
        leak_words = ["let me think", "first I need to", "step by step", "therefore we see", "in order to"]
        for lw in leak_words:
            if lw in trace_content.lower():
                diagnostics.append(
                    Diagnostic("E301", f"Conversational English filler detected: '{lw}'", level=ErrorLevel.WARNING)
                )

        extracted = ast.conclusion.value if (ast and ast.conclusion) else ans_val
        is_valid = not any(d.level == ErrorLevel.ERROR for d in diagnostics)
        return ParseResult(ast=ast, is_valid=is_valid, diagnostics=diagnostics, extracted_answer=extracted)

    def _parse_reasoning_trace(self, content: str) -> Tuple[Optional[TraceNode], List[Diagnostic]]:
        """Parses a reasoning sequence containing steps and a conclusion."""
        diagnostics: List[Diagnostic] = []
        conclusion_node: Optional[ConclusionNode] = None

        # Check for conclusion marker: ∴
        if "∴" in content:
            parts = content.split("∴", 1)
            steps_part = parts[0].strip()
            conc_part = parts[1].strip()

            # Parse conclusion: ans = <val>
            conc_match = re.search(r"ans\s*=\s*(.+)", conc_part)
            if conc_match:
                conclusion_node = ConclusionNode(raw_text=conc_part, value=conc_match.group(1).strip())
            else:
                conclusion_node = ConclusionNode(raw_text=conc_part, value=conc_part.strip())
        else:
            steps_part = content.strip()
            diagnostics.append(Diagnostic("E102", "Missing conclusion marker ('∴ ans=').", level=ErrorLevel.WARNING))

        # Split steps by '|'
        raw_steps = [s.strip() for s in steps_part.split("|") if s.strip()]
        if not raw_steps and not conclusion_node:
            diagnostics.append(Diagnostic("E104", "No valid steps found in trace.", level=ErrorLevel.ERROR))
            return None, diagnostics

        parsed_steps: List[StepNode] = []
        for idx, raw_step in enumerate(raw_steps):
            step_node = self._parse_single_step(raw_step, idx)
            if step_node:
                parsed_steps.append(step_node)
            else:
                diagnostics.append(
                    Diagnostic("E104", f"Could not parse step {idx + 1}: '{raw_step}'", level=ErrorLevel.WARNING)
                )

        ast = TraceNode(raw_text=content, steps=parsed_steps, conclusion=conclusion_node)
        return ast, diagnostics

    def _parse_single_step(self, step_str: str, index: int) -> Optional[StepNode]:
        """Classifies and parses an individual step into a typed StepNode."""
        step_str = step_str.strip()

        # 1. Declaration: let(var = expr)
        let_match = re.match(r"let\s*\(\s*([a-zA-Z0-9_]+)\s*=\s*(.+?)\s*\)$", step_str)
        if let_match:
            return DeclarationNode(
                raw_text=step_str,
                step_index=index,
                var_name=let_match.group(1),
                expression=let_match.group(2),
                is_hypothesis=False,
            )

        # 2. Hypothesis: hyp(expr)
        hyp_match = re.match(r"hyp\s*\(\s*(.+?)\s*\)$", step_str)
        if hyp_match:
            return DeclarationNode(
                raw_text=step_str,
                step_index=index,
                var_name=None,
                expression=hyp_match.group(1),
                is_hypothesis=True,
            )

        # 3. Verification: verify(cond)✓ or verify(cond)✗
        verify_match = re.match(r"verify\s*\(\s*(.+?)\s*\)\s*([✓✗])?$", step_str)
        if verify_match:
            cond = verify_match.group(1)
            marker = verify_match.group(2) or "✓"
            return VerifyNode(
                raw_text=step_str,
                step_index=index,
                condition=cond,
                passed=(marker == "✓"),
            )

        # 4. StateFrame: state(k1=v1, k2=v2)
        state_match = re.match(r"state\s*\((.*?)\)$", step_str)
        if state_match:
            pairs = {}
            for pair in state_match.group(1).split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    pairs[k.strip()] = v.strip()
            return StateFrameNode(raw_text=step_str, step_index=index, state_dict=pairs)

        # 5. Transformation: lhs → rhs
        if "→" in step_str:
            parts = step_str.split("→", 1)
            lhs = parts[0].strip()
            rhs = parts[1].strip()

            target_var = None
            if "=" in rhs:
                v_parts = rhs.split("=", 1)
                target_var = v_parts[0].strip()

            return TransformNode(
                raw_text=step_str,
                step_index=index,
                lhs=lhs,
                rhs=rhs,
                target_var=target_var,
            )

        # Fallback: Treat as implicit transformation or atomic expression
        return TransformNode(raw_text=step_str, step_index=index, lhs=step_str, rhs=step_str)

    def _parse_intent_directive(self, content: str) -> IntentNode:
        """Parses an intent directive block."""
        parts = [p.strip() for p in content.split("|")]
        intent_val = ""
        strat_val = None
        tone_val = None

        for p in parts:
            if p.startswith("intent:"):
                intent_val = p.replace("intent:", "").strip()
            elif p.startswith("strategy:"):
                strat_val = p.replace("strategy:", "").strip()
            elif p.startswith("tone:"):
                tone_val = p.replace("tone:", "").strip()

        return IntentNode(
            raw_text=content,
            intent=intent_val,
            strategy=strat_val,
            tone=tone_val,
        )


def evaluate_invariants(ast: TraceNode) -> List[Diagnostic]:
    """
    Evaluates mathematical invariants in the AST using SymPy with state tracking.
    Ensures that every declaration, mutation, and implication step maintains truth invariance.
    """
    diagnostics: List[Diagnostic] = []
    env: dict[str, Any] = {}

    for step in ast.steps:
        # 1. Track Declarations: let(var = expr)
        if isinstance(step, DeclarationNode):
            if step.var_name and step.expression:
                try:
                    val = sp.sympify(step.expression)
                    env[step.var_name] = val
                except Exception:
                    pass

        # 2. Check and Track Transformations: lhs → rhs
        elif isinstance(step, TransformNode):
            lhs = step.lhs.strip()
            rhs = step.rhs.strip()

            # Pattern A: Variable Mutation e.g. tot -= 3 or eat -= 3
            mut_match = re.match(r"^([a-zA-Z0-9_]+)\s*([-+*/])=\s*(.+)$", lhs)
            if mut_match:
                v_name, op, delta_str = mut_match.groups()
                # If variable is in env, or if env has only 1 variable (e.g. tot=16)
                cur_val = env.get(v_name)
                active_key = v_name
                if cur_val is None and len(env) == 1:
                    active_key = list(env.keys())[0]
                    cur_val = env[active_key]

                if cur_val is not None:
                    try:
                        delta = sp.sympify(delta_str)
                        if op == "-":
                            new_val = cur_val - delta
                        elif op == "+":
                            new_val = cur_val + delta
                        elif op == "*":
                            new_val = cur_val * delta
                        elif op == "/":
                            new_val = cur_val / delta
                        else:
                            new_val = cur_val

                        expected_rhs = sp.sympify(rhs)
                        if new_val != expected_rhs:
                            diagnostics.append(
                                Diagnostic(
                                    "E201",
                                    f"Invariant violation at step {step.step_index + 1}: {cur_val} {op} {delta} = {new_val}, but got '{rhs}'.",
                                    level=ErrorLevel.ERROR
                                )
                            )
                        else:
                            env[active_key] = new_val
                        continue
                    except Exception:
                        pass

            # Pattern B: Function/Wrapper call e.g. sell(9 * 2) -> 18
            fn_match = re.match(r"^[a-zA-Z0-9_]+\s*\((.+)\)$", lhs)
            if fn_match:
                inner_expr = fn_match.group(1).strip()
                try:
                    res_inner = sp.sympify(inner_expr)
                    expected_rhs = sp.sympify(rhs)
                    if res_inner != expected_rhs:
                        diagnostics.append(
                            Diagnostic(
                                "E201",
                                f"Invariant violation at step {step.step_index + 1}: '{inner_expr}' evaluates to {res_inner}, but got '{rhs}'.",
                                level=ErrorLevel.ERROR
                            )
                        )
                    continue
                except Exception:
                    pass

            # Pattern C: Equation Transitions e.g. 5x-15=3x+25 or -3x→2x-15=25 or +15→2x=40
            if "=" in rhs and not any(op in rhs for op in ["==", "<=", ">="]):
                try:
                    l_side, r_side = rhs.split("=", 1)
                    # Convert implicit 2x, 3x to 2*x, 3*x for sympy
                    clean_l = re.sub(r"(\d+)([a-zA-Z])", r"\1*\2", l_side.strip())
                    clean_r = re.sub(r"(\d+)([a-zA-Z])", r"\1*\2", r_side.strip())
                    eq_expr = sp.sympify(f"({clean_l}) - ({clean_r})")

                    active_eq = env.get("__active_eq__")
                    if active_eq is not None:
                        sol_active = sp.solve(active_eq)
                        sol_curr = sp.solve(eq_expr)
                        if sol_active and sol_curr and sol_active != sol_curr:
                            diagnostics.append(
                                Diagnostic(
                                    "E201",
                                    f"Invariant violation at step {step.step_index + 1}: Equation '{rhs}' solution {sol_curr} does not match previous solution {sol_active}.",
                                    level=ErrorLevel.ERROR
                                )
                            )

                    env["__active_eq__"] = eq_expr
                    # If single variable is solved (e.g. x = 20), store in env
                    free_syms = list(eq_expr.free_symbols)
                    if len(free_syms) == 1:
                        sol = sp.solve(eq_expr, free_syms[0])
                        if sol:
                            env[str(free_syms[0])] = sol[0]
                    continue
                except Exception:
                    pass

            # Pattern D: Direct Arithmetic Check e.g. 10 - 3 -> 7
            clean_lhs = lhs
            clean_rhs = rhs
            if any(c in clean_lhs for c in "+-*/^0123456789") and clean_rhs:
                try:
                    expr_l = sp.sympify(clean_lhs)
                    expr_r = sp.sympify(clean_rhs)
                    diff = sp.simplify((expr_l - expr_r).subs(env))
                    if diff != 0:
                        diagnostics.append(
                            Diagnostic(
                                "E201",
                                f"Invariant violation at step {step.step_index + 1}: '{clean_lhs}' does not equal '{clean_rhs}'.",
                                level=ErrorLevel.ERROR
                            )
                        )
                except Exception:
                    pass

        # 3. Check Verification Nodes
        elif isinstance(step, VerifyNode):
            cond = step.condition
            if "==" in cond:
                l_part, r_part = cond.split("==", 1)
                try:
                    res_l = sp.sympify(l_part.strip()).subs(env)
                    res_r = sp.sympify(r_part.strip()).subs(env)
                    is_equal = (res_l == res_r)
                    if is_equal != step.passed:
                        diagnostics.append(
                            Diagnostic(
                                "E202",
                                f"Verification mismatch at step {step.step_index + 1}: '{cond}' evaluated {is_equal}, but marked {step.passed}.",
                                level=ErrorLevel.ERROR
                            )
                        )
                except Exception:
                    pass

    return diagnostics
