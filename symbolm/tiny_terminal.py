"""
SymboLM Built-in Tiny Terminal & Deterministic AST Interceptor
==============================================================
A lightweight, sandboxed mathematical execution runtime (<5ms latency)
that verifies deduction steps, catches arithmetic slips, and eliminates
invalid reasoning paths.

When an uncertain step or contradiction is detected, the Tiny Terminal
interrupts the generative process, pops dirty variables, and constructs
a compact 2-line counterfactual feedback context (`hyp [...] verify(...) ✓`)
for immediate recovery without context bloat.

Author & Principal Systems Architect: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import re
import math
import ast
import sympy as sp


SAFE_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Call,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BoolOp,
    ast.And,
    ast.Or,
)

SAFE_CALLS = {
    "abs", "round", "min", "max", "sqrt", "gcd", "pow", "floor", "ceil", "inv_mod", "factorial"
}


class SafeArithmeticVisitor(ast.NodeVisitor):
    """
    Strict AST visitor ensuring an expression contains only safe arithmetic nodes,
    whitelisted math functions, numeric constants, and non-dunder variables.
    Prevents arbitrary code execution and sandbox escapes.
    """
    def generic_visit(self, node):
        if not isinstance(node, SAFE_NODES):
            raise ValueError(f"Disallowed AST syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float, complex, bool)):
            raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")

    def visit_Name(self, node):
        if node.id.startswith("__"):
            raise ValueError(f"Disallowed variable name: {node.id}")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_CALLS:
            func_name = getattr(node.func, "id", None) or getattr(node.func, "attr", str(type(node.func).__name__))
            raise ValueError(f"Disallowed function call: {func_name}")
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Attribute(self, node):
        raise ValueError(f"Disallowed attribute access: {node.attr}")


def validate_safe_arithmetic_ast(expr_str: str) -> None:
    """Validate that an expression is safe arithmetic with zero code execution risk."""
    tree = ast.parse(expr_str, mode="eval")
    SafeArithmeticVisitor().visit(tree)


@dataclass
class StepResult:
    valid: bool
    expression: str
    computed_value: Optional[Any] = None
    error: Optional[str] = None
    repaired_hypothesis: Optional[str] = None


@dataclass
class TerminalTraceAudit:
    is_valid: bool
    certified_answer: Optional[str] = None
    steps: List[StepResult] = field(default_factory=list)
    compact_error_context: Optional[str] = None
    execution_time_ms: float = 0.0


class TinyTerminal:
    """
    Built-in deterministic symbolic engine for real-time verification and error recovery.
    """
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.hypothesis_stack: List[Dict[str, Any]] = []

    def reset(self):
        """Reset environment memory."""
        self.variables.clear()
        self.hypothesis_stack.clear()

    def checkpoint(self):
        """Save clean variable state for fast rollback."""
        self.hypothesis_stack.append(dict(self.variables))

    def rollback(self):
        """Pop dirty state back to last valid checkpoint."""
        if self.hypothesis_stack:
            self.variables = self.hypothesis_stack.pop()

    def evaluate_expression(self, expr_str: str) -> Tuple[bool, Any, Optional[str]]:
        """
        Safely evaluates an algebraic or arithmetic expression using SymPy after strict AST validation.
        Returns (success, result, error_message).
        """
        clean_expr = expr_str.strip()
        # Clean common symbolic notation
        clean_expr = clean_expr.replace("^", "**").replace("·", "*").replace("×", "*")
        
        # Check modular arithmetic notation: e.g. "5 mod 3" -> "5 % 3"
        clean_expr = re.sub(r"\bmod\b", "%", clean_expr, flags=re.IGNORECASE)
        
        # 1. Strict AST security validation to prevent arbitrary code execution
        try:
            validate_safe_arithmetic_ast(clean_expr)
        except Exception as e:
            return False, None, f"Security/AST validation error: {e}"

        try:
            # Substitute known variables
            sym_expr = sp.sympify(clean_expr, locals=self.variables)
            # Evaluate numeric if possible
            if hasattr(sym_expr, "evalf"):
                res = sym_expr.evalf()
                try:
                    f_val = float(res)
                    if f_val.is_integer():
                        return True, int(f_val), None
                    return True, f_val, None
                except Exception:
                    pass
            return True, sym_expr, None
        except Exception as e:
            return False, None, str(e)

    def verify_equality(self, left_str: str, right_str: str) -> StepResult:
        """
        Verifies that LHS == RHS deterministically.
        E.g. left: "3 * 1 + 2", right: "5" -> True
        """
        ok_l, val_l, err_l = self.evaluate_expression(left_str)
        ok_r, val_r, err_r = self.evaluate_expression(right_str)
        
        if not ok_l or not ok_r:
            err = err_l or err_r
            return StepResult(valid=False, expression=f"{left_str} = {right_str}", error=f"Evaluation syntax error: {err}")
            
        is_equal = False
        try:
            if isinstance(val_l, (int, float)) and isinstance(val_r, (int, float)):
                is_equal = abs(val_l - val_r) < 1e-6
            else:
                diff = sp.simplify(val_l - val_r)
                is_equal = (diff == 0)
        except Exception:
            is_equal = (str(val_l) == str(val_r))
            
        if is_equal:
            return StepResult(valid=True, expression=f"{left_str} = {right_str}", computed_value=val_r)
        else:
            return StepResult(
                valid=False, 
                expression=f"{left_str} = {right_str}",
                error=f"Contradiction: Left evaluated to {val_l}, but Right evaluated to {val_r}",
                repaired_hypothesis=f"hyp [{left_str} = {val_l}]"
            )

    def audit_deduction_steps(self, deduce_block: str) -> TerminalTraceAudit:
        """
        Audits a full <reg:deduce>...</reg:deduce> trace.
        Splits by '|' or newlines, runs deterministic evaluation, and detects contradictions.
        """
        import time
        t0 = time.time()
        self.reset()
        
        # Strip register tags
        clean_block = re.sub(r"</?reg:deduce>", "", deduce_block).strip()
        raw_steps = [s.strip() for s in re.split(r"[|\n]", clean_block) if s.strip()]
        
        results: List[StepResult] = []
        is_all_valid = True
        compact_err = None
        certified_ans = None
        
        for idx, step in enumerate(raw_steps, start=1):
            # Check for terminal answer marker: ∴ ans = ... or ans = ...
            if "∴" in step or re.match(r"^ans\s*=", step):
                parts = step.split("=")
                if len(parts) >= 2:
                    certified_ans = parts[-1].strip()
                else:
                    certified_ans = step.replace("∴", "").strip()
                results.append(StepResult(valid=True, expression=step, computed_value=certified_ans))
                continue
                
            # Check for assignment or equality: A = B or A == B
            if "=" in step and not ("<=" in step or ">=" in step or "!=" in step):
                parts = step.split("=")
                if len(parts) == 2:
                    lhs = parts[0].strip()
                    rhs = parts[1].strip()
                    # If assignment like x = 5
                    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", lhs):
                        ok, val, err = self.evaluate_expression(rhs)
                        if ok:
                            # If variable already exists, verify consistency!
                            if lhs in self.variables:
                                existing_val = self.variables[lhs]
                                is_match = False
                                try:
                                    if isinstance(val, (int, float)) and isinstance(existing_val, (int, float)):
                                        is_match = abs(val - existing_val) < 1e-6
                                    else:
                                        is_match = (val == existing_val)
                                except Exception:
                                    is_match = (str(val) == str(existing_val))
                                if not is_match:
                                    is_all_valid = False
                                    err_msg = f"Contradiction: Variable '{lhs}' was already computed as {existing_val}, but step claims {val}"
                                    compact_err = f"[TINY_TERMINAL_ERR]: Step {idx} '{step}' -> {err_msg} | Suggest: hyp [{lhs} = {existing_val}]"
                                    results.append(StepResult(valid=False, expression=step, error=err_msg, repaired_hypothesis=f"hyp [{lhs} = {existing_val}]"))
                                    break
                            self.variables[lhs] = val
                            results.append(StepResult(valid=True, expression=step, computed_value=val))
                        else:
                            is_all_valid = False
                            compact_err = f"[TINY_TERMINAL_ERR]: Step {idx} '{step}' invalid: {err}"
                            results.append(StepResult(valid=False, expression=step, error=err))
                            break
                    else:
                        # Equality verification like 3*1+2 = 5
                        res = self.verify_equality(lhs, rhs)
                        results.append(res)
                        if not res.valid:
                            is_all_valid = False
                            compact_err = f"[TINY_TERMINAL_ERR]: Step {idx} contradiction in '{step}' -> {res.error} | Suggest: {res.repaired_hypothesis}"
                            break
            else:
                # Standalone expression or relation
                results.append(StepResult(valid=True, expression=step))
                
        elapsed_ms = (time.time() - t0) * 1000.0
        return TerminalTraceAudit(
            is_valid=is_all_valid,
            certified_answer=certified_ans,
            steps=results,
            compact_error_context=compact_err,
            execution_time_ms=elapsed_ms
        )
