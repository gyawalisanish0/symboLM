"""
SymboLM Abstract Syntax Tree (AST) Definitions
==============================================
Formal typed AST representation for the SymboLM Reasoning Language (SRL v1.0).
Author: Sanish Gyawali
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorLevel(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Diagnostic:
    code: str
    message: str
    level: ErrorLevel = ErrorLevel.ERROR
    line: int = 1
    col: int = 1

    def __str__(self) -> str:
        return f"[{self.code}] {self.level.value}: {self.message} (line {self.line}, col {self.col})"


@dataclass
class ASTNode:
    """Base class for all SRL AST nodes."""
    raw_text: str = ""


@dataclass
class StepNode(ASTNode):
    """Base class for an individual reasoning step within a trace."""
    step_index: int = 0


@dataclass
class DeclarationNode(StepNode):
    """Represents a variable declaration or premise: let(var = expr) or hyp(expr)."""
    var_name: Optional[str] = None
    expression: str = ""
    is_hypothesis: bool = False


@dataclass
class TransformNode(StepNode):
    """Represents an algebraic/arithmetic implication: lhs → rhs."""
    lhs: str = ""
    rhs: str = ""
    target_var: Optional[str] = None


@dataclass
class VerifyNode(StepNode):
    """Represents an invariant check: verify(condition)✓ or verify(condition)✗."""
    condition: str = ""
    passed: bool = True  # True if ✓, False if ✗


@dataclass
class StateFrameNode(StepNode):
    """Represents a structured state machine entry: state(k1=v1, k2=v2)."""
    state_dict: Dict[str, str] = field(default_factory=dict)


@dataclass
class IntentNode(StepNode):
    """Represents conversational intent planning: intent: tag | strategy: tag | tone: tag."""
    intent: str = ""
    strategy: Optional[str] = None
    tone: Optional[str] = None


@dataclass
class CaseNode(StepNode):
    """Represents a branch split: case(cond)[steps]."""
    condition: str = ""
    sub_steps: List[StepNode] = field(default_factory=list)


@dataclass
class ConclusionNode(ASTNode):
    """Represents the final conclusion step: ∴ ans = value."""
    value: str = ""


@dataclass
class AnswerNode(ASTNode):
    """Represents the exact final ground-truth tag: <ans>value</ans>."""
    value: str = ""


@dataclass
class TraceNode(ASTNode):
    """Root node of a parsed SRL trace."""
    steps: List[StepNode] = field(default_factory=list)
    conclusion: Optional[ConclusionNode] = None
    answer: Optional[AnswerNode] = None
    is_empty: bool = False
    is_intent_only: bool = False

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def get_declarations(self) -> List[DeclarationNode]:
        return [s for s in self.steps if isinstance(s, DeclarationNode)]

    def get_transformations(self) -> List[TransformNode]:
        return [s for s in self.steps if isinstance(s, TransformNode)]

    def get_verifications(self) -> List[VerifyNode]:
        return [s for s in self.steps if isinstance(s, VerifyNode)]
