"""
SymboLM: Biomimetic Multi-Register Reasoning & Adaptive Symbolic Engine
=======================================================================
Version: 2.0.0
Author & Principal Systems Architect: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
GitHub: https://github.com/gyawalisanish0/symboLM
Model Hub: https://huggingface.co/gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5
"""

__version__ = "2.0.0"
__author__ = "Sanish Gyawali"
__license__ = "MIT"

from .registers import (
    RegisterType,
    ReasoningTrace,
    parse_registers,
)
from .client import (
    SymboLMClient,
    load,
)

# Export core symbolic parser components for convenience
try:
    from symbolic.parser import SRLParser, evaluate_invariants
    from symbolic.grammar import SymbolicGrammar
    from symbolic.ast import PlanNode, StepNode, VerifyNode, TransformNode, IntentNode
except ImportError:
    pass

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "SymboLMClient",
    "load",
    "RegisterType",
    "ReasoningTrace",
    "parse_registers",
    "SRLParser",
    "evaluate_invariants",
    "SymbolicGrammar",
    "PlanNode",
    "StepNode",
    "VerifyNode",
    "TransformNode",
    "IntentNode",
]
