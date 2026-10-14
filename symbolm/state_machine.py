"""
SymboLM State Machine & Counterfactual Branch Coordinator
==========================================================
Coordinates register transitions, stack-based rollback, and counterfactual
branching (`hyp [...] verify(...) ✓`) in collaboration with the Built-in Tiny Terminal.

Author & Principal Systems Architect: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
from symbolm.tiny_terminal import TinyTerminal, TerminalTraceAudit


class EngineState(str, Enum):
    INIT = "INIT"
    ROUTED_BYPASS = "ROUTED_BYPASS"
    IN_PLAN = "IN_PLAN"
    IN_DEDUCE = "IN_DEDUCE"
    IN_VERIFY = "IN_VERIFY"
    COUNTERFACTUAL_ROLLBACK = "COUNTERFACTUAL_ROLLBACK"
    CERTIFIED_COMPLETE = "CERTIFIED_COMPLETE"
    ERROR_STATE = "ERROR_STATE"


@dataclass
class StateSnapshot:
    state: EngineState
    plan: Optional[str] = None
    deduce_steps: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    active_hypothesis: Optional[str] = None


class SymboStateMachine:
    """
    Manages deterministic register flow and 1-step rollback on Tiny Terminal contradiction.
    """
    def __init__(self, terminal: Optional[TinyTerminal] = None):
        self.terminal = terminal or TinyTerminal()
        self.current_state = EngineState.INIT
        self.history_stack: List[StateSnapshot] = []
        
        self.plan_text: Optional[str] = None
        self.deduce_text: Optional[str] = None
        self.verify_text: Optional[str] = None
        self.answer_text: Optional[str] = None
        self.bypass_text: Optional[str] = None
        
        self.error_context: Optional[str] = None
        self.rollback_count: int = 0

    def reset(self):
        """Reset state machine for new problem."""
        self.terminal.reset()
        self.current_state = EngineState.INIT
        self.history_stack.clear()
        self.plan_text = None
        self.deduce_text = None
        self.verify_text = None
        self.answer_text = None
        self.bypass_text = None
        self.error_context = None
        self.rollback_count = 0

    def route_to_bypass(self, bypass_content: str):
        """Route directly to atomic reflex bypass."""
        self.current_state = EngineState.ROUTED_BYPASS
        self.bypass_text = bypass_content.strip()
        self.answer_text = self.bypass_text

    def enter_plan(self, plan_content: str):
        """Set plan declaration and transition to deduction."""
        self.current_state = EngineState.IN_PLAN
        self.plan_text = plan_content.strip()

    def process_deduction(self, deduce_content: str) -> bool:
        """
        Executes Tiny Terminal audit on deduction block.
        If valid: transitions to IN_VERIFY.
        If invalid: captures compact error context and transitions to COUNTERFACTUAL_ROLLBACK.
        Returns True if valid, False if rollback needed.
        """
        self.deduce_text = deduce_content.strip()
        self.current_state = EngineState.IN_DEDUCE
        
        # Save snapshot before executing
        snapshot = StateSnapshot(
            state=self.current_state,
            plan=self.plan_text,
            deduce_steps=[self.deduce_text],
            variables=dict(self.terminal.variables)
        )
        self.history_stack.append(snapshot)
        
        audit: TerminalTraceAudit = self.terminal.audit_deduction_steps(self.deduce_text)
        
        if audit.is_valid:
            if audit.certified_answer:
                self.answer_text = audit.certified_answer
            self.current_state = EngineState.IN_VERIFY
            return True
        else:
            self.current_state = EngineState.COUNTERFACTUAL_ROLLBACK
            self.rollback_count += 1
            self.error_context = audit.compact_error_context
            return False

    def build_counterfactual_prompt(self, base_prompt: str) -> str:
        """
        Generates the compact 2-line recovery context for the LLM when rollback occurs.
        Eliminates verbose English backtraces and directly injects SRL v1.0 syntax.
        """
        if not self.error_context:
            return base_prompt
            
        recovery_context = (
            f"\n{self.error_context}\n"
            f"<think>\n<reg:plan>{self.plan_text}</reg:plan>\n"
            f"<reg:deduce>"
        )
        return recovery_context

    def commit_verification(self, verify_content: str, answer_content: str):
        """Finalize verification and certification."""
        self.verify_text = verify_content.strip()
        self.answer_text = answer_content.strip()
        self.current_state = EngineState.CERTIFIED_COMPLETE
