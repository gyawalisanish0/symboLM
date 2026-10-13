"""
SymboLM Register v2.0 Parser & Cognitive Register Abstractions
==============================================================
Author & Principal Systems Architect: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any


class RegisterType(str, Enum):
    BYPASS = "bypass"      # Δ0: System 1 Instant Reflex (0-step factual retrieval)
    INTENT = "intent"      # Social / Emotional Pragmatic Scratchpad (dialogue)
    PLAN = "plan"          # Δ2/Δ3: System 2 Invariant & Theorem Planning
    DEDUCE = "deduce"      # Δ1-Δ3: Pure Symbolic AST Execution
    VERIFY = "verify"      # Runtime Assertion & Backtracking (✓ / ✗)
    STANDARD = "standard"  # Unclassified reasoning


@dataclass
class ReasoningTrace:
    raw_text: str
    register_type: RegisterType
    plan: Optional[str] = None
    deduction: Optional[str] = None
    verification: Optional[str] = None
    bypass: Optional[str] = None
    intent: Optional[str] = None
    answer: Optional[str] = None
    token_count: Optional[int] = None
    ast_valid: bool = False
    metadata: Optional[Dict[str, Any]] = None

    @property
    def is_compact(self) -> bool:
        """Returns True if the trace achieves significant compression (< 50 tokens)."""
        return self.token_count is not None and self.token_count < 50


def parse_registers(completion_text: str) -> ReasoningTrace:
    """
    Parses a raw SymboLM completion into its structured Register v2.0 components.
    
    Supports:
      • <reg:bypass>...</reg:bypass>
      • <reg:intent>...</reg:intent>
      • <reg:plan>...</reg:plan>
      • <reg:deduce>...</reg:deduce>
      • <reg:verify>...</reg:verify>
      • <ans>...</ans> or \\boxed{...}
    """
    text = completion_text.strip()
    
    # 1. Check for Zero-Shot Factual Bypass (<reg:bypass>)
    bypass_match = re.search(r"<reg:bypass>(.*?)</reg:bypass>", text, re.DOTALL)
    bypass_val = bypass_match.group(1).strip() if bypass_match else None
    
    # 2. Check for Social / Pragmatic Intent (<reg:intent>)
    intent_match = re.search(r"<reg:intent>(.*?)</reg:intent>", text, re.DOTALL)
    intent_val = intent_match.group(1).strip() if intent_match else None
    
    # 3. Check for Invariant Planning (<reg:plan>)
    plan_match = re.search(r"<reg:plan>(.*?)</reg:plan>", text, re.DOTALL)
    plan_val = plan_match.group(1).strip() if plan_match else None
    
    # 4. Check for Symbolic AST Deduction (<reg:deduce>)
    deduce_match = re.search(r"<reg:deduce>(.*?)</reg:deduce>", text, re.DOTALL)
    deduce_val = deduce_match.group(1).strip() if deduce_match else None
    
    # 5. Check for Runtime Assertion Verification (<reg:verify>)
    verify_match = re.search(r"<reg:verify>(.*?)</reg:verify>", text, re.DOTALL)
    verify_val = verify_match.group(1).strip() if verify_match else None
    
    # 6. Extract Answer (<ans>...</ans> or \\boxed{...})
    ans_match = re.search(r"<ans>(.*?)</ans>", text, re.DOTALL)
    if not ans_match:
        ans_match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    ans_val = ans_match.group(1).strip() if ans_match else None
    
    # If no <ans> tag, and bypass is present, the bypass itself is the answer
    if not ans_val and bypass_val:
        ans_val = bypass_val
    
    # Classify dominant register
    if bypass_val:
        dominant = RegisterType.BYPASS
    elif intent_val and not deduce_val:
        dominant = RegisterType.INTENT
    elif plan_val or deduce_val or verify_val:
        dominant = RegisterType.DEDUCE
    else:
        dominant = RegisterType.STANDARD
    
    # Basic token approximation (words * 1.3 or space split)
    approx_tokens = len(text.split())
    
    return ReasoningTrace(
        raw_text=text,
        register_type=dominant,
        plan=plan_val,
        deduction=deduce_val,
        verification=verify_val,
        bypass=bypass_val,
        intent=intent_val,
        answer=ans_val,
        token_count=approx_tokens,
        ast_valid=bool(deduce_val or bypass_val or ans_val)
    )
