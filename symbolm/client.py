"""
SymboLM Runtime Client: Unified Inference Interface
===================================================
Author & Principal Systems Architect: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import json
import urllib.request
from typing import Optional, Dict, Any

from .registers import ReasoningTrace, parse_registers


class SymboLMClient:
    """
    High-level Python client for SymboLM reasoning models.
    Supports local Ollama C++ runtime (default: symbolm:4bit) or remote endpoints.
    """

    def __init__(
        self,
        model_name: str = "symbolm:4bit",
        endpoint: str = "http://127.0.0.1:11434",
        temperature: float = 0.6,
        max_tokens: int = 128
    ):
        self.model_name = model_name
        self.endpoint = endpoint.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Emits raw string response from the model."""
        url = f"{self.endpoint}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except Exception as e:
            raise ConnectionError(
                f"Failed to communicate with SymboLM Ollama backend at {self.endpoint}: {e}\n"
                f"Ensure Ollama is running (`ollama serve`) and model '{self.model_name}' is loaded."
            )

    def solve(self, prompt: str) -> ReasoningTrace:
        """
        Solves a reasoning or factual problem, returning a structured ReasoningTrace
        with Register v2.0 fields (<reg:plan>, <reg:deduce>, <reg:verify>, <ans>).
        """
        raw_output = self.generate(prompt)
        trace = parse_registers(raw_output)
        return trace


def load(model_name: str = "symbolm:4bit", endpoint: str = "http://127.0.0.1:11434") -> SymboLMClient:
    """Convenience factory to load a SymboLM client instance."""
    return SymboLMClient(model_name=model_name, endpoint=endpoint)
