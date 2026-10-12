"""
SymboLM Standalone Merged Model Verification
============================================
Performs a live smoke-test on the newly fused standalone model
(checkpoints/SymboLM-1.5B-v1.0-DSR1-Qwen2.5) WITHOUT PEFT wrappers.

Verifies:
  1. Δ0 Atomic Factual Bypass (<reg:bypass>)
  2. Δ1 Linear Polynomial Invariant (<reg:plan> + <reg:deduce>)
  3. Δ3 Deep Congruence / Chinese Remainder Theorem (Scratchpad 2.0)

Principal Systems Architect & Research Director: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_PATH = Path("checkpoints/SymboLM-1.5B-v1.0-DSR1-Qwen2.5")


def run_smoke_test():
    print("=" * 70)
    print("SymboLM-1.5B-v1.0-DSR1-Qwen2.5 Standalone Smoke Verification")
    print(f"Loading merged model directly from: {MODEL_PATH}")
    print("=" * 70)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"[OK] Standalone model loaded in {load_time:.2f}s! Parameters: {model.num_parameters():,}\n")

    test_cases = [
        {
            "tier": "Δ0 Atomic Fact Bypass",
            "prompt": "What is the chemical symbol for potassium?",
            "expected_ans": "K",
            "max_tokens": 30,
        },
        {
            "tier": "Δ1 Polynomial Invariant",
            "prompt": "If x + 1/x = 6, find the value of x^2 + 1/x^2.",
            "expected_ans": "34",
            "max_tokens": 60,
        },
        {
            "tier": "Δ3 Chinese Remainder Theorem",
            "prompt": "Find the smallest positive integer x such that x = 4 (mod 7) and x = 9 (mod 11).",
            "expected_ans": "53",
            "max_tokens": 100,
        },
    ]

    for tc in test_cases:
        tier = tc["tier"]
        q = tc["prompt"]
        formatted_prompt = f"<｜User｜>{q}<｜Assistant｜><think>\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        t_gen = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=tc["max_tokens"],
                do_sample=False,  # greedy decoding for verification
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t_gen

        generated_ids = outputs[0][input_len:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
        tokens_emitted = len(generated_ids)

        print(f"[{tier}]")
        print(f"Prompt        : {q}")
        print(f"Tokens Emitted: {tokens_emitted} tokens ({elapsed:.2f}s)")
        print(f"Completion    :\n{completion.strip()}")
        print("-" * 70)


if __name__ == "__main__":
    run_smoke_test()
