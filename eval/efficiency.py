"""
SymboLM Efficiency & Speed Benchmark
=====================================
Measures token counts, wall-clock generation latency, tokens/second,
and reasoning compression ratio between English CoT vs SymboLM DSL.
"""

from __future__ import annotations

import argparse
import sys
import time
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

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import config
from symbolic.grammar import EXAMPLE_TRACES


def parse_args():
    parser = argparse.ArgumentParser(description="SymboLM Speed & Efficiency Audit")
    parser.add_argument("--base_model", type=str, default=config.base_model_name)
    parser.add_argument("--adapter_path", type=str, default=str(config.grpo_output_dir))
    return parser.parse_args()


def benchmark_efficiency(args):
    from inference.generate import load_symbo_model
    model, tokenizer = load_symbo_model(args.base_model, args.adapter_path)
    model.eval()

    print("\n" + "=" * 70)
    print("SymboLM Efficiency & Speed Audit")
    print("=" * 70)

    total_eng_tokens = 0
    total_sym_tokens = 0
    total_latency = 0.0

    for i, ex in enumerate(EXAMPLE_TRACES):
        prob = ex["problem"]
        eng_cot = ex["english_cot"]
        sym_cot = ex["symbolic_cot"]

        eng_token_ids = tokenizer.encode(eng_cot, add_special_tokens=False)
        sym_token_ids = tokenizer.encode(sym_cot, add_special_tokens=False)

        messages = [{"role": "user", "content": prob}]
        if getattr(config, "system_prompt", None):
            messages.insert(0, {"role": "system", "content": config.system_prompt})

        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = f"<｜User｜>{prob}<｜Assistant｜><think>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        gen_tokens = len(outputs[0]) - inputs.input_ids.shape[1]
        elapsed = t1 - t0
        speed = gen_tokens / elapsed if elapsed > 0 else 0

        total_eng_tokens += len(eng_token_ids)
        total_sym_tokens += len(sym_token_ids)
        total_latency += elapsed

        saved_pct = (1.0 - len(sym_token_ids) / len(eng_token_ids)) * 100.0

        print(f"\n[Case {i+1}] {prob[:60]}...")
        print(f"  English tokens  : {len(eng_token_ids)}")
        print(f"  Symbolic tokens : {len(sym_token_ids)} ({saved_pct:.1f}% reduction)")
        print(f"  Latency         : {elapsed*1000:.1f} ms ({speed:.1f} tokens/s)")

    avg_savings = (1.0 - total_sym_tokens / total_eng_tokens) * 100.0
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS:")
    print(f"  Total English Tokens   : {total_eng_tokens}")
    print(f"  Total Symbolic Tokens  : {total_sym_tokens}")
    print(f"  Average Token Savings  : {avg_savings:.1f}%")
    print(f"  Total Generation Time  : {total_latency:.2f} s")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_efficiency(parse_args())
