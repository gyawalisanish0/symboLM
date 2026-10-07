"""
SymboLM Efficiency & Speed Benchmark
=====================================
Measures token counts, wall-clock generation latency, tokens/second,
and reasoning compression ratio between English CoT vs SymboLM DSL.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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
    tok_dir = Path(config.tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_dir if tok_dir.exists() else args.base_model,
        trust_remote_code=True,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    if args.adapter_path and Path(args.adapter_path).exists():
        model = PeftModel.from_pretrained(model, args.adapter_path)
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

        prompt = (
            f"<|im_start|>system\n{config.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prob}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.0,
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
