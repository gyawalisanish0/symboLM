"""
SymboLM Interactive Inference & Generation
==========================================
Generate symbolic reasoning traces and solutions with KV-caching.
Allows interactive console chat or single-prompt execution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)

from config import config
from symbolic.grammar import extract_answer


def parse_args():
    parser = argparse.ArgumentParser(description="SymboLM Inference Engine")
    parser.add_argument("--base_model", type=str, default=config.base_model_name)
    parser.add_argument("--adapter_path", type=str, default=str(config.grpo_output_dir))
    parser.add_argument("--prompt", type=str, default=None, help="Single query to solve")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def load_symbo_model(base_model: str, adapter_path: str):
    tok_dir = Path(config.tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_dir if tok_dir.exists() else base_model,
        trust_remote_code=True,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    if adapter_path and Path(adapter_path).exists():
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print(f"Note: Running base model without adapter ({base_model})")

    model.eval()
    return model, tokenizer


def generate_solution(
    model,
    tokenizer,
    question: str,
    temperature: float = 0.2,
    max_new_tokens: int = 256,
    stream: bool = True,
):
    prompt = (
        f"<|im_start|>system\n{config.system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    full_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    answer = extract_answer(full_text)
    return full_text, answer


def main():
    args = parse_args()
    model, tokenizer = load_symbo_model(args.base_model, args.adapter_path)

    if args.prompt:
        print(f"\nProblem: {args.prompt}")
        print("\n[SymboLM Generation]")
        gen_text, ans = generate_solution(
            model, tokenizer, args.prompt, temperature=args.temperature, max_new_tokens=args.max_new_tokens
        )
        print(f"\nExtracted Answer: {ans}")
        return

    print("\n" + "=" * 60)
    print("SymboLM Interactive Console (Type 'exit' to quit)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nEnter Question > ").strip()
            if not user_input or user_input.lower() in ("exit", "quit"):
                break
            print("\n[Thinking & Solving...]")
            _, ans = generate_solution(
                model, tokenizer, user_input, temperature=args.temperature, max_new_tokens=args.max_new_tokens
            )
            print(f"\nExtracted Answer: {ans}")
        except KeyboardInterrupt:
            break

    print("\nExiting SymboLM console.")


if __name__ == "__main__":
    main()
