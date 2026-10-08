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
    adapter_p = Path(adapter_path) if adapter_path else None

    # Check if adapter_path is a standalone merged model directory
    is_merged_model = (
        adapter_p
        and adapter_p.exists()
        and (adapter_p / "config.json").exists()
        and not (adapter_p / "adapter_config.json").exists()
    )

    if is_merged_model:
        print(f"Loading standalone merged SymboLM model from: {adapter_p}")
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_p), trust_remote_code=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            str(adapter_p),
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        return model, tokenizer

    # Otherwise load base model + optional LoRA adapter
    tok_dir = Path(config.tokenizer_dir)
    tok_source = (
        str(adapter_p)
        if (adapter_p and (adapter_p / "tokenizer_config.json").exists())
        else (str(tok_dir) if tok_dir.exists() else base_model)
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)

    if torch.cuda.is_available():
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
    else:
        print("[CPU Mode] Loading base model in float32 on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    model.resize_token_embeddings(len(tokenizer))

    if adapter_p and adapter_p.exists():
        print(f"Loading adapter: {adapter_p}")
        model = PeftModel.from_pretrained(model, str(adapter_p))
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
    messages = [
        {"role": "user", "content": question},
    ]
    if getattr(config, "system_prompt", None):
        messages.insert(0, {"role": "system", "content": config.system_prompt})

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        prompt = f"<｜User｜>{question}<｜Assistant｜><think>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

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
