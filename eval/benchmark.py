"""
SymboLM Benchmark Evaluation
=============================
Evaluates model pass@1 accuracy on GSM8K and MATH benchmarks.
Tests baseline model vs SymboLM SFT vs SymboLM GRPO.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import config
from data.verify_answers import verify_answer
from symbolic.grammar import extract_answer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SymboLM Accuracy")
    parser.add_argument("--base_model", type=str, default=config.base_model_name)
    parser.add_argument("--adapter_path", type=str, default=str(config.grpo_output_dir))
    parser.add_argument("--test_file", type=str, default=str(config.data_dir / "test.jsonl"))
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_file", type=str, default="./eval/eval_results.json")
    return parser.parse_args()


def load_model_and_tokenizer(base_model: str, adapter_path: str):
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
        print(f"Loading adapter weights from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def run_eval(args):
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path)

    records: List[Dict] = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    test_samples = records[: args.max_samples]
    print(f"Evaluating {len(test_samples)} problems...")

    correct = 0
    total = 0
    detailed_results = []

    for item in tqdm(test_samples, desc="Benchmarking"):
        question = item["messages"][1]["content"]
        ground_truth = item["answer"]

        prompt = (
            f"<|im_start|>system\n{config.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,  # Greedy for pass@1 benchmark
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        extracted = extract_answer(gen_text)
        is_match = verify_answer(extracted, ground_truth)

        if is_match:
            correct += 1
        total += 1

        detailed_results.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": extracted,
            "generation": gen_text,
            "correct": is_match,
        })

    accuracy = (correct / total) * 100.0 if total > 0 else 0.0
    print("\n" + "=" * 50)
    print(f"Evaluation Complete!")
    print(f"Accuracy : {accuracy:.2f}% ({correct}/{total})")
    print("=" * 50)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "total": total, "results": detailed_results}, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    run_eval(parse_args())
