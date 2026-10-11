"""
SymboLM Stage 4: Symbolic Efficiency & Hard EOS Alignment (SE-HEA)
==================================================================
Reinforcement learning via Group Relative Policy Optimization (GRPO) targeting
Information Density, Monotonic Length Economy, and Hard EOS Termination.

Eliminates:
  1. Post-proof looping repetition (e.g. '527 527 527...')
  2. Base-model web-scraping debris ('endphp', 'IconModule', HTML tags)
  3. Trailing conversational monologue after answer verification

Warm-Starts From:
  Stage 3.5 Concept GRPO Policy Adapter (checkpoints/stage3_concept_adapter)

Author: Sanish Gyawali
AI Systems Exoskeleton: Antigravity (Google DeepMind)
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from config import config
from inference.generate import load_symbo_model
from training.reward import (
    correctness_reward_func,
    syntax_and_noise_reward_func,
    concept_alignment_reward_func,
    efficiency_and_bypass_reward_func,
    hard_eos_and_economy_reward_func,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SymboLM Stage 4 Efficiency GRPO")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--warmstart_adapter", type=str, default="checkpoints/stage3_concept_adapter")
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage4_efficient_adapter")
    parser.add_argument("--data_file", type=str, default="data/stage3_unified_curriculum.jsonl")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.3)
    return parser.parse_args()


def load_efficiency_curriculum(data_file_path: Path) -> Dataset:
    """Loads unified curriculum formatted for GRPOTrainer."""
    if not data_file_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_file_path}")

    records = []
    with open(data_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item.get("prompt", "")
            ans = item.get("ground_truth", item.get("answer", ""))
            reg = item.get("register", item.get("type", "symbolic_deductive"))
            concept = item.get("concept", "")

            # DeepSeek-R1 style chat template prompt
            formatted_prompt = f"<｜User｜>{prompt}<｜Assistant｜><think>\n"
            records.append({
                "prompt": formatted_prompt,
                "answer": str(ans),
                "answers": str(ans),
                "register": str(reg),
                "registers": str(reg),
                "concept": str(concept),
                "concepts": str(concept),
            })

    print(f"[DATASET] Loaded {len(records)} curriculum samples for Stage 4 Efficiency Alignment.")
    return Dataset.from_list(records)


def main():
    args = parse_args()
    print("=" * 78)
    print("  SymboLM Stage 4: Symbolic Efficiency & Hard EOS Alignment")
    print("  Author & Principal Architect: Sanish Gyawali")
    print("  Exoskeleton: Antigravity (Google DeepMind)")
    print("=" * 78)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"Base Model    : {args.base_model}")
    print(f"Warm-Start    : {args.warmstart_adapter}")
    print(f"Output Dir    : {args.output_dir}")
    print(f"Max Comp Len  : {args.max_completion_length} tokens (forced compact proof)")
    print(f"Max Steps     : {args.max_steps}")

    # 1. Load Tokenizer & Model with warm-start adapter
    print("\n[1/4] Loading Tokenizer & Base Model with 151,712 Vocab...")
    target_vocab_size = 151712

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_sm80 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if is_sm80 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(target_vocab_size)

    # 2. Attach Warm-Start Adapter
    print(f"[2/4] Attaching Warm-Start LoRA Adapter from {args.warmstart_adapter}...")
    adapter_path = Path(args.warmstart_adapter)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        print("[OK] Warm-start adapter successfully attached as trainable LoRA policy!")
    else:
        raise FileNotFoundError(f"CRITICAL: Warm-start adapter not found at: {adapter_path}")

    # 3. Load Curriculum
    print("\n[3/4] Preparing Curriculum Dataset...")
    train_ds = load_efficiency_curriculum(Path(args.data_file))

    # 4. GRPOTrainer Configuration
    print("\n[4/4] Initializing Stage 4 GRPOTrainer with Hard EOS Reward Engine...")
    sig_params = inspect.signature(GRPOConfig.__init__).parameters
    grpo_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_generations": args.num_generations,
        "max_prompt_length": 256,
        "max_completion_length": args.max_completion_length,
        "max_steps": args.max_steps,
        "logging_steps": 5,
        "save_strategy": "no",
        "beta": args.beta,
        "temperature": args.temperature,
        "report_to": "none",
        "use_vllm": False,
        "bf16": is_sm80,
        "fp16": torch.cuda.is_available() and not is_sm80,
    }
    filtered_kwargs = {k: v for k, v in grpo_kwargs.items() if k in sig_params}
    training_args = GRPOConfig(**filtered_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,
            syntax_and_noise_reward_func,
            concept_alignment_reward_func,
            efficiency_and_bypass_reward_func,
            hard_eos_and_economy_reward_func,  # Stage 4 Key Reward
        ],
        args=training_args,
        train_dataset=train_ds,
    )

    print("\n[TRAINING] Starting Stage 4 Efficiency GRPO Optimization...")
    trainer.train()

    # 5. Persist Stage 4 Adapter
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[SAVE] Saving Stage 4 aligned adapter to: {out_dir}...")
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[COMPLETE] Stage 4 Efficiency Alignment finished successfully! Adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
