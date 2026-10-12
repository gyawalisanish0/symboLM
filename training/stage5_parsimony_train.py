"""
SymboLM Stage 5: Scratchpad 2.0 & Unbounded Parsimony GRPO Training
===================================================================
Reinforcement learning via Group Relative Policy Optimization (GRPO) targeting:
  1. Unbounded Parsimony Gradient: No artificial token ceilings; rewards minimal description length.
  2. Scratchpad 2.0: Elastic cognitive depth allocation across the Cognitive Depth Spectrum (Δ0 to Δ3).
  3. Latent Knowledge Activation: Incentivizes the model to leverage distilled internal knowledge.

Warm-Starts From:
  Stage 4 Efficiency Aligned Policy Adapter (checkpoints/stage4_efficient_adapter)

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
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from config import config
from training.reward import (
    unbounded_parsimony_reward_func,
    syntax_and_noise_reward_func,
    concept_alignment_reward_func,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SymboLM Stage 5 Parsimony GRPO")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--warmstart_adapter", type=str, default="checkpoints/stage4_efficient_adapter")
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage5_parsimony_adapter")
    parser.add_argument("--data_file", type=str, default="data/stage5_parsimony_curriculum.jsonl")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--max_completion_length", type=int, default=192, help="Elastic budget with zero ceiling lock")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.3)
    return parser.parse_args()


def load_parsimony_curriculum(data_file_path: Path) -> Dataset:
    """Loads Stage 5 stratified curriculum formatted for GRPOTrainer."""
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
            reg = item.get("register", "symbolic_deductive")
            concept = item.get("concept", "")
            depth = item.get("cognitive_depth", "Δ1")

            # Formatted chat template prompt
            formatted_prompt = f"<｜User｜>{prompt}<｜Assistant｜><think>\n"
            records.append({
                "prompt": formatted_prompt,
                "answer": str(ans),
                "answers": str(ans),
                "register": str(reg),
                "registers": str(reg),
                "concept": str(concept),
                "concepts": str(concept),
                "cognitive_depth": str(depth),
                "depth": str(depth),
            })

    print(f"[DATASET] Loaded {len(records)} samples for Stage 5 Parsimony & Scratchpad 2.0 Training.")
    return Dataset.from_list(records)


def main():
    args = parse_args()
    print("=" * 78)
    print("  SymboLM Stage 5: Scratchpad 2.0 & Unbounded Parsimony GRPO Training")
    print("  Author & Principal Architect: Sanish Gyawali")
    print("  AI Systems Exoskeleton: Antigravity (Google DeepMind)")
    print("=" * 78)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"Base Model    : {args.base_model}")
    print(f"Warm-Start    : {args.warmstart_adapter}")
    print(f"Output Dir    : {args.output_dir}")
    print(f"Max Comp Len  : {args.max_completion_length} tokens (Elastic budget, zero ceiling lock)")
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

    # 2. Attach Warm-Start Adapter from Stage 4
    print(f"[2/4] Attaching Warm-Start LoRA Adapter from {args.warmstart_adapter}...")
    adapter_path = Path(args.warmstart_adapter)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        print("[OK] Stage 4 warm-start adapter attached as trainable policy!")
    else:
        raise FileNotFoundError(f"CRITICAL: Warm-start adapter not found at: {adapter_path}")

    # 3. Load Curriculum
    print("\n[3/4] Preparing Curriculum Dataset across Cognitive Depth Spectrum (Δ0 to Δ3)...")
    train_ds = load_parsimony_curriculum(Path(args.data_file))

    # 4. GRPOTrainer Configuration
    print("\n[4/4] Initializing Stage 5 GRPOTrainer with Unbounded Parsimony Reward Engine...")
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
            unbounded_parsimony_reward_func,  # Stage 5 Core Parsimony Gradient
            syntax_and_noise_reward_func,     # Web debris & repeat guard
            concept_alignment_reward_func,    # Theorem & invariant bonus
        ],
        args=training_args,
        train_dataset=train_ds,
    )

    print("\n[TRAINING] Starting Stage 5 Parsimony GRPO Optimization...")
    trainer.train()

    # 5. Persist Stage 5 Adapter
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[SAVE] Saving Stage 5 Parsimony-aligned adapter to: {out_dir}...")
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[COMPLETE] Stage 5 Training finished successfully! Adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
