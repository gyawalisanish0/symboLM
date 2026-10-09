"""
SymboLM Stage 2: DeepSeek R1-Style GRPO Reinforcement Learning
==============================================================
Reinforcement learning via Group Relative Policy Optimization (GRPO).
Generates G=4 reasoning traces per prompt, normalizes rewards within the group,
and reinforces compact symbolic reasoning without requiring a critic model.
Optimized for Google Colab TPU v5e (bfloat16) and CUDA GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import config
from symbolic.grammar import extract_answer, validate_trace, _estimate_tokens
from training.reward import (
    answers_match,
    compute_reward,
    correctness_reward_func,
    syntax_and_noise_reward_func,
    self_correction_reward_func,
    efficiency_and_bypass_reward_func,
)

HAS_TPU = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_TPU = True
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="SymboLM GRPO Stage (TPU & GPU)")
    parser.add_argument("--base_model", type=str, default=config.base_model_name)
    parser.add_argument("--sft_adapter", type=str, default=str(config.sft_output_dir))
    parser.add_argument("--output_dir", type=str, default=str(config.grpo_output_dir))
    parser.add_argument("--data_dir", type=str, default=str(config.data_dir))
    parser.add_argument("--drive_backup", type=str, default=config.colab_drive_checkpoint)
    parser.add_argument("--num_generations", type=int, default=config.grpo_num_generations)
    parser.add_argument("--learning_rate", type=float, default=config.grpo_lr)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--tpu", action="store_true", help="Force TPU execution via PyTorch/XLA")
    return parser.parse_args()


def load_prompts_dataset(data_dir: Path) -> Dataset:
    """Load queries, ground-truth answers, and cognitive register for GRPO sampling."""
    curriculum_file = data_dir / "grpo_curriculum.jsonl"
    train_file = data_dir / "train.jsonl"

    data_file = curriculum_file if curriculum_file.exists() else train_file
    print(f"Loading GRPO training prompts from: {data_file}")

    prompts = []
    answers = []
    registers = []

    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # Extract user prompt based on schema
            if "prompt" in item:
                user_text = item["prompt"]
            elif "messages" in item and len(item["messages"]) > 1:
                user_text = item["messages"][1]["content"]
            else:
                continue

            query = f"<｜User｜>{user_text}<｜Assistant｜><think>\n"
            ans = str(item.get("answer", ""))
            reg = item.get("register", "symbolic_deductive")

            prompts.append(query)
            answers.append(ans)
            registers.append(reg)

    return Dataset.from_dict({
        "prompt": prompts,
        "answers": answers,
        "registers": registers,
    })


def main():
    args = parse_args()
    use_tpu = args.tpu or (HAS_TPU and os.environ.get("PJRT_DEVICE") == "TPU")

    print("=" * 65)
    print("SymboLM GRPO Reinforcement Learning (Stage 2: R1 Optimization)")
    print(f"Base Model  : {args.base_model}")
    print(f"Backend     : {'Google TPU v5e (PyTorch/XLA)' if use_tpu else 'NVIDIA CUDA GPU / CPU'}")
    print(f"SFT Adapter : {args.sft_adapter}")
    print(f"Output Dir  : {args.output_dir}")
    print("=" * 65)

    # 1. Load Tokenizer
    tok_dir = Path(config.tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_dir if tok_dir.exists() else args.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"[CUDA] Loading base model in native {dtype} (unquantized, ~3.0 GB)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer))
    elif use_tpu:
        print("[TPU v5e] Loading model in native bfloat16...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer))
    else:
        print("[CPU] Loading base model in float32...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer))

    # Attach SFT LoRA weights if present
    adapter_cfg = Path(args.sft_adapter) / "adapter_config.json"
    if os.path.exists(args.sft_adapter) and adapter_cfg.exists():
        print(f"Attaching SFT LoRA weights from {args.sft_adapter}...")
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    else:
        print(f"[Warning] No valid SFT adapter found at {args.sft_adapter} (missing adapter_config.json). Training GRPO directly on base model.")

    # 3. Load dataset
    raw_ds = load_prompts_dataset(Path(args.data_dir))
    print(f"Loaded {len(raw_ds)} prompts for GRPO training.")

    # 4. GRPO Configuration
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=config.grpo_batch_size,
        gradient_accumulation_steps=config.grpo_grad_accum_steps,
        num_generations=args.num_generations,
        max_prompt_length=config.grpo_max_prompt_len,
        max_completion_length=config.grpo_max_completion_len,
        max_steps=args.max_steps,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        beta=config.grpo_beta,
        temperature=config.grpo_temperature,
        report_to="none",
        use_vllm=False,
        bf16=use_tpu or torch.cuda.is_bf16_supported(),
        fp16=not use_tpu and not torch.cuda.is_bf16_supported(),
    )

    # 5. Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,
            syntax_and_noise_reward_func,
            self_correction_reward_func,
            efficiency_and_bypass_reward_func,
        ],
        args=training_args,
        train_dataset=raw_ds,
    )

    print("Beginning GRPO training...")
    trainer.train()

    print(f"Saving GRPO checkpoint to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Drive sync
    drive_path = Path(args.drive_backup) / "grpo_adapter"
    if Path(args.drive_backup).parent.exists():
        print(f"Syncing GRPO adapter to Google Drive: {drive_path}...")
        shutil.copytree(args.output_dir, drive_path, dirs_exist_ok=True)

    print("Stage 2 GRPO completed!")


if __name__ == "__main__":
    main()
