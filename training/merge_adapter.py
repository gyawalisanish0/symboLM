"""
SymboLM Weight Merger
======================
Fuses the Stage 1 LoRA adapter weights directly into the base model weights
(deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) to produce a standalone, high-speed
unified model ready for inference and Stage 2 GRPO training.

Author: Sanish Gyawali
"""

import argparse
from pathlib import Path
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Merge SymboLM LoRA adapter into base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Base foundation model repository or path",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./checkpoints/sft_adapter",
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/symbolm_stage1_merged",
        help="Destination directory for fused standalone weights",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="Save as safetensors format",
    )
    return parser.parse_args()


def merge_and_save():
    args = parse_args()
    adapter_path = Path(args.adapter_path).resolve()
    output_path = Path(args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SymboLM Model Merger (Stage 1 SFT)")
    print(f"Base model   : {args.base_model}")
    print(f"Adapter path : {adapter_path}")
    print(f"Output path  : {output_path}")
    print("=" * 65)

    # 1. Load Tokenizer
    print("\n1. Loading extended tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    print(f"[OK] Tokenizer loaded! Vocab size: {len(tokenizer):,}")

    # 2. Load Base Model
    print("\n2. Loading base model in float32 / bfloat16 on CPU...")
    # On CPU, float32 is most stable for exact weight fusion
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"[OK] Base model loaded! Base vocab size: {model.config.vocab_size:,}")

    # Resize embeddings to match extended tokenizer
    if len(tokenizer) != model.config.vocab_size:
        print(f"Resizing token embeddings: {model.config.vocab_size:,} -> {len(tokenizer):,}...")
        model.resize_token_embeddings(len(tokenizer))

    # 3. Attach LoRA Adapter
    print("\n3. Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    print("[OK] LoRA adapter attached successfully!")

    # 4. Merge and Unload
    print("\n4. Fusing LoRA weights into base model layers (merge_and_unload)...")
    model = model.merge_and_unload()
    print("[OK] Weights fused successfully! Model is now a standalone CausalLM.")

    # 5. Save Merged Model & Tokenizer
    print(f"\n5. Saving standalone model to {output_path}...")
    model.save_pretrained(
        str(output_path),
        safe_serialization=args.safe_serialization,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(str(output_path))
    print("[OK] Model and tokenizer saved successfully!")

    print("\n" + "=" * 65)
    print("Merge Complete!")
    print(f"Standalone SymboLM Stage 1 model ready at: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    merge_and_save()
