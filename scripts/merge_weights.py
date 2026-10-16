"""
SymboLM Standalone Weight Merger
=================================
Fuses the Stage 5 Parsimony LoRA adapter weights directly into the base foundation
weights (deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) to produce a standalone, unified,
1-click deployable model: SymboLM-1.5B-v1.0-DSR1-Qwen2.5.

Principal Systems Architect & Research Director: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import argparse
from pathlib import Path
import shutil
import sys
import time
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
        "--model_type",
        type=str,
        default="deepseek",
        choices=["deepseek", "smollm", "custom"],
        help="Model preset to merge",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base foundation model repository or path",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Destination directory for fused standalone weights",
    )
    parser.add_argument(
        "--target_vocab",
        type=int,
        default=None,
        help="Target vocabulary size for embedding matrix (auto-detected if None)",
    )
    return parser.parse_args()


def merge_and_export():
    args = parse_args()

    # Apply presets
    if args.model_type == "deepseek":
        base_model = args.base_model or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        adapter_path = Path(args.adapter_path or "checkpoints/symbolm_v2_adapter").resolve()
        output_path = Path(args.output_dir or "checkpoints/SymboLM-2.1-DeepSeek-1.5B-Merged").resolve()
        target_vocab = args.target_vocab or 151712
    elif args.model_type == "smollm":
        base_model = args.base_model or "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        adapter_path = Path(args.adapter_path or "eval/cross_family_llama/symbolm_llama_1_7b_adapter").resolve()
        output_path = Path(args.output_dir or "checkpoints/SymboLM-2.1-SmolLM2-1.7B-Merged").resolve()
        target_vocab = args.target_vocab or 49152
    else:
        assert args.base_model, "--base_model is required for custom model_type"
        assert args.adapter_path, "--adapter_path is required for custom model_type"
        assert args.output_dir, "--output_dir is required for custom model_type"
        base_model = args.base_model
        adapter_path = Path(args.adapter_path).resolve()
        output_path = Path(args.output_dir).resolve()
        target_vocab = args.target_vocab

    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("SymboLM Model Merger — Official Standalone Release Pipeline")
    print(f"Preset       : {args.model_type}")
    print(f"Author       : Sanish Gyawali (Rupandehi, Nepal)")
    print(f"Base Model   : {base_model}")
    print(f"Adapter Path : {adapter_path}")
    print(f"Output Path  : {output_path}")
    print("=" * 75)

    start_time = time.time()

    # 1. Load Tokenizer
    print("\n[1/6] Loading canonical tokenizer from adapter...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[OK] Tokenizer loaded. Vocab size: {len(tokenizer):,}")

    # 2. Load Base Model in bfloat16
    print("\n[2/6] Loading base model in torch.bfloat16 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"[OK] Base model loaded. Base vocab size: {model.config.vocab_size:,}")

    # 3. Resize Embeddings to match target vocabulary structure if needed
    effective_target_vocab = target_vocab or len(tokenizer)
    if model.config.vocab_size != effective_target_vocab:
        print(f"Resizing token embeddings: {model.config.vocab_size:,} -> {effective_target_vocab:,}...")
        model.resize_token_embeddings(effective_target_vocab)
        print(f"[OK] Token embeddings resized to {effective_target_vocab:,}")

    # 4. Attach LoRA Adapter
    print(f"\n[3/6] Attaching LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    print("[OK] LoRA adapter attached successfully!")

    # 5. Merge and Unload
    print("\n[4/6] Fusing LoRA weights into base model layers (merge_and_unload)...")
    fuse_start = time.time()
    model = model.merge_and_unload()
    fuse_time = time.time() - fuse_start
    print(f"[OK] Weights fused successfully in {fuse_time:.2f}s! Model is now a standalone CausalLM.")

    # 6. Save Merged Model & Tokenizer
    print(f"\n[5/6] Saving standalone safetensors weights to {output_path}...")
    save_start = time.time()
    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(str(output_path))
    save_time = time.time() - save_start
    print(f"[OK] Saved model and tokenizer in {save_time:.2f}s!")

    # 7. Copy Chat Template & GBNF if present
    print("\n[6/6] Packaging chat template and release metadata...")
    chat_template_src = adapter_path / "chat_template.jinja"
    if chat_template_src.exists():
        shutil.copy2(chat_template_src, output_path / "chat_template.jinja")
        print("[OK] Packaged chat_template.jinja.")
    gbnf_src = Path("symbolic/symbolm.gbnf")
    if gbnf_src.exists():
        shutil.copy2(gbnf_src, output_path / "symbolm.gbnf")
        print("[OK] Packaged symbolm.gbnf.")

    total_time = time.time() - start_time
    total_size_bytes = sum(f.stat().st_size for f in output_path.glob("*") if f.is_file())
    total_size_mb = total_size_bytes / (1024 * 1024)

    print("\n" + "=" * 75)
    print(f"STANDALONE MERGE COMPLETE! ({total_time:.2f}s elapsed)")
    print(f"Output Path : {output_path}")
    print(f"Total Size  : {total_size_mb:.2f} MB ({total_size_mb / 1024:.2f} GB)")
    print("Files in package:")
    for f in sorted(output_path.glob("*")):
        if f.is_file():
            print(f"  • {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 75)


if __name__ == "__main__":
    merge_and_export()
