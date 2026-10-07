"""
SymboLM Stage 1: Supervised Fine-Tuning (SFT)
=============================================
Fine-tunes DeepSeek-R1-Distill-Qwen-1.5B on symbolic reasoning traces.
Supports both Google Colab TPU v5e (PyTorch/XLA in native bfloat16)
and NVIDIA CUDA GPUs (T4 in fp16/4-bit).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from config import config
from data.symbolic_dataset import SymbolicDataCollator, SymbolicSFTDataset
from symbolic.tokenizer_ext import extend_tokenizer, initialize_symbol_embeddings

# Check for TPU environment
HAS_TPU = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    torch.xla = torch_xla
    HAS_TPU = True
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="SymboLM SFT Stage (TPU & GPU)")
    parser.add_argument("--base_model", type=str, default=config.base_model_name)
    parser.add_argument("--data_dir", type=str, default=str(config.data_dir))
    parser.add_argument("--output_dir", type=str, default=str(config.sft_output_dir))
    parser.add_argument("--epochs", type=int, default=config.sft_epochs)
    parser.add_argument("--batch_size", type=int, default=config.sft_batch_size)
    parser.add_argument("--grad_accum", type=int, default=config.sft_grad_accum_steps)
    parser.add_argument("--lr", type=float, default=config.sft_lr)
    parser.add_argument("--tpu", action="store_true", help="Force TPU v5e execution via PyTorch/XLA")
    parser.add_argument("--full_finetune", action="store_true", help="Full fine-tuning instead of LoRA (TPU)")
    parser.add_argument("--drive_backup", type=str, default=config.colab_drive_checkpoint)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def setup_drive_sync(drive_dir: str):
    """Verifies Google Drive backup directory."""
    drive_path = Path(drive_dir)
    if drive_path.parent.exists():
        drive_path.mkdir(parents=True, exist_ok=True)
        print(f"[Drive Sync] Enabled backup to: {drive_path}")
        return True
    return False


def main():
    args = parse_args()
    use_tpu = args.tpu or (HAS_TPU and os.environ.get("PJRT_DEVICE") == "TPU")

    print("=" * 65)
    print("SymboLM SFT Training (Stage 1: Symbolic Grammar Grounding)")
    print(f"Base model : {args.base_model}")
    print(f"Backend    : {'Google TPU v5e (PyTorch/XLA)' if use_tpu else 'NVIDIA CUDA GPU / CPU'}")
    print(f"Precision  : {'Native bfloat16' if use_tpu else 'float16'}")
    print(f"Mode       : {'Full Fine-Tuning' if args.full_finetune else 'LoRA Adapter'}")
    print(f"Output dir : {args.output_dir}")
    print("=" * 65)

    # 1. Tokenizer Extension
    tok_dir = Path(config.tokenizer_dir)
    if not tok_dir.exists() or not (tok_dir / "tokenizer_config.json").exists():
        print("Extending base tokenizer with symbolic operators...")
        tokenizer = extend_tokenizer(args.base_model, tok_dir)
    else:
        print(f"Loading pre-extended tokenizer from {tok_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Model Initialization
    if use_tpu:
        # -------------------------------------------------------------
        # TPU v5e Native Path: Pure bfloat16 (No bitsandbytes/CUDA dependencies)
        # -------------------------------------------------------------
        print("[TPU v5e] Loading model in native bfloat16 for PyTorch/XLA...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # Do NOT use device_map="auto" on TPU; HF Trainer places onto XLA device
        )
        model.resize_token_embeddings(len(tokenizer))
        initialize_symbol_embeddings(model, tokenizer)

        if not args.full_finetune:
            print("[TPU v5e] Attaching bfloat16 LoRA adapters...")
            lora_cfg = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
    else:
        # -------------------------------------------------------------
        # GPU / CUDA Fallback Path: 4-bit QLoRA / fp16
        # -------------------------------------------------------------
        print("[CUDA] Initializing model with 4-bit NF4 quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer))
        initialize_symbol_embeddings(model, tokenizer)
        model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # 3. Datasets
    train_path = Path(args.data_dir) / "train.jsonl"
    val_path = Path(args.data_dir) / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {train_path}. Run 'python -m data.build_dataset' first!"
        )

    print(f"Loading dataset from {train_path}...")
    train_dataset = SymbolicSFTDataset(train_path, tokenizer, max_length=config.sft_max_seq_len)
    val_dataset = (
        SymbolicSFTDataset(val_path, tokenizer, max_length=config.sft_max_seq_len)
        if val_path.exists()
        else None
    )
    collator = SymbolicDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=384 if use_tpu else None,
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=100 if val_dataset else None,
        bf16=use_tpu or (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=not use_tpu and torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=not use_tpu,
        report_to="none",
        optim="adamw_torch" if use_tpu else "paged_adamw_8bit",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # 5. Execute Training
    print("Starting SFT training loop...")
    resume_path = args.resume_from_checkpoint
    if resume_path is None and os.path.exists(args.output_dir):
        checkpoints = list(Path(args.output_dir).glob("checkpoint-*"))
        if checkpoints:
            latest = max(checkpoints, key=os.path.getmtime)
            print(f"[Auto-Resume] Found previous checkpoint: {latest}")
            resume_path = str(latest)

    trainer.train(resume_from_checkpoint=resume_path)

    # 6. Save Adapter & Final Checkpoints
    print(f"Saving final trained model/adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Backup to Drive if available
    drive_ok = setup_drive_sync(args.drive_backup)
    if drive_ok:
        drive_dest = Path(args.drive_backup) / "sft_adapter"
        print(f"Copying artifacts to Google Drive: {drive_dest}...")
        shutil.copytree(args.output_dir, drive_dest, dirs_exist_ok=True)
        print("Drive backup complete!")

    print("Stage 1 SFT finished successfully!")


if __name__ == "__main__":
    main()
