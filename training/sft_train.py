"""
SymboLM Stage 1: Supervised Fine-Tuning (SFT)
=============================================
Fine-tunes DeepSeek-R1-Distill-Qwen-1.5B on symbolic reasoning traces.
Supports both Google Colab TPU v5e (PyTorch/XLA in native bfloat16)
and NVIDIA CUDA GPUs (T4 in fp16/4-bit).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from config import config
from data.symbolic_dataset import SymbolicDataCollator, SymbolicSFTDataset
from symbolic.prompts import contains_loop_marker
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

# Ensure PyTorch 2.6+ unpickler allows numpy structures in rng_state.pth
try:
    import numpy as np
    import numpy._core.multiarray as ma
    torch.serialization.add_safe_globals([ma._reconstruct, np.ndarray, np.dtype])
except Exception:
    pass

_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.project_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def package_version(name: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def write_run_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="SymboLM SFT Stage (TPU & GPU)")
    parser.add_argument("--base_model", type=str, default=config.base_model_name)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(config.project_root / "data" / "research_sft"),
        help="Directory containing canonical train.jsonl / val.jsonl",
    )
    parser.add_argument("--output_dir", type=str, default=str(config.sft_output_dir))
    parser.add_argument("--epochs", type=int, default=config.sft_epochs)
    parser.add_argument("--batch_size", type=int, default=config.sft_batch_size)
    parser.add_argument("--grad_accum", type=int, default=config.sft_grad_accum_steps)
    parser.add_argument("--lr", type=float, default=config.sft_lr)
    parser.add_argument("--tpu", action="store_true", help="Force TPU v5e execution via PyTorch/XLA")
    parser.add_argument("--full_finetune", action="store_true", help="Full fine-tuning instead of LoRA (TPU)")
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization on GPU (use unquantized fp16/bf16 LoRA for seamless TPU checkpoint resume)")
    parser.add_argument("--train_file", type=str, default=None, help="Explicit path to training JSONL file")
    parser.add_argument("--val_file", type=str, default=None, help="Explicit path to validation JSONL file")
    parser.add_argument("--drive_backup", type=str, default=config.colab_drive_checkpoint)
    parser.add_argument(
        "--no_drive_backup",
        action="store_true",
        help="Disable the Google Drive checkpoint mirror (use on Kaggle).",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Resume the newest checkpoint-* in output_dir. Off by default so a stale "
             "checkpoint cannot be silently repackaged after a data failure.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument(
        "--generation_check",
        type=int,
        default=2,
        help="Generate N validation prompts after training to check the output contract (0=off).",
    )
    parser.add_argument(
        "--strict_generation_check",
        action="store_true",
        help="Fail the run if the post-training generation check does not emit a terminal answer.",
    )
    parser.add_argument(
        "--run_manifest",
        type=str,
        default=None,
        help="Path for run_manifest.json (default: output_dir/run_manifest.json).",
    )
    return parser.parse_args()


def setup_drive_sync(drive_dir: str):
    """Verifies Google Drive backup directory."""
    drive_path = Path(drive_dir)
    if drive_path.parent.exists():
        drive_path.mkdir(parents=True, exist_ok=True)
        print(f"[Drive Sync] Enabled backup to: {drive_path}")
        return True
    return False


def run_generation_check(
    model,
    tokenizer,
    dataset: SymbolicSFTDataset,
    n: int,
    max_new_tokens: int = 192,
) -> list[Dict[str, Any]]:
    """Generate a few validation prompts and check the terminal-answer contract."""
    results: list[Dict[str, Any]] = []
    if n <= 0 or dataset is None or len(dataset) == 0:
        return results

    model.eval()
    for idx in range(min(n, len(dataset))):
        prompt = dataset.get_prompt(idx)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=8,
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False
        )
        terminal = ("</ans>" in generated) or ("</reg:bypass>" in generated)
        loop_marker = contains_loop_marker(generated)
        record = {
            "idx": idx,
            "id": dataset.records[idx].get("id"),
            "terminal_answer": terminal,
            # The EOS must be the last thing generated.  The old run never
            # learned the real EOS and could "finish" only by hitting the token
            # budget; that is a failure, not a completed answer.
            "eos_emitted": bool(
                tokenizer.eos_token
                and generated.rstrip().endswith(tokenizer.eos_token)
            ),
            "loop_marker": loop_marker,
            "generated": generated,
        }
        results.append(record)
        print(f"[GenerationCheck] idx={idx} id={record['id']} terminal={terminal} "
              f"eos={record['eos_emitted']} loop={loop_marker}")
        print(generated[:1200].replace("\n", "\\n"))
    return results


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
        if not args.no_quant:
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
        else:
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            print(f"[CUDA] Initializing model in native {dtype} (unquantized) for seamless checkpoint compatibility...")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            model.resize_token_embeddings(len(tokenizer))
            initialize_symbol_embeddings(model, tokenizer)

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
    if args.train_file:
        train_path = Path(args.train_file)
    else:
        train_path = Path(args.data_dir) / "train.jsonl"

    if args.val_file:
        val_path = Path(args.val_file)
    else:
        val_path = Path(args.data_dir) / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {train_path}. Build the canonical split first:\n"
            f"  python scripts/prepare_research_sft.py\n"
            f"Then run this script with --train_file {train_path}"
        )

    print(f"Loading dataset from {train_path}...")
    train_dataset = SymbolicSFTDataset(
        train_path, tokenizer, max_length=config.sft_max_seq_len, strict_format=True
    )
    if args.max_train_samples > 0:
        train_dataset.records = train_dataset.records[: args.max_train_samples]
    val_dataset = (
        SymbolicSFTDataset(
            val_path, tokenizer, max_length=config.sft_max_seq_len, strict_format=True
        )
        if val_path.exists()
        else None
    )
    if val_dataset is not None and args.max_val_samples > 0:
        val_dataset.records = val_dataset.records[: args.max_val_samples]

    print(f"[Dataset] train: {train_dataset.summary()}")
    if val_dataset is not None:
        print(f"[Dataset] val  : {val_dataset.summary()}")
    else:
        print("[Dataset] val  : none (training will not run evaluation)")

    train_ids = {str(r.get("id")) for r in train_dataset.records}
    if val_dataset is not None:
        overlap = train_ids & {str(r.get("id")) for r in val_dataset.records}
        if overlap:
            raise ValueError(
                f"Train/val ID overlap ({len(overlap)} IDs). Rebuild the split with "
                "scripts/prepare_research_sft.py; do not evaluate on training problems."
            )

    collator = SymbolicDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=256 if use_tpu else None,
    )

    # Make the generation contract explicit.  The old run left generation EOS at
    # the model default, which was not the tokenizer's real EOS for this adapter.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
    if getattr(model, "config", None) is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    run_manifest_path = (
        Path(args.run_manifest)
        if args.run_manifest
        else Path(args.output_dir) / "run_manifest.json"
    )
    manifest: Dict[str, Any] = {
        "status": "started",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "base_model": args.base_model,
        "backend": "tpu" if use_tpu else "cuda/cpu",
        "args": vars(args),
        "train_file": str(train_path),
        "train_sha256": sha256_file(train_path),
        "train_summary": train_dataset.summary(),
        "val_file": str(val_path) if val_dataset is not None else None,
        "val_sha256": sha256_file(val_path) if val_dataset is not None else None,
        "val_summary": val_dataset.summary() if val_dataset is not None else None,
        "tokenizer": {
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
        },
    }
    write_run_manifest(run_manifest_path, manifest)
    print(f"[Manifest] Wrote {run_manifest_path}")

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        seed=args.seed,
        data_seed=args.seed,
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
        prediction_loss_only=True,
        eval_accumulation_steps=1,
        bf16=use_tpu or (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=not use_tpu and torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
        optim="adamw_torch" if (use_tpu or args.no_quant) else "paged_adamw_8bit",
        dataloader_num_workers=0,
    )

    class HostRAMCleanupCallback(TrainerCallback):
        """Periodically runs garbage collection to eliminate host memory accumulation on Colab and syncs checkpoints to Drive."""
        def __init__(self, drive_backup=None):
            self.drive_backup = drive_backup

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 25 == 0:
                gc.collect()

        def on_save(self, args, state, control, **kwargs):
            gc.collect()
            if self.drive_backup:
                try:
                    drive_ckpt = Path(self.drive_backup) / "sft_adapter"
                    drive_ckpt.mkdir(parents=True, exist_ok=True)
                    latest_ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    if latest_ckpt.exists():
                        dest_ckpt = drive_ckpt / f"checkpoint-{state.global_step}"
                        shutil.copytree(latest_ckpt, dest_ckpt, dirs_exist_ok=True)
                        print(f"\n[Drive Backup] Synced step {state.global_step} to {dest_ckpt}")
                except Exception as e:
                    print(f"\n[Drive Backup Warning] Failed to backup checkpoint: {e}")

    drive_backup = None if args.no_drive_backup else args.drive_backup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[HostRAMCleanupCallback(drive_backup=drive_backup)],
    )

    # 5. Execute Training
    print("Starting SFT training loop...")
    resume_path = args.resume_from_checkpoint
    if resume_path is None and args.auto_resume and os.path.exists(args.output_dir):
        checkpoints = list(Path(args.output_dir).glob("checkpoint-*"))
        if checkpoints:
            latest = max(checkpoints, key=os.path.getmtime)
            print(f"[Auto-Resume] Explicitly enabled; using checkpoint: {latest}")
            resume_path = str(latest)
    elif resume_path is None and os.path.exists(args.output_dir):
        checkpoints = list(Path(args.output_dir).glob("checkpoint-*"))
        if checkpoints:
            print(
                "[Resume] Found existing checkpoints but auto-resume is OFF. "
                "Pass --resume_from_checkpoint <path> or --auto_resume to reuse them."
            )

    trainer.train(resume_from_checkpoint=resume_path)

    # 6. Save Adapter & Final Checkpoints
    print(f"Saving final trained model/adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 7. Post-training generation sanity check.  This catches the old failure
    # mode where training "completed" but the model looped or never emitted a
    # terminal answer.
    generation_results: list[Dict[str, Any]] = []
    check_dataset = val_dataset if val_dataset is not None else train_dataset
    if args.generation_check > 0:
        print("Running post-training generation check...")
        generation_results = run_generation_check(
            trainer.model,
            tokenizer,
            check_dataset,
            n=args.generation_check,
        )
        failures = [
            r for r in generation_results
            if not r["terminal_answer"] or not r["eos_emitted"] or r["loop_marker"]
        ]
        if failures and args.strict_generation_check:
            reasons = {
                "terminal_answer": sum(not r["terminal_answer"] for r in failures),
                "eos_emitted": sum(not r["eos_emitted"] for r in failures),
                "loop_marker": sum(r["loop_marker"] for r in failures),
            }
            raise RuntimeError(
                f"Post-training generation check failed for {len(failures)} prompt(s): "
                f"{reasons}. Refusing to package a model that does not emit a "
                "declared terminal answer followed by the real EOS token."
            )

    manifest.update(
        {
            "status": "completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run_manifest": str(run_manifest_path),
            "final_output_dir": str(args.output_dir),
            "generation_check": args.generation_check,
            "generation_results": [
                {
                    "id": r["id"],
                    "terminal_answer": r["terminal_answer"],
                    "eos_emitted": r["eos_emitted"],
                    "loop_marker": r["loop_marker"],
                    "generated_preview": r["generated"][:2000],
                }
                for r in generation_results
            ],
            "train_log_history": trainer.state.log_history,
        }
    )
    write_run_manifest(run_manifest_path, manifest)

    # Backup to Drive only when explicitly enabled.  Kaggle runs use
    # --no_drive_backup so they do not create a hidden duplicate checkpoint
    # tree under /kaggle/working.
    if args.no_drive_backup:
        print("[Drive Sync] Disabled by --no_drive_backup.")
    else:
        drive_ok = setup_drive_sync(args.drive_backup)
        if drive_ok:
            drive_dest = Path(args.drive_backup) / "sft_adapter"
            print(f"Copying artifacts to Google Drive: {drive_dest}...")
            shutil.copytree(args.output_dir, drive_dest, dirs_exist_ok=True)
            print("Drive backup complete!")

    print("Stage 1 SFT finished successfully!")


if __name__ == "__main__":
    main()
