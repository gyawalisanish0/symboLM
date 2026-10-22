#!/usr/bin/env python
"""
Kaggle v3 SFT runner for SymboLM.

This is deliberately stricter than the old Kaggle notebook:

1. Verify the canonical train/validation hashes against manifest.json.
2. Build the same extended tokenizer that training will use.
3. Walk every training row and assert prompt-prefix equality, label masking,
   declared terminal answer, and a real EOS token.
4. Train with auto-resume off and the strict post-training generation check.
5. Refuse to package unless the generation check emitted both a declared
   terminal answer and the real EOS token.
6. Write a kernel receipt with the git commit, hashes, package versions, and
   generated previews.

Usage:
    python experiments/kaggle/sft_v3/run_sft_kernel.py --mode check
    python experiments/kaggle/sft_v3/run_sft_kernel.py --mode train

The script is run from the repository root. On Kaggle the repository is either
cloned from the pinned branch or extracted from a bundle dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

EXPECTED_TRAIN_SHA256 = (
    "5D05003B495DE3322568C4B40713CCE376A1AC05AC9B85A50B579A68A49FF63E"
)
EXPECTED_VAL_SHA256 = (
    "EB854AFF33DD31210D0B4BE61AF0165E5392F44B7B6D472C8C747161062E3EE3"
)
EXPECTED_TRAIN_ROWS = 4140
EXPECTED_VAL_ROWS = 460


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def repo_root(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).resolve()
    else:
        root = Path(__file__).resolve().parents[3]
    require((root / "config.py").exists(), f"Not a SymboLM repository: {root}")
    require(
        (root / "training" / "sft_train.py").exists(),
        f"Missing training/sft_train.py under {root}",
    )
    require(
        (root / "data" / "research_sft" / "manifest.json").exists(),
        f"Missing canonical SFT manifest under {root}",
    )
    return root


def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def verify_data(root: Path) -> Dict[str, Any]:
    data_dir = root / "data" / "research_sft"
    train = data_dir / "train.jsonl"
    val = data_dir / "val.jsonl"
    manifest_path = data_dir / "manifest.json"
    for path in (train, val, manifest_path):
        require(path.exists(), f"Missing required data artifact: {path}")

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_sha = sha256_file(train)
    val_sha = sha256_file(val)

    require(
        source_manifest.get("train_sha256", "").upper() == EXPECTED_TRAIN_SHA256,
        "data/research_sft/manifest.json does not declare the expected train hash",
    )
    require(
        source_manifest.get("val_sha256", "").upper() == EXPECTED_VAL_SHA256,
        "data/research_sft/manifest.json does not declare the expected validation hash",
    )
    require(
        train_sha == EXPECTED_TRAIN_SHA256,
        f"Train hash mismatch: got {train_sha}, expected {EXPECTED_TRAIN_SHA256}",
    )
    require(
        val_sha == EXPECTED_VAL_SHA256,
        f"Validation hash mismatch: got {val_sha}, expected {EXPECTED_VAL_SHA256}",
    )
    require(
        int(source_manifest.get("train", -1)) == EXPECTED_TRAIN_ROWS,
        "Manifest train count does not match the canonical split",
    )
    require(
        int(source_manifest.get("val", -1)) == EXPECTED_VAL_ROWS,
        "Manifest validation count does not match the canonical split",
    )
    require(
        source_manifest.get("target_kinds") == {"reasoning": 4200, "bypass": 400},
        f"Unexpected target-kind counts: {source_manifest.get('target_kinds')}",
    )

    def count_rows(path: Path) -> int:
        with open(path, "r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    train_rows = count_rows(train)
    val_rows = count_rows(val)
    require(train_rows == EXPECTED_TRAIN_ROWS, f"train rows {train_rows} != {EXPECTED_TRAIN_ROWS}")
    require(val_rows == EXPECTED_VAL_ROWS, f"val rows {val_rows} != {EXPECTED_VAL_ROWS}")

    result = {
        "train_file": str(train),
        "train_sha256": train_sha,
        "train_rows": train_rows,
        "val_file": str(val),
        "val_sha256": val_sha,
        "val_rows": val_rows,
        "source_sha256": source_manifest.get("source_sha256"),
        "target_kinds": source_manifest.get("target_kinds"),
        "source_records": source_manifest.get("source_records"),
        "kept": source_manifest.get("kept"),
        "filtered": source_manifest.get("filtered"),
        "renamed_duplicate_ids": source_manifest.get("renamed_duplicate_ids"),
    }
    print("[PREFLIGHT] Data hashes and counts verified.")
    print(json.dumps(result, indent=2))
    return result


def build_tokenizer(root: Path, base_model: str):
    """Create/load the exact extended tokenizer that training will use."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from symbolic.tokenizer_ext import extend_tokenizer

    tokenizer_dir = root / "checkpoints" / "tokenizer_extended"
    if tokenizer_dir.exists() and (tokenizer_dir / "tokenizer_config.json").exists():
        from transformers import AutoTokenizer

        print(f"[PREFLIGHT] Loading existing extended tokenizer: {tokenizer_dir}")
        return AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    print(f"[PREFLIGHT] Building extended tokenizer from {base_model}")
    return extend_tokenizer(base_model, tokenizer_dir)


def preflight_dataset(root: Path, base_model: str, full: bool = True) -> Dict[str, Any]:
    sys.path.insert(0, str(root))
    from data.symbolic_dataset import SymbolicSFTDataset
    from symbolic.prompts import uses_qwen_chat_tokens

    tokenizer = build_tokenizer(root, base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    require(tokenizer.bos_token_id is not None, "Tokenizer has no BOS token")
    require(tokenizer.eos_token_id is not None, "Tokenizer has no EOS token")
    require(
        uses_qwen_chat_tokens(tokenizer),
        "Tokenizer is not the Qwen/DeepSeek chat family; manual prompt formatting "
        "would fall back to a template and would not match training.",
    )
    # The terminal answer and think markers must be atomic tokens because the
    # training target and evaluator depend on them.  Register tags such as
    # <reg:bypass> may legitimately tokenize into several pieces; the scorer
    # reads them as text, so they are not required to be single tokens.
    for token in ("<ans>", "</ans>", "<think>", "</think>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        require(
            token_id is not None and token_id != tokenizer.unk_token_id,
            f"Tokenizer does not know required marker {token!r}",
        )

    train_path = root / "data" / "research_sft" / "train.jsonl"
    val_path = root / "data" / "research_sft" / "val.jsonl"
    max_length = 1024
    train_ds = SymbolicSFTDataset(
        train_path, tokenizer, max_length=max_length, strict_format=True
    )
    val_ds = SymbolicSFTDataset(
        val_path, tokenizer, max_length=max_length, strict_format=True
    )

    require(len(train_ds) == EXPECTED_TRAIN_ROWS, f"train dataset {len(train_ds)} rows")
    require(len(val_ds) == EXPECTED_VAL_ROWS, f"val dataset {len(val_ds)} rows")
    require(not train_ds.skipped, f"Unusable training rows were silently skipped: {train_ds.skipped}")
    require(not val_ds.skipped, f"Unusable validation rows were silently skipped: {val_ds.skipped}")
    require(
        train_ds.kinds["reasoning"] > 0 and train_ds.kinds["bypass"] > 0,
        f"Training split lost a target kind: {train_ds.kinds}",
    )

    indices = range(len(train_ds)) if full else sorted(
        {0, 1, len(train_ds) - 1} | {i * (len(train_ds) - 1) // 7 for i in range(8)}
    )
    checked = 0
    for idx in indices:
        item = train_ds[idx]
        prompt_ids = tokenizer(
            train_ds.get_prompt(idx),
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        ids = item["input_ids"].tolist()
        labels = item["labels"].tolist()
        require(
            ids[: len(prompt_ids)] == prompt_ids,
            f"Prompt is not an exact token prefix at train index {idx}",
        )
        require(
            all(label == -100 for label in labels[: len(prompt_ids)]),
            f"Prompt labels are not masked at train index {idx}",
        )
        require(
            ids[-1] == tokenizer.eos_token_id,
            f"Train index {idx} does not end with the real EOS token",
        )
        require(
            any(label != -100 for label in labels[len(prompt_ids):]),
            f"Train index {idx} has no supervised target tokens",
        )
        checked += 1
        if full and checked % 500 == 0:
            print(f"[PREFLIGHT] checked {checked}/{len(train_ds)} training rows")

    # The validation set is small enough to check in full even in sample mode.
    for idx in range(len(val_ds)):
        item = val_ds[idx]
        require(
            item["input_ids"][-1].item() == tokenizer.eos_token_id,
            f"Validation index {idx} does not end with the real EOS token",
        )

    summary = {
        "train": train_ds.summary(),
        "val": val_ds.summary(),
        "tokenizer": {
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "checked_train_rows": checked,
        "checked_val_rows": len(val_ds),
        "full_preflight": full,
    }
    print("[PREFLIGHT] Dataset smoke test passed.")
    print(json.dumps(summary, indent=2))
    return summary


def environment_snapshot() -> Dict[str, Any]:
    import platform

    snapshot: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        snapshot.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
            }
        )
        if torch.cuda.is_available():
            snapshot["gpus"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / (1024**3), 2
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        snapshot["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import transformers
        import peft

        snapshot["transformers"] = transformers.__version__
        snapshot["peft"] = peft.__version__
    except Exception as exc:
        snapshot["package_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def run_training(
    root: Path,
    output_root: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_train_samples: int,
) -> int:
    import torch

    require(torch.cuda.is_available(), "Training requested but CUDA is not available")
    adapter_dir = output_root / "adapter"
    run_manifest = output_root / "run_manifest.json"
    if adapter_dir.exists() or run_manifest.exists():
        raise RuntimeError(
            f"Training artifacts already exist under {output_root}. "
            "Use a fresh path or remove them explicitly; auto-resume is disabled."
        )
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "training.sft_train",
        "--base_model",
        base_model,
        "--train_file",
        str(root / "data" / "research_sft" / "train.jsonl"),
        "--val_file",
        str(root / "data" / "research_sft" / "val.jsonl"),
        "--output_dir",
        str(adapter_dir),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--grad_accum",
        str(grad_accum),
        "--lr",
        str(lr),
        "--no_quant",
        "--generation_check",
        "2",
        "--strict_generation_check",
        "--run_manifest",
        str(run_manifest),
        "--seed",
        "42",
        "--no_drive_backup",
    ]
    if max_train_samples > 0:
        command.extend(["--max_train_samples", str(max_train_samples)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    print("[TRAIN] Command:")
    print(" ".join(command))
    print(f"[TRAIN] cwd={root}")
    completed = subprocess.run(command, cwd=root, env=env)
    return completed.returncode


def package_output(root: Path, output_root: Path, receipt: Dict[str, Any]) -> Dict[str, Any]:
    adapter_dir = output_root / "adapter"
    run_manifest = output_root / "run_manifest.json"
    require(adapter_dir.exists(), f"Missing adapter directory: {adapter_dir}")
    require(run_manifest.exists(), f"Missing run manifest: {run_manifest}")
    require(
        (adapter_dir / "adapter_config.json").exists(),
        f"Adapter was not saved with adapter_config.json: {adapter_dir}",
    )
    require(
        any(
            (adapter_dir / name).exists()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        ),
        f"No adapter weights found in {adapter_dir}",
    )
    adapter_config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    saved_modules = adapter_config.get("modules_to_save") or []
    require(
        "embed_tokens" in saved_modules and "lm_head" in saved_modules,
        "Adapter does not persist the resized embedding/lm_head modules; "
        f"modules_to_save={saved_modules!r}",
    )

    run_data = json.loads(run_manifest.read_text(encoding="utf-8"))
    require(run_data.get("status") == "completed", "run_manifest.json is not completed")
    generation = run_data.get("generation_results") or []
    require(generation, "No generation-check results were recorded")
    for item in generation:
        require(item.get("terminal_answer"), f"Generation check missing terminal answer: {item}")
        require(item.get("eos_emitted"), f"Generation check missing real EOS: {item}")
        require(not item.get("loop_marker"), f"Generation check found a loop marker: {item}")

    receipt["run_manifest"] = str(run_manifest)
    receipt["adapter_dir"] = str(adapter_dir)
    receipt["adapter_files"] = sorted(
        str(path.relative_to(adapter_dir))
        for path in adapter_dir.rglob("*")
        if path.is_file()
    )
    receipt["adapter_sha256"] = {
        name: sha256_file(adapter_dir / name)
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
        if (adapter_dir / name).exists()
    }
    receipt["status"] = "completed"

    receipt_path = output_root / "KERNEL_RECEIPT.json"
    write_json(receipt_path, receipt)

    bundle_path = output_root / "symbolm_v3_sft_adapter.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as archive:
        archive.add(adapter_dir, arcname="symbolm_v3_sft/adapter")
        archive.add(run_manifest, arcname="symbolm_v3_sft/run_manifest.json")
        archive.add(receipt_path, arcname="symbolm_v3_sft/KERNEL_RECEIPT.json")

    receipt["bundle"] = str(bundle_path)
    receipt["bundle_sha256"] = sha256_file(bundle_path)
    write_json(receipt_path, receipt)
    return receipt


def failure_receipt(output_root: Path, root: Path, error: str) -> None:
    payload = {
        "status": "failed",
        "created_utc": utc_now(),
        "git_head": git_head(root),
        "error": error,
        "environment": environment_snapshot(),
    }
    write_json(output_root / "KERNEL_FAILURE.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SymboLM v3 Kaggle SFT runner")
    parser.add_argument("--mode", choices=("check", "train"), default="check")
    parser.add_argument("--repo", default=None, help="Repository root (default: this checkout)")
    parser.add_argument(
        "--base_model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    parser.add_argument(
        "--output_root",
        default="/kaggle/working/symbolm_v3_sft",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Diagnostic only; 0 means the full canonical split.",
    )
    parser.add_argument(
        "--sample_preflight",
        action="store_true",
        help="Check a sample instead of every training row (do not use for the primary run).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    root = repo_root(args.repo)
    output_root = Path(args.output_root)
    print(f"[INFO] repo={root}")
    print(f"[INFO] git_head={git_head(root)}")
    print(f"[INFO] mode={args.mode}")
    print(json.dumps(environment_snapshot(), indent=2))

    receipt: Dict[str, Any] = {
        "status": "started",
        "created_utc": utc_now(),
        "mode": args.mode,
        "git_head": git_head(root),
        "base_model": args.base_model,
        "environment": environment_snapshot(),
    }

    try:
        receipt["data"] = verify_data(root)
        receipt["preflight"] = preflight_dataset(
            root, args.base_model, full=not args.sample_preflight
        )
        if args.mode == "check":
            receipt["status"] = "check_passed"
            write_json(output_root / "KERNEL_CHECK.json", receipt)
            print("[DONE] Check-only mode passed.")
            return 0

        output_root.mkdir(parents=True, exist_ok=True)
        return_code = run_training(
            root=root,
            output_root=output_root,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            max_train_samples=args.max_train_samples,
        )
        receipt["train_return_code"] = return_code
        receipt["training"] = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "lr": args.lr,
            "max_train_samples": args.max_train_samples,
            "no_quant": True,
            "auto_resume": False,
            "strict_generation_check": True,
            "cuda_visible_devices": "0",
            "modules_to_save": ["embed_tokens", "lm_head"],
        }
        if return_code != 0:
            raise RuntimeError(f"training/sft_train.py exited with code {return_code}")

        package_output(root, output_root, receipt)
        print("[DONE] SFT run completed and adapter package was verified.")
        print(f"[DONE] Output root: {output_root}")
        return 0
    except Exception as exc:
        failure_receipt(output_root, root, f"{type(exc).__name__}: {exc}")
        print(f"[FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
