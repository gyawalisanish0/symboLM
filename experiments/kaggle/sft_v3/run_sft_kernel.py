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
from typing import Any, Dict, Optional, Tuple

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
EXPECTED_TRAIN_KINDS = {"reasoning": 3780, "bypass": 360}
EXPECTED_VAL_KINDS = {"reasoning": 420, "bypass": 40}
EXPECTED_BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EXPECTED_BASE_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
EXPECTED_TOKENIZER_VOCAB = 151712
EXPECTED_BOS_ID = 151646
EXPECTED_EOS_ID = 151643
EXPECTED_PACKAGE_VERSIONS = {
    "transformers": "5.20.1",
    "peft": "0.21.2",
    "accelerate": "1.15.0",
    "datasets": "5.1.0",
    "sympy": "1.14.0",
    "huggingface_hub": "2.2.0",
    "tokenizers": "0.23.3",
    "safetensors": "0.8.0",
}


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


def git_tag(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def resolve_git_identity(
    root: Path,
    expected_head: Optional[str],
    expected_tag: Optional[str],
    observed_head: Optional[str],
    observed_tag: Optional[str],
) -> Tuple[str, str]:
    """Resolve and verify the immutable git identity of this checkout/bundle."""
    has_git = (root / ".git").exists()
    actual_head = git_head(root) if has_git else (observed_head or "unknown")
    actual_tag = git_tag(root) if has_git else (observed_tag or "unknown")

    if has_git and observed_head:
        require(
            actual_head == observed_head,
            f"Observed git head {observed_head} does not match checkout {actual_head}",
        )
    if has_git and observed_tag:
        require(
            actual_tag == observed_tag,
            f"Observed git tag {observed_tag} does not match checkout {actual_tag}",
        )

    require(
        actual_head != "unknown",
        "Could not determine the checked-out commit. For a bundle, pass --observed_git_head "
        "from the verified bundle_manifest.json.",
    )
    if expected_head:
        require(
            actual_head == expected_head,
            f"Commit mismatch: got {actual_head}, expected {expected_head}. "
            "Branch fallback is not permitted.",
        )
    if expected_tag:
        require(
            actual_tag == expected_tag,
            f"Tag mismatch: got {actual_tag!r}, expected {expected_tag!r}. "
            "Branch fallback is not permitted.",
        )
    return actual_head, actual_tag


def require_clean_tracked_tree(root: Path) -> None:
    """A training checkout must match the tagged commit exactly."""
    if not (root / ".git").exists():
        return
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    require(
        not status,
        "Training checkout has tracked modifications; commit them and retag, "
        f"or start from a clean tag clone:\n{status}",
    )


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


def build_tokenizer(root: Path, base_model: str, base_revision: str):
    """Create/load the exact extended tokenizer that training will use."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from symbolic.tokenizer_ext import extend_tokenizer

    tokenizer_dir = root / "checkpoints" / "tokenizer_extended"
    meta_path = tokenizer_dir / "tokenizer_extension_meta.json"
    rebuild = not (
        tokenizer_dir.exists() and (tokenizer_dir / "tokenizer_config.json").exists()
    )
    if not rebuild:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if meta.get("base_model") not in (None, base_model):
            print("[PREFLIGHT] Cached tokenizer belongs to a different base model; rebuilding.")
            rebuild = True
        if meta.get("base_revision") != base_revision:
            print(
                "[PREFLIGHT] Cached tokenizer has no matching immutable revision "
                f"({meta.get('base_revision')!r} != {base_revision!r}); rebuilding."
            )
            rebuild = True

    if not rebuild:
        from transformers import AutoTokenizer

        print(f"[PREFLIGHT] Loading existing extended tokenizer: {tokenizer_dir}")
        return AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    print(f"[PREFLIGHT] Building extended tokenizer from {base_model}@{base_revision}")
    if tokenizer_dir.exists():
        shutil.rmtree(tokenizer_dir)
    return extend_tokenizer(base_model, tokenizer_dir, revision=base_revision)


def preflight_dataset(
    root: Path, base_model: str, base_revision: str, full: bool = True
) -> Dict[str, Any]:
    sys.path.insert(0, str(root))
    from config import config
    from data.symbolic_dataset import SymbolicSFTDataset
    from eval.strict_scorer import validate_register_trace
    from symbolic.prompts import (
        classify_target_body,
        extract_target_answer,
        uses_qwen_chat_tokens,
    )

    tokenizer = build_tokenizer(root, base_model, base_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    require(tokenizer.bos_token_id is not None, "Tokenizer has no BOS token")
    require(tokenizer.eos_token_id is not None, "Tokenizer has no EOS token")
    require(
        uses_qwen_chat_tokens(tokenizer),
        "Tokenizer is not the Qwen/DeepSeek chat family; manual prompt formatting "
        "would fall back to a template and would not match training.",
    )
    require(
        len(tokenizer) == EXPECTED_TOKENIZER_VOCAB,
        f"Tokenizer vocab is {len(tokenizer)}, expected {EXPECTED_TOKENIZER_VOCAB}",
    )
    require(
        tokenizer.bos_token_id == EXPECTED_BOS_ID,
        f"BOS id is {tokenizer.bos_token_id}, expected {EXPECTED_BOS_ID}",
    )
    require(
        tokenizer.eos_token_id == EXPECTED_EOS_ID,
        f"EOS id is {tokenizer.eos_token_id}, expected {EXPECTED_EOS_ID}",
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
    max_length = config.sft_max_seq_len
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
        dict(train_ds.kinds) == EXPECTED_TRAIN_KINDS,
        f"Training split kinds are {dict(train_ds.kinds)}, expected {EXPECTED_TRAIN_KINDS}",
    )
    require(
        dict(val_ds.kinds) == EXPECTED_VAL_KINDS,
        f"Validation split kinds are {dict(val_ds.kinds)}, expected {EXPECTED_VAL_KINDS}",
    )

    def check_row(dataset, idx: int, split: str) -> None:
        item = dataset[idx]
        target_body = dataset.get_target_body(idx)
        kind, reason = classify_target_body(target_body)
        require(
            kind in {"reasoning", "bypass"},
            f"{split} index {idx} has invalid target kind {kind}: {reason}",
        )
        target_answer = extract_target_answer(target_body)
        require(
            bool(target_answer),
            f"{split} index {idx} does not declare a nonempty terminal answer",
        )
        target_format_valid, target_format_reason = validate_register_trace(
            target_body, arm="C", expected_kind=kind
        )
        require(
            target_format_valid,
            f"{split} index {idx} has an invalid target contract: "
            f"{target_format_reason}",
        )
        prompt_ids = tokenizer(
            dataset.get_prompt(idx),
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        ids = item["input_ids"].tolist()
        labels = item["labels"].tolist()
        require(
            ids and ids[0] == tokenizer.bos_token_id,
            f"{split} index {idx} does not begin with the real BOS token",
        )
        require(
            ids[: len(prompt_ids)] == prompt_ids,
            f"Prompt is not an exact token prefix at {split} index {idx}",
        )
        require(
            len(prompt_ids) < len(ids),
            f"{split} index {idx} has no target tokens after the prompt",
        )
        require(
            all(label == -100 for label in labels[: len(prompt_ids)]),
            f"Prompt labels are not masked at {split} index {idx}",
        )
        require(
            all(label != -100 for label in labels[len(prompt_ids):]),
            f"Target labels are not fully supervised at {split} index {idx}",
        )
        require(
            ids[-1] == tokenizer.eos_token_id,
            f"{split} index {idx} does not end with the real EOS token",
        )

    indices = range(len(train_ds)) if full else sorted(
        {0, 1, len(train_ds) - 1} | {i * (len(train_ds) - 1) // 7 for i in range(8)}
    )
    checked = 0
    for idx in indices:
        check_row(train_ds, idx, "train")
        checked += 1
        if full and checked % 500 == 0:
            print(f"[PREFLIGHT] checked {checked}/{len(train_ds)} training rows")

    # The validation set is small enough to check in full even in sample mode.
    for idx in range(len(val_ds)):
        check_row(val_ds, idx, "val")

    extension_meta = {}
    try:
        extension_meta = json.loads(
            (root / "checkpoints" / "tokenizer_extended" / "tokenizer_extension_meta.json")
            .read_text(encoding="utf-8")
        )
    except Exception:
        pass
    summary = {
        "train": train_ds.summary(),
        "val": val_ds.summary(),
        "tokenizer": {
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "extension_meta": extension_meta,
        },
        "checked_train_rows": checked,
        "checked_val_rows": len(val_ds),
        "expected_train_kinds": EXPECTED_TRAIN_KINDS,
        "expected_val_kinds": EXPECTED_VAL_KINDS,
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
    base_revision: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_train_samples: int,
    git_head: str,
    git_tag: str,
) -> int:
    import torch

    require(torch.cuda.is_available(), "Training requested but CUDA is not available")
    trainer_dir = output_root / "trainer"
    adapter_dir = output_root / "adapter"
    run_manifest = output_root / "run_manifest.json"
    if trainer_dir.exists() or adapter_dir.exists() or run_manifest.exists():
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
        "--base_revision",
        base_revision,
        "--train_file",
        str(root / "data" / "research_sft" / "train.jsonl"),
        "--val_file",
        str(root / "data" / "research_sft" / "val.jsonl"),
        "--output_dir",
        str(trainer_dir),
        "--adapter_dir",
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
    env["SYMBOLM_GIT_HEAD"] = git_head
    env["SYMBOLM_GIT_TAG"] = git_tag

    print("[TRAIN] Command:")
    print(" ".join(command))
    print(f"[TRAIN] cwd={root}")
    completed = subprocess.run(command, cwd=root, env=env)
    return completed.returncode


def adapter_tensor_keys(weights_path: Path) -> set[str]:
    """Read the actual adapter tensor names; config-only checks are insufficient."""
    if weights_path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            return set(handle.keys())

    import torch

    payload = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    require(isinstance(payload, dict), f"Unrecognized adapter weights format: {weights_path}")
    return set(payload.keys())


def package_output(root: Path, output_root: Path, receipt: Dict[str, Any]) -> Dict[str, Any]:
    adapter_dir = output_root / "adapter"
    trainer_dir = output_root / "trainer"
    run_manifest = output_root / "run_manifest.json"
    require(adapter_dir.exists(), f"Missing adapter directory: {adapter_dir}")
    require(trainer_dir.exists(), f"Missing trainer checkpoint directory: {trainer_dir}")
    require(run_manifest.exists(), f"Missing run manifest: {run_manifest}")
    require(
        (adapter_dir / "adapter_config.json").exists(),
        f"Adapter was not saved with adapter_config.json: {adapter_dir}",
    )
    weights_name = next(
        (
            name
            for name in ("adapter_model.safetensors", "adapter_model.bin")
            if (adapter_dir / name).exists()
        ),
        None,
    )
    require(weights_name is not None, f"No adapter weights found in {adapter_dir}")
    weights_path = adapter_dir / weights_name

    adapter_config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    saved_modules = adapter_config.get("modules_to_save") or []
    require(
        "embed_tokens" in saved_modules and "lm_head" in saved_modules,
        "Adapter does not persist the resized embedding/lm_head modules; "
        f"modules_to_save={saved_modules!r}",
    )
    tensor_keys = adapter_tensor_keys(weights_path)
    require(
        any("embed_tokens" in key for key in tensor_keys),
        f"adapter weights contain no embed_tokens tensor: {sorted(tensor_keys)[:10]}",
    )
    require(
        any("lm_head" in key for key in tensor_keys),
        f"adapter weights contain no lm_head tensor: {sorted(tensor_keys)[:10]}",
    )

    run_data = json.loads(run_manifest.read_text(encoding="utf-8"))
    require(run_data.get("status") == "completed", "run_manifest.json is not completed")
    require(
        run_data.get("train_sha256", "").upper() == EXPECTED_TRAIN_SHA256,
        "run_manifest train hash does not match the canonical split",
    )
    require(
        run_data.get("val_sha256", "").upper() == EXPECTED_VAL_SHA256,
        "run_manifest validation hash does not match the canonical split",
    )
    require(
        run_data.get("base_model") == EXPECTED_BASE_MODEL
        and run_data.get("base_model") == receipt.get("base_model"),
        "run_manifest base_model does not match the pinned receipt",
    )
    require(
        run_data.get("base_revision") == EXPECTED_BASE_REVISION
        and run_data.get("base_revision") == receipt.get("base_revision"),
        "run_manifest base_revision does not match the pinned receipt",
    )
    require(
        run_data.get("git_head") == receipt.get("git_head"),
        "run_manifest git_head does not match the receipt",
    )
    require(
        run_data.get("git_tag") == receipt.get("git_tag"),
        "run_manifest git_tag does not match the receipt",
    )

    train_summary = run_data.get("train_summary") or {}
    val_summary = run_data.get("val_summary") or {}
    require(
        train_summary.get("records") == EXPECTED_TRAIN_ROWS
        and train_summary.get("kinds") == EXPECTED_TRAIN_KINDS,
        f"Unexpected train summary: {train_summary}",
    )
    require(
        val_summary.get("records") == EXPECTED_VAL_ROWS
        and val_summary.get("kinds") == EXPECTED_VAL_KINDS,
        f"Unexpected validation summary: {val_summary}",
    )

    tokenizer_data = run_data.get("tokenizer") or {}
    require(
        tokenizer_data.get("vocab_size") == EXPECTED_TOKENIZER_VOCAB,
        f"Unexpected tokenizer vocab: {tokenizer_data}",
    )
    require(
        tokenizer_data.get("bos_token_id") == EXPECTED_BOS_ID
        and tokenizer_data.get("eos_token_id") == EXPECTED_EOS_ID,
        f"Unexpected tokenizer special ids: {tokenizer_data}",
    )
    extension_meta = tokenizer_data.get("extension_meta") or {}
    require(
        extension_meta.get("base_revision") == EXPECTED_BASE_REVISION,
        "Tokenizer extension was not built from the pinned base revision",
    )

    args = run_data.get("args") or {}
    require(args.get("epochs") == 3, f"Unexpected epochs: {args.get('epochs')}")
    require(args.get("batch_size") == 2, f"Unexpected batch_size: {args.get('batch_size')}")
    require(args.get("grad_accum") == 8, f"Unexpected grad_accum: {args.get('grad_accum')}")
    require(abs(float(args.get("lr", 0.0)) - 2e-4) < 1e-12, f"Unexpected lr: {args.get('lr')}")
    require(args.get("seed") == 42, f"Unexpected seed: {args.get('seed')}")
    require(args.get("max_train_samples") == 0, "Training used a truncated sample, not the full split")
    require(args.get("no_quant") is True, "Training did not use --no_quant")
    require(args.get("auto_resume") is False, "Training enabled auto-resume")
    require(args.get("no_drive_backup") is True, "Training did not disable Drive backup")
    require(args.get("generation_check") == 2, "Training did not run the two-prompt generation gate")
    require(
        args.get("strict_generation_check") is True,
        "Training did not enforce the strict generation gate",
    )
    require(
        args.get("base_revision") == receipt.get("base_revision"),
        "Training base_revision does not match the receipt",
    )

    generation = run_data.get("generation_results") or []
    require(generation, "No generation-check results were recorded")
    require(
        {item.get("target_kind") for item in generation} == {"reasoning", "bypass"},
        f"Generation check did not cover both target kinds: {generation}",
    )
    for item in generation:
        require(item.get("terminal_answer"), f"Generation check missing terminal answer: {item}")
        require(item.get("eos_emitted"), f"Generation check missing real EOS: {item}")
        require(not item.get("loop_marker"), f"Generation check found a loop marker: {item}")
        require(item.get("format_valid"), f"Generation check format invalid: {item}")
        require(item.get("answer"), f"Generation check missing declared answer: {item}")
        require(
            item.get("closure") == item.get("expected_closure"),
            f"Generation check used the wrong closure kind: {item}",
        )

    package_versions = run_data.get("packages") or {}
    for name, expected in EXPECTED_PACKAGE_VERSIONS.items():
        require(
            package_versions.get(name) == expected,
            f"Package {name} version {package_versions.get(name)!r} != pinned {expected!r}",
        )

    adapter_files = sorted(
        str(path.relative_to(adapter_dir))
        for path in adapter_dir.rglob("*")
        if path.is_file()
    )
    require(
        not any(name.startswith("checkpoint-") for name in adapter_files),
        f"Trainer checkpoints leaked into the adapter directory: {adapter_files}",
    )

    receipt["run_manifest"] = str(run_manifest)
    receipt["trainer_dir"] = str(trainer_dir)
    receipt["adapter_dir"] = str(adapter_dir)
    receipt["adapter_files"] = adapter_files
    receipt["adapter_tensor_keys"] = sorted(tensor_keys)
    receipt["adapter_sha256"] = {
        name: sha256_file(adapter_dir / name)
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
        if (adapter_dir / name).exists()
    }

    receipt_path = output_root / "KERNEL_RECEIPT.json"
    receipt["status"] = "packaging"
    write_json(receipt_path, receipt)

    # The package contains only the final adapter and the run manifest.  The
    # Trainer checkpoint tree stays outside the archive and outside the
    # reproducibility boundary.
    bundle_path = output_root / "symbolm_v3_sft_adapter.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as archive:
        archive.add(adapter_dir, arcname="symbolm_v3_sft/adapter")
        archive.add(run_manifest, arcname="symbolm_v3_sft/run_manifest.json")

    receipt["bundle"] = str(bundle_path)
    receipt["bundle_sha256"] = sha256_file(bundle_path)
    receipt["status"] = "completed"
    write_json(receipt_path, receipt)
    return receipt


def failure_receipt(
    output_root: Path,
    root: Path,
    error: str,
    receipt: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "status": "failed",
        "created_utc": utc_now(),
        "git_head": git_head(root),
        "git_tag": git_tag(root),
        "error": error,
        "environment": environment_snapshot(),
    }
    if receipt:
        payload["expected_git_head"] = receipt.get("expected_git_head")
        payload["expected_git_tag"] = receipt.get("expected_git_tag")
        payload["base_model"] = receipt.get("base_model")
        payload["base_revision"] = receipt.get("base_revision")
    write_json(output_root / "KERNEL_FAILURE.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SymboLM v3 Kaggle SFT runner")
    parser.add_argument("--mode", choices=("check", "train"), default="check")
    parser.add_argument("--repo", default=None, help="Repository root (default: this checkout)")
    parser.add_argument(
        "--base_model",
        default=EXPECTED_BASE_MODEL,
    )
    parser.add_argument(
        "--base_revision",
        default=EXPECTED_BASE_REVISION,
        help="Immutable Hugging Face revision/commit for the base model.",
    )
    parser.add_argument(
        "--expected_git_head",
        default=None,
        help="Required full commit hash. Training refuses to start without it.",
    )
    parser.add_argument(
        "--expected_git_tag",
        default=None,
        help="Required exact tag. Training refuses to start without it.",
    )
    parser.add_argument(
        "--observed_git_head",
        default=None,
        help="Commit recorded in a verified bundle manifest (bundle checkouts have no .git).",
    )
    parser.add_argument(
        "--observed_git_tag",
        default=None,
        help="Tag recorded in a verified bundle manifest (bundle checkouts have no .git).",
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
    print(f"[INFO] mode={args.mode}")
    print(json.dumps(environment_snapshot(), indent=2))

    receipt: Dict[str, Any] = {
        "status": "started",
        "created_utc": utc_now(),
        "mode": args.mode,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "expected_git_head": args.expected_git_head,
        "expected_git_tag": args.expected_git_tag,
        "observed_git_head": args.observed_git_head,
        "observed_git_tag": args.observed_git_tag,
        "environment": environment_snapshot(),
    }

    try:
        require(
            args.base_model == EXPECTED_BASE_MODEL,
            f"Base model {args.base_model!r} != pinned {EXPECTED_BASE_MODEL!r}",
        )
        require(
            args.base_revision == EXPECTED_BASE_REVISION,
            f"Base revision {args.base_revision!r} != pinned {EXPECTED_BASE_REVISION!r}",
        )
        if args.mode == "train":
            require(
                bool(args.expected_git_head and args.expected_git_tag),
                "Training requires --expected_git_head and --expected_git_tag; "
                "a branch name is not a valid provenance anchor.",
            )
            require_clean_tracked_tree(root)
        actual_head, actual_tag = resolve_git_identity(
            root,
            expected_head=args.expected_git_head,
            expected_tag=args.expected_git_tag,
            observed_head=args.observed_git_head,
            observed_tag=args.observed_git_tag,
        )
        receipt["git_head"] = actual_head
        receipt["git_tag"] = actual_tag
        print(f"[INFO] git_head={actual_head}")
        print(f"[INFO] git_tag={actual_tag}")

        receipt["data"] = verify_data(root)
        receipt["preflight"] = preflight_dataset(
            root,
            args.base_model,
            args.base_revision,
            full=not args.sample_preflight,
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
            base_revision=args.base_revision,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            max_train_samples=args.max_train_samples,
            git_head=actual_head,
            git_tag=actual_tag,
        )
        receipt["train_return_code"] = return_code
        receipt["training"] = {
            "base_revision": args.base_revision,
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
            "adapter_dir": str(output_root / "adapter"),
            "trainer_dir": str(output_root / "trainer"),
        }
        if return_code != 0:
            raise RuntimeError(f"training/sft_train.py exited with code {return_code}")

        package_output(root, output_root, receipt)
        print("[DONE] SFT run completed and adapter package was verified.")
        print(f"[DONE] Output root: {output_root}")
        return 0
    except Exception as exc:
        failure_receipt(
            output_root,
            root,
            f"{type(exc).__name__}: {exc}",
            receipt=receipt,
        )
        print(f"[FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
