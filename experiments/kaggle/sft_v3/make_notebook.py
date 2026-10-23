#!/usr/bin/env python
"""Generate symboLM_v3_sft.ipynb for the Kaggle v3 SFT kernel."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "symbolm_v3_sft.ipynb"


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": None,
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# SymboLM v3 - Kaggle SFT Rescue

**Author / Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**Branch:** `research/rescue-v3`  
**Pinned tag:** `sft-v3-20261023-r6`
**Base revision:** `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
**Status:** pre-run kernel. This notebook trains one adapter; it does **not** run the evaluation.

## What this kernel guarantees

- Requires the exact pinned tag and commit identity. There is **no branch fallback**.
- Verifies the canonical SFT hashes and all 4,600 split records before training.
- Walks **all 4,140 training rows** and all **460 validation rows**: prompt must be an
  exact token prefix, prompt labels must be `-100`, every target must begin with the
  real BOS and end with the real EOS token, and every target must declare a terminal
  answer of the expected kind.
- Asserts the exact split kinds: 3,780 reasoning + 360 bypass train rows, and 420
  reasoning + 40 bypass validation rows.
- Uses the same manual Qwen chat format as evaluation. The stock chat template is not
  used for training targets.
- Initializes both the input **and output** rows for genuinely added symbols, then
  trains a small delta on those rows. The deltas break exact ties between added
  tokens and their English anchors; the r5 failure was caused by frozen `<ans>` and
  `</ans>` output rows that were byte-identical to the base `answer` row. The deltas
  are merged back into the exact resized `embed_tokens` and `lm_head` before save,
  and the artifact still uses `modules_to_save`.
- Trains with auto-resume **off**, one GPU, and `--no_quant`. Trainer checkpoints go
  to `trainer/`; only the final `adapter/` artifact is packaged.
- Runs a mixed-kind post-training gate: one reasoning prompt and one bypass prompt must
  each emit a nonempty declared answer, the expected closure, valid register structure,
  and the real EOS token, with no loop marker.
- Pins the dependency versions used for the run and records them in the receipt.
- Removes Kaggle's unused `torchao` before LoRA injection. PEFT 0.21.2 raises on
  `torchao < 0.16.0`; this run does not use torchao, and the package gate requires it
  to be absent.
- Writes `KERNEL_RECEIPT.json`, `run_manifest.json`, and
  `symbolm_v3_sft_adapter.tar.gz` to `/kaggle/working/symbolm_v3_sft/`.

## Inputs

If `/kaggle/input/symbolm-v3-sft-bundle/` is mounted, the kernel requires
`bundle_manifest.json`, checks the archive SHA-256, requires the manifest's tag to
match `sft-v3-20261023-r6`, and safely extracts it. Otherwise it clones the exact
tag from GitHub. A missing tag or missing bundle manifest is a hard failure.

## Not admissible

Do not interpret this notebook as evidence for format value, routing value, or any
benchmark score. The SFT run only produces the adapter. The primary test is the
preregistered 200 x 4 x 6 strict evaluation in `PREREG_VALUATION_V2.md`.
"""
    ),
    code(
        """# Cell 1: Fixed configuration
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

REPO_URL = "https://github.com/gyawalisanish0/symboLM.git"
GIT_TAG = "sft-v3-20261023-r6"
EXPECTED_BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EXPECTED_BASE_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
EXPECTED_PACKAGES = {
    "transformers": "5.20.1",
    "peft": "0.21.2",
    "accelerate": "1.15.0",
    "datasets": "5.1.0",
    "sympy": "1.14.0",
    "huggingface_hub": "2.2.0",
    "tokenizers": "0.23.3",
    "safetensors": "0.8.0",
}
# PEFT 0.21.2 probes for torchao while building LoRA layers and raises if it
# finds a version below 0.16.0. This run does not use torchao, and replacing the
# Kaggle image's torch stack is riskier than removing the unused package.
FORBIDDEN_PACKAGES = ("torchao",)
REPO_DIR = Path("/kaggle/working/symboLM")
OUTPUT_ROOT = Path("/kaggle/working/symbolm_v3_sft")
BUNDLE_TAR = Path("/kaggle/input/symbolm-v3-sft-bundle/repo_bundle.tar.gz")
BUNDLE_MANIFEST = BUNDLE_TAR.parent / "bundle_manifest.json"

print("REPO_URL     :", REPO_URL)
print("GIT_TAG      :", GIT_TAG)
print("BASE_REVISION:", EXPECTED_BASE_REVISION)
print("REPO_DIR     :", REPO_DIR)
print("OUTPUT_ROOT  :", OUTPUT_ROOT)
print("BUNDLE_TAR   :", BUNDLE_TAR, "exists=", BUNDLE_TAR.exists())
print("Python       :", sys.version.split()[0])
print("Platform     :", platform.platform())
"""
    ),
    code(
        """# Cell 2: GPU check
import torch

print("torch       :", torch.__version__)
print("CUDA        :", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA device. Set the Kaggle accelerator to GPU T4 before running."
    )
print("Device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GiB)")
"""
    ),
    code(
        """# Cell 3: Exact, pinned training dependencies
# Torch is supplied by Kaggle. Do not upgrade or replace it here.
packages = [f"{name}=={version}" for name, version in EXPECTED_PACKAGES.items()]
cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages]
print("Installing:", " ".join(packages))
subprocess.run(cmd, check=True)

import importlib.metadata as metadata
for name, expected in EXPECTED_PACKAGES.items():
    actual = metadata.version(name)
    print(f"{name:16s}: {actual}")
    if actual != expected:
        raise RuntimeError(f"Pinned package mismatch for {name}: {actual} != {expected}")

# PEFT 0.21.2 raises during LoRA dispatch if it finds torchao < 0.16.0, even
# though this run never uses torchao. Kaggle supplies 0.10.0. Remove the
# optional package instead of changing the pinned torch/CUDA stack.
for name in FORBIDDEN_PACKAGES:
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        installed = None
    if installed is not None:
        print(f"Removing unused {name} {installed} before LoRA injection ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", name],
            check=True,
        )
    else:
        print(f"{name:16s}: absent")
for name in FORBIDDEN_PACKAGES:
    try:
        remaining = metadata.version(name)
    except metadata.PackageNotFoundError:
        remaining = None
    if remaining is not None:
        raise RuntimeError(f"Forbidden package {name} is still installed: {remaining}")
    print(f"{name:16s}: absent (verified)")
"""
    ),
    code(
        """# Cell 4: Fetch the pinned repository, or use a verified bundle
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Unsafe tar member escapes destination: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Refusing link member in bundle: {member.name}")
    archive.extractall(destination)


shutil.rmtree(REPO_DIR, ignore_errors=True)

if BUNDLE_TAR.exists():
    if not BUNDLE_MANIFEST.exists():
        raise RuntimeError(
            "Bundle is mounted without bundle_manifest.json; refusing to extract "
            "an unverified archive. Mount the complete dataset or use the tag clone."
        )
    manifest = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))
    expected_bundle_sha = str(manifest.get("repo_bundle_sha256", "")).upper()
    actual_bundle_sha = sha256_file(BUNDLE_TAR)
    if not expected_bundle_sha or actual_bundle_sha != expected_bundle_sha:
        raise RuntimeError(
            f"Bundle hash mismatch: got {actual_bundle_sha}, "
            f"expected {expected_bundle_sha or '<missing>'}"
        )
    if manifest.get("git_tag") != GIT_TAG:
        raise RuntimeError(
            f"Bundle was built from tag {manifest.get('git_tag')!r}, "
            f"expected {GIT_TAG!r}"
        )
    if not manifest.get("git_head"):
        raise RuntimeError("Bundle manifest has no git_head")
    print("Bundle SHA-256 :", actual_bundle_sha)
    print("Bundle git head:", manifest["git_head"])
    print("Bundle git tag :", manifest["git_tag"])
    with tarfile.open(BUNDLE_TAR, "r:gz") as archive:
        safe_extract(archive, Path("/kaggle/working"))
    OBSERVED_GIT_HEAD = manifest["git_head"]
    OBSERVED_GIT_TAG = manifest["git_tag"]
    SOURCE = "bundle"
else:
    print("Bundle not mounted; cloning the exact pinned tag (no branch fallback).")
    clone = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", GIT_TAG, REPO_URL, str(REPO_DIR)],
        text=True,
    )
    if clone.returncode != 0:
        raise RuntimeError(
            f"Tag clone failed for {GIT_TAG!r}. No branch fallback is permitted."
        )
    OBSERVED_GIT_HEAD = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_DIR
    ).strip()
    OBSERVED_GIT_TAG = subprocess.check_output(
        ["git", "describe", "--tags", "--exact-match", "HEAD"], text=True, cwd=REPO_DIR
    ).strip()
    if OBSERVED_GIT_TAG != GIT_TAG:
        raise RuntimeError(
            f"Checkout tag is {OBSERVED_GIT_TAG!r}, expected {GIT_TAG!r}"
        )
    SOURCE = "git"

if not (REPO_DIR / "experiments" / "kaggle" / "sft_v3" / "run_sft_kernel.py").exists():
    raise RuntimeError(f"Repository is incomplete at {REPO_DIR}")

os.chdir(REPO_DIR)
print("Source         :", SOURCE)
print("Repository     :", REPO_DIR)
print("Working dir    :", Path.cwd())
print("Observed head  :", OBSERVED_GIT_HEAD)
print("Observed tag   :", OBSERVED_GIT_TAG)
"""
    ),
    code(
        """# Cell 5: Full hash, tokenizer, prompt-prefix, label-mask, BOS/EOS, and kind preflight
# This reads every one of the 4,140 training rows and all 460 validation rows.
cmd = [
    sys.executable,
    "experiments/kaggle/sft_v3/run_sft_kernel.py",
    "--repo", str(REPO_DIR),
    "--mode", "check",
    "--output_root", str(OUTPUT_ROOT),
    "--base_model", EXPECTED_BASE_MODEL,
    "--base_revision", EXPECTED_BASE_REVISION,
    "--expected_git_head", OBSERVED_GIT_HEAD,
    "--expected_git_tag", GIT_TAG,
    "--observed_git_head", OBSERVED_GIT_HEAD,
    "--observed_git_tag", OBSERVED_GIT_TAG,
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, cwd=REPO_DIR, check=True)
"""
    ),
    code(
        """# Cell 6: Fresh SFT run with the strict mixed-kind generation gate
# Effective batch = 2 x 8 = 16. Auto-resume is off by design.
# If this cell fails, do not package the adapter. Read run_manifest.json and the
# generation previews first.
cmd = [
    sys.executable,
    "experiments/kaggle/sft_v3/run_sft_kernel.py",
    "--repo", str(REPO_DIR),
    "--mode", "train",
    "--output_root", str(OUTPUT_ROOT),
    "--base_model", EXPECTED_BASE_MODEL,
    "--base_revision", EXPECTED_BASE_REVISION,
    "--expected_git_head", OBSERVED_GIT_HEAD,
    "--expected_git_tag", GIT_TAG,
    "--observed_git_head", OBSERVED_GIT_HEAD,
    "--observed_git_tag", OBSERVED_GIT_TAG,
    "--epochs", "3",
    "--batch_size", "2",
    "--grad_accum", "8",
    "--lr", "2e-4",
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, cwd=REPO_DIR, check=True)
"""
    ),
    code(
        """# Cell 7: Inspect the package and the strict acceptance gates
receipt_path = OUTPUT_ROOT / "KERNEL_RECEIPT.json"
run_manifest_path = OUTPUT_ROOT / "run_manifest.json"
bundle_path = OUTPUT_ROOT / "symbolm_v3_sft_adapter.tar.gz"
adapter_config_path = OUTPUT_ROOT / "adapter" / "adapter_config.json"
adapter_weights_path = OUTPUT_ROOT / "adapter" / "adapter_model.safetensors"

for path in (receipt_path, run_manifest_path, bundle_path, adapter_config_path):
    if not path.exists():
        raise RuntimeError(f"Missing expected output: {path}")

receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))

if receipt.get("status") != "completed":
    raise RuntimeError(f"Receipt is not completed: {receipt.get('status')!r}")
if receipt.get("git_head") != OBSERVED_GIT_HEAD:
    raise RuntimeError(
        f"Receipt commit {receipt.get('git_head')!r} != observed {OBSERVED_GIT_HEAD!r}"
    )
if receipt.get("git_tag") != GIT_TAG:
    raise RuntimeError(
        f"Receipt tag {receipt.get('git_tag')!r} != expected {GIT_TAG!r}"
    )
if receipt.get("expected_git_tag") != GIT_TAG:
    raise RuntimeError(
        f"Receipt expected tag {receipt.get('expected_git_tag')!r} != {GIT_TAG!r}"
    )
if receipt.get("base_model") != EXPECTED_BASE_MODEL:
    raise RuntimeError("Receipt base model mismatch")
if receipt.get("base_revision") != EXPECTED_BASE_REVISION:
    raise RuntimeError("Receipt base revision mismatch")
if run_manifest.get("git_head") != OBSERVED_GIT_HEAD:
    raise RuntimeError("run_manifest git head mismatch")
if run_manifest.get("git_tag") != GIT_TAG:
    raise RuntimeError("run_manifest git tag mismatch")
if run_manifest.get("base_model") != EXPECTED_BASE_MODEL:
    raise RuntimeError("run_manifest base model mismatch")
if run_manifest.get("base_revision") != EXPECTED_BASE_REVISION:
    raise RuntimeError("run_manifest base revision mismatch")
manifest_args = run_manifest.get("args") or {}
if manifest_args.get("generation_check") != 2:
    raise RuntimeError("run_manifest did not run the two-prompt generation gate")
if manifest_args.get("strict_generation_check") is not True:
    raise RuntimeError("run_manifest did not enforce the strict generation gate")

if receipt.get("data", {}).get("train_sha256") != "BB50D9CBD56B0D2B515D7B922DF10578F587AA60358B0FF87A11821C4D3D94DC":
    raise RuntimeError("Receipt train hash mismatch")
if receipt.get("data", {}).get("val_sha256") != "3FCFCC4141522062087D0F35FDF2C54226D454A1A71FE59BEA26F42636FDEDCD":
    raise RuntimeError("Receipt validation hash mismatch")
if run_manifest.get("train_summary", {}).get("kinds") != {"reasoning": 3780, "bypass": 360}:
    raise RuntimeError(f"Unexpected train kinds: {run_manifest.get('train_summary')}")
if run_manifest.get("val_summary", {}).get("kinds") != {"reasoning": 420, "bypass": 40}:
    raise RuntimeError(f"Unexpected val kinds: {run_manifest.get('val_summary')}")

for name, expected in EXPECTED_PACKAGES.items():
    actual = (run_manifest.get("packages") or {}).get(name)
    if actual != expected:
        raise RuntimeError(f"Package {name} in manifest is {actual!r}, expected {expected!r}")
for name in FORBIDDEN_PACKAGES:
    actual = (run_manifest.get("packages") or {}).get(name)
    if actual != "absent":
        raise RuntimeError(
            f"Forbidden package {name} in manifest is {actual!r}, expected 'absent'"
        )

if run_manifest.get("tokenizer", {}).get("vocab_size") != 151712:
    raise RuntimeError("Tokenizer vocabulary is not the expected 151,712 rows")
extension_meta = run_manifest.get("tokenizer", {}).get("extension_meta") or {}
if extension_meta.get("base_revision") != EXPECTED_BASE_REVISION:
    raise RuntimeError("Tokenizer extension base revision mismatch")

saved_modules = set(adapter_config.get("modules_to_save") or [])
if not {"embed_tokens", "lm_head"} <= saved_modules:
    raise RuntimeError(
        "Adapter does not persist resized embeddings/lm_head: "
        f"modules_to_save={sorted(saved_modules)}"
    )
if not adapter_weights_path.exists():
    raise RuntimeError("adapter_model.safetensors is missing")
from safetensors import safe_open
with safe_open(str(adapter_weights_path), framework="pt", device="cpu") as handle:
    tensor_keys = set(handle.keys())
if not any("embed_tokens" in key for key in tensor_keys):
    raise RuntimeError("Adapter weights contain no embed_tokens tensor")
if not any("lm_head" in key for key in tensor_keys):
    raise RuntimeError("Adapter weights contain no lm_head tensor")

if any(name.startswith("checkpoint-") for name in receipt.get("adapter_files", [])):
    raise RuntimeError("Trainer checkpoint leaked into the adapter package")
if not (OUTPUT_ROOT / "trainer").exists():
    raise RuntimeError("Trainer checkpoint directory is missing")

actual_bundle_sha = sha256_file(bundle_path)
if receipt.get("bundle_sha256") != actual_bundle_sha:
    raise RuntimeError(
        f"Bundle receipt hash {receipt.get('bundle_sha256')!r} != {actual_bundle_sha!r}"
    )
with tarfile.open(bundle_path, "r:gz") as archive:
    names = archive.getnames()
if any("trainer" in name or "checkpoint-" in name for name in names):
    raise RuntimeError("Trainer checkpoints leaked into the output tarball")

generation = run_manifest.get("generation_results") or []
if {item.get("target_kind") for item in generation} != {"reasoning", "bypass"}:
    raise RuntimeError(f"Generation checks did not cover both kinds: {generation}")
for item in generation:
    if not item.get("terminal_answer"):
        raise RuntimeError(f"Missing terminal answer: {item}")
    if not item.get("eos_emitted"):
        raise RuntimeError(f"Missing real EOS: {item}")
    if item.get("loop_marker"):
        raise RuntimeError(f"Loop marker: {item}")
    if not item.get("format_valid"):
        raise RuntimeError(f"Invalid register format: {item}")
    if not item.get("answer"):
        raise RuntimeError(f"Missing declared answer: {item}")
    if item.get("closure") != item.get("expected_closure"):
        raise RuntimeError(f"Wrong closure kind: {item}")

print("Receipt status :", receipt.get("status"))
print("Git head       :", receipt.get("git_head"))
print("Git tag        :", receipt.get("git_tag"))
print("Base revision  :", receipt.get("base_revision"))
print("Artifact rows  :", sorted(saved_modules))
print("Adapter tensors:", sorted(tensor_keys))
print("Adapter dir    :", receipt.get("adapter_dir"))
print("Trainer dir    :", receipt.get("trainer_dir"))
print("Bundle         :", receipt.get("bundle"))
print("Bundle SHA-256 :", receipt.get("bundle_sha256"))
print("Train SHA-256  :", receipt.get("data", {}).get("train_sha256"))
print("Val SHA-256    :", receipt.get("data", {}).get("val_sha256"))
print("Generation      :")
for item in generation:
    print(json.dumps(item, ensure_ascii=False))

print("\\nTop-level output tree:")
for path in sorted(OUTPUT_ROOT.iterdir()):
    kind = "dir" if path.is_dir() else "file"
    size = "" if path.is_dir() else f" ({path.stat().st_size} bytes)"
    print(f"  {path.name} [{kind}]{size}")
"""
    ),
    markdown(
        """## Acceptance gates

The adapter is admissible for the next stage only when all of these are true:

- `KERNEL_RECEIPT.json` has `"status": "completed"` and the exact pinned commit/tag.
- `data.train_sha256` is `BB50D9CBD56B0D2B515D7B922DF10578F587AA60358B0FF87A11821C4D3D94DC`.
- `data.val_sha256` is `3FCFCC4141522062087D0F35FDF2C54226D454A1A71FE59BEA26F42636FDEDCD`.
- The train split is exactly 3,780 reasoning + 360 bypass; validation is 420 + 40.
- Every generation check has the expected kind, a nonempty answer, `format_valid: true`,
  `terminal_answer: true`, `eos_emitted: true`, and `loop_marker: false`.
- `adapter_config.json` lists `embed_tokens` and `lm_head` in `modules_to_save`, and the
  actual safetensors keys contain both.
- The adapter directory and tarball contain no Trainer checkpoint.
- `symbolm_v3_sft_adapter.tar.gz` exists and its SHA-256 is in the receipt.
- The recorded base revision is `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
- `torchao` is absent from the runtime package record; it is not used by this run.

If any gate fails, do not run the 4,800-cell evaluation. Fix the pipeline or report the
run as a pipeline failure.

## Next stage

1. Run the 100 x 4 x 6 diagnostic to catch format/EOS regressions.
2. If the diagnostic passes, run the preregistered 200 x 4 x 6 strict evaluation.
3. Report strict accuracy, J/correct, format validity, route validity, and the C-vs-D
   decision from `PREREG_VALUATION_V2.md`.

The SFT run itself is not a result.
"""
    ),
]

for index, cell in enumerate(cells, start=1):
    cell["id"] = f"cell-{index}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
    handle.write(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")
print(f"[OK] Wrote {OUT}")
