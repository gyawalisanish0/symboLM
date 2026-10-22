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
**Pinned tag:** `sft-v3-20261022`  
**Status:** pre-run kernel. This notebook trains one adapter; it does **not** run the evaluation.

## What this kernel guarantees

- Verifies the canonical SFT hashes and counts before training.
- Walks **all 4,140 training rows**, not a sample: prompt must be an exact token prefix,
  prompt labels must be `-100`, every target must end with the real EOS token, and every
  target must declare a terminal answer.
- Uses the same manual Qwen chat format as evaluation. The stock chat template is not
  used for training targets.
- Trains with auto-resume **off**, one GPU, `--no_quant`, and the strict post-training
  generation check.
- Refuses to package the adapter if a generation check is missing a terminal answer,
  the real EOS, or contains a loop marker.
- Writes `KERNEL_RECEIPT.json`, `run_manifest.json`, and
  `symbolm_v3_sft_adapter.tar.gz` to `/kaggle/working/symbolm_v3_sft/`.

## Inputs

The kernel prefers the optional dataset `sanishgyawali/symbolm-v3-sft-bundle` at
`/kaggle/input/symbolm-v3-sft-bundle/`. If it is not mounted, the kernel clones the
pinned tag from GitHub. If the tag is missing, it falls back to the branch and prints a
warning; use the receipt's git commit for provenance.

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

# One T4 is enough for a 1.5B LoRA run and makes energy accounting unambiguous.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

REPO_URL = "https://github.com/gyawalisanish0/symboLM.git"
GIT_TAG = "sft-v3-20261022"
GIT_BRANCH = "research/rescue-v3"
REPO_DIR = Path("/kaggle/working/symboLM")
OUTPUT_ROOT = Path("/kaggle/working/symbolm_v3_sft")
BUNDLE_TAR = Path("/kaggle/input/symbolm-v3-sft-bundle/repo_bundle.tar.gz")
BUNDLE_MANIFEST = BUNDLE_TAR.parent / "bundle_manifest.json"

print("REPO_URL     :", REPO_URL)
print("GIT_TAG      :", GIT_TAG)
print("GIT_BRANCH   :", GIT_BRANCH)
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
        """# Cell 3: Minimal training dependencies
# Torch is supplied by Kaggle. Do not upgrade it here.
packages = [
    "transformers>=4.48,<6",
    "peft>=0.14",
    "accelerate>=0.34",
    "datasets>=2.19",
    "sympy>=1.12",
]
cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages]
print("Installing:", " ".join(packages))
subprocess.run(cmd, check=True)
import transformers, peft, accelerate
print("transformers:", transformers.__version__)
print("peft        :", peft.__version__)
print("accelerate  :", accelerate.__version__)
"""
    ),
    code(
        """# Cell 4: Fetch the pinned repository, or use the optional bundle
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()

shutil.rmtree(REPO_DIR, ignore_errors=True)

if BUNDLE_TAR.exists():
    if BUNDLE_MANIFEST.exists():
        manifest = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))
        expected = manifest.get("repo_bundle_sha256")
        actual = sha256_file(BUNDLE_TAR)
        print("Bundle SHA-256:", actual)
        if expected and actual != expected:
            raise RuntimeError(f"Bundle hash mismatch: got {actual}, expected {expected}")
        print("Bundle git head:", manifest.get("git_head"))
        print("Bundle data    :", manifest.get("data"))
    else:
        print("[WARN] bundle_manifest.json is missing; extracting without a hash check")

    print("Extracting bundle to /kaggle/working")
    with tarfile.open(BUNDLE_TAR, "r:gz") as archive:
        archive.extractall("/kaggle/working")
    source = "bundle"
else:
    print("Bundle not mounted; cloning the pinned tag.")
    clone = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", GIT_TAG, REPO_URL, str(REPO_DIR)],
        text=True,
    )
    if clone.returncode != 0:
        print("[WARN] Tag clone failed; falling back to the branch.")
        shutil.rmtree(REPO_DIR, ignore_errors=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, REPO_URL, str(REPO_DIR)],
            check=True,
            text=True,
        )
    source = "git"

if not (REPO_DIR / "experiments" / "kaggle" / "sft_v3" / "run_sft_kernel.py").exists():
    raise RuntimeError(f"Repository is incomplete at {REPO_DIR}")

os.chdir(REPO_DIR)
print("Source      :", source)
print("Repository  :", REPO_DIR)
print("Working dir :", Path.cwd())
if (REPO_DIR / ".git").exists():
    print("Git commit  :", subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_DIR
    ).strip())
    print("Git describe:", subprocess.run(
        ["git", "describe", "--tags", "--always", "--dirty"],
        text=True, cwd=REPO_DIR, capture_output=True
    ).stdout.strip())
"""
    ),
    code(
        """# Cell 5: Full hash, tokenizer, prompt-prefix, label-mask, and EOS preflight
# This reads every one of the 4,140 training rows and all 460 validation rows.
cmd = [
    sys.executable,
    "experiments/kaggle/sft_v3/run_sft_kernel.py",
    "--repo", str(REPO_DIR),
    "--mode", "check",
    "--output_root", str(OUTPUT_ROOT),
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, cwd=REPO_DIR, check=True)
"""
    ),
    code(
        """# Cell 6: Fresh SFT run with the strict generation check
# Effective batch = 2 x 8 = 16. Auto-resume is off by design.
# If this cell fails, do not package the adapter. Read run_manifest.json and the
# generation previews first.
cmd = [
    sys.executable,
    "experiments/kaggle/sft_v3/run_sft_kernel.py",
    "--repo", str(REPO_DIR),
    "--mode", "train",
    "--output_root", str(OUTPUT_ROOT),
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

for path in (receipt_path, run_manifest_path, bundle_path):
    if not path.exists():
        raise RuntimeError(f"Missing expected output: {path}")

receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

print("Receipt status :", receipt.get("status"))
print("Git head       :", receipt.get("git_head"))
print("Adapter dir    :", receipt.get("adapter_dir"))
print("Bundle         :", receipt.get("bundle"))
print("Bundle SHA-256 :", receipt.get("bundle_sha256"))
print("Train SHA-256  :", receipt.get("data", {}).get("train_sha256"))
print("Val SHA-256    :", receipt.get("data", {}).get("val_sha256"))
print("Generation      :")
for item in run_manifest.get("generation_results", []):
    print(json.dumps(item, ensure_ascii=False))

if receipt.get("status") != "completed":
    raise RuntimeError("Receipt is not completed")
if not run_manifest.get("generation_results"):
    raise RuntimeError("No generation checks were recorded")
for item in run_manifest["generation_results"]:
    if not item.get("terminal_answer") or not item.get("eos_emitted") or item.get("loop_marker"):
        raise RuntimeError(f"Strict generation check failed: {item}")

print("\\nOutput tree:")
for path in sorted(OUTPUT_ROOT.rglob("*")):
    if path.is_file():
        print(f"  {path.relative_to(OUTPUT_ROOT)}  ({path.stat().st_size} bytes)")
"""
    ),
    markdown(
        """## Acceptance gates

The adapter is admissible for the next stage only when all of these are true:

- `KERNEL_RECEIPT.json` has `"status": "completed"`.
- `data.train_sha256` is `5D05003B495DE3322568C4B40713CCE376A1AC05AC9B85A50B579A68A49FF63E`.
- `data.val_sha256` is `EB854AFF33DD31210D0B4BE61AF0165E5392F44B7B6D472C8C747161062E3EE3`.
- Every generation check has `terminal_answer: true`, `eos_emitted: true`, and
  `loop_marker: false`.
- `symbolm_v3_sft_adapter.tar.gz` exists and its SHA-256 is in the receipt.

If any gate fails, do not run the 4,800-cell evaluation. Fix the pipeline or report the
run as a pipeline failure.

## Next stage

1. Run the 100 x 4 x 6 diagnostic to catch format/EOS regressions.
2. If the diagnostic passes, run the preregistered 200 x 4 x 6 strict evaluation.
3. Report strict accuracy, J/correct, format validity, and the C-vs-D decision from
   `PREREG_VALUATION_V2.md`.

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

OUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[OK] Wrote {OUT}")
