"""
Download Cross-Family LLaMA-3.2 Transfer Results from Kaggle
============================================================
Author: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import sys
from pathlib import Path
import kaggle

KERNEL_SLUG = "sanishgyawali/symbolm-cross-family-transfer-meta-llama-3-2-1b"
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TARGET_DIR = ROOT_DIR / "eval" / "cross_family_llama"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

api = kaggle.KaggleApi()
api.authenticate()

status = api.kernels_status(KERNEL_SLUG)
print(f"Current Kernel Status: {status}")

if status.get("status") == "COMPLETE":
    print(f"Downloading kernel output to {TARGET_DIR}...")
    api.kernels_output(KERNEL_SLUG, path=str(TARGET_DIR))
    print("[OK] Output downloaded successfully!")
    for f in TARGET_DIR.iterdir():
        print(f"  • {f.name} ({f.stat().st_size / 1e3:.1f} KB)")
else:
    print(f"Kernel is still running or pending ({status.get('status')}). Please wait.")
