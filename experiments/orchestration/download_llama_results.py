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
status_str = str(getattr(status, "status", status))
print(f"Current Kernel Status: {status_str}")

if "COMPLETE" in status_str:
    res_file = TARGET_DIR / "llama_1_7b_transfer_results.json"
    if res_file.exists():
        print(f"[OK] Output files already downloaded to {TARGET_DIR}!")
    else:
        print(f"Downloading kernel output to {TARGET_DIR}...")
        try:
            api.kernels_output(KERNEL_SLUG, path=str(TARGET_DIR))
        except UnicodeEncodeError:
            # On Windows cp1252, Kaggle SDK writes raw log using system code page; binary outputs are preserved
            pass
        # Clean up any cloned repo subfolder if present
        clone_sub = TARGET_DIR / "symboLM"
        if clone_sub.exists():
            import shutil
            shutil.rmtree(clone_sub, ignore_errors=True)
        print("[OK] Output downloaded successfully!")

    for f in TARGET_DIR.iterdir():
        if f.is_file():
            print(f"  • {f.name} ({f.stat().st_size / 1e3:.1f} KB)")
        elif f.is_dir():
            print(f"  • [DIR] {f.name}/")
else:
    print(f"Kernel is still running or pending ({status_str}). Please wait.")
