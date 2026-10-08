"""
SymboLM Platform Manager & Automation Helper
============================================
Author: Sanish Gyawali
Checks authentication and automates deployment across:
- Kaggle (30h/week free GPU)
- Lightning AI (Persistent cloud development studio)
- Google Colab
"""

import os
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

LOCAL_BIN = Path.home() / ".local" / "bin"
KAGGLE_EXE = LOCAL_BIN / "kaggle.exe" if sys.platform == "win32" else LOCAL_BIN / "kaggle"
LIGHTNING_EXE = LOCAL_BIN / "lightning.exe" if sys.platform == "win32" else LOCAL_BIN / "lightning"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


def check_status():
    print("=" * 60)
    print("🚀 SymboLM Cloud Platform Status Check")
    print("=" * 60)

    # 1. Check Kaggle
    print("\n[1] Kaggle CLI Status:")
    if KAGGLE_EXE.exists():
        print(f"  ✓ Binary installed: {KAGGLE_EXE}")
        if KAGGLE_JSON.exists():
            print(f"  ✓ API Token found: {KAGGLE_JSON}")
        else:
            print(f"  ⚠ API Token missing: {KAGGLE_JSON}")
            print("    -> Go to https://www.kaggle.com/settings -> Create New Token")
            print(f"    -> Save kaggle.json to: {Path.home() / '.kaggle'}")
    else:
        print("  ✗ Kaggle CLI not found. Run: uv tool install kaggle")

    # 2. Check Lightning AI
    print("\n[2] Lightning AI CLI Status:")
    if LIGHTNING_EXE.exists():
        print(f"  ✓ Binary installed: {LIGHTNING_EXE}")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        try:
            res = subprocess.run([str(LIGHTNING_EXE), "user", "whoami"], capture_output=True, text=True, env=env)
            if res.returncode == 0 and "username" in res.stdout.lower():
                print(f"  ✓ Authenticated as: {res.stdout.strip()}")
            else:
                print("  ℹ Not yet logged in.")
                print("    -> Run: lightning login")
        except Exception:
            print("  ℹ Run 'lightning login' to authenticate.")
    else:
        print("  ✗ Lightning CLI not found. Run: uv tool install lightning-sdk")

    print("\n" + "=" * 60)


def push_to_kaggle():
    if not KAGGLE_JSON.exists():
        print(f"Error: Kaggle credentials not found at {KAGGLE_JSON}")
        print("Please place kaggle.json in that folder first.")
        return

    kaggle_dir = Path(__file__).resolve().parent.parent / "kaggle"
    print(f"Pushing Kaggle notebook from {kaggle_dir}...")
    subprocess.run([str(KAGGLE_EXE), "kernels", "push", "-p", str(kaggle_dir)])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "push-kaggle":
        push_to_kaggle()
    else:
        check_status()
