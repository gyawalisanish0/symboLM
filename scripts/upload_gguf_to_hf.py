"""
Upload SymboLM GGUF Binaries to Hugging Face Hub
=================================================
Uploads official GGUF quantized models (q8_0 and bf16) to:
gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5

Principal Systems Architect & Research Director: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import argparse
from pathlib import Path
import sys
import time
from huggingface_hub import HfApi

if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def upload_gguf(repo_id: str = "gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5"):
    api = HfApi()
    user_info = api.whoami()
    username = user_info.get("name")
    print("=" * 70)
    print("SymboLM Official GGUF Upload Pipeline")
    print(f"Authenticated as : @{username}")
    print(f"Target Repo      : {repo_id}")
    print(f"Author           : Sanish Gyawali (Rupandehi, Nepal)")
    print("=" * 70)

    gguf_dir = Path("checkpoints/SymboLM-1.5B-v1.0-DSR1-Qwen2.5-GGUF").resolve()
    if not gguf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {gguf_dir}")

    # 1. Upload updated README.md with GGUF documentation
    readme_path = Path("checkpoints/SymboLM-1.5B-v1.0-DSR1-Qwen2.5/README.md").resolve()
    if readme_path.exists():
        print(f"\n[UPLOAD] Updating README.md model card on {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update Model Card with GGUF / Ollama / llama.cpp instructions",
        )
        print("[DONE] README.md updated!")

    gguf_files = [
        "symbolm-1.5b-v1.0-q4_0.gguf",
        "symbolm-1.5b-v1.0-q8_0.gguf",
        "symbolm-1.5b-v1.0-bf16.gguf",
    ]

    for fname in gguf_files:
        fpath = gguf_dir / fname
        if not fpath.exists():
            print(f"[WARN] File not found: {fname}, skipping.")
            continue

        fsize_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"\n[UPLOAD] Uploading {fname} ({fsize_mb:.2f} MB) to {repo_id}...")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add official GGUF model: {fname} (Sanish Gyawali)",
        )
        dur = time.time() - t0
        speed = fsize_mb / dur if dur > 0 else 0
        print(f"[DONE] {fname} successfully uploaded in {dur:.1f}s ({speed:.2f} MB/s)!")

    print("\n" + "=" * 70)
    print("All GGUF models are live on Hugging Face Hub!")
    print(f"Hub URL: https://huggingface.co/{repo_id}")
    print("=" * 70)


if __name__ == "__main__":
    upload_gguf()
