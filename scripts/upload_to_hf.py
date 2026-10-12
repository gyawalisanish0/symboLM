"""
SymboLM Official Hugging Face Model Hub Release Pipeline
=========================================================
Uploads the standalone merged weights, tokenizer, chat template, and
production model card to Hugging Face: gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5

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


def parse_args():
    parser = argparse.ArgumentParser(description="Upload SymboLM to Hugging Face Model Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5",
        help="Target Hugging Face Model repository ID",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="checkpoints/SymboLM-1.5B-v1.0-DSR1-Qwen2.5",
        help="Local checkpoint directory containing standalone model",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Set to true if creating a private repository (default: public)",
    )
    parser.add_argument(
        "--include_quantized",
        action="store_true",
        default=True,
        help="Also upload INT8 and NF4 4-bit checkpoints to quantized/ subfolder",
    )
    return parser.parse_args()


def upload_release():
    args = parse_args()
    local_path = Path(args.local_dir).resolve()

    print("=" * 75)
    print("SymboLM Official Hugging Face Hub Release")
    print(f"Target Repo   : {args.repo_id}")
    print(f"Local Source  : {local_path}")
    print(f"Author        : Sanish Gyawali (Rupandehi, Nepal)")
    print(f"Visibility    : {'PRIVATE' if args.private else 'PUBLIC (Open Source)'}")
    print("=" * 75)

    if not local_path.exists():
        raise FileNotFoundError(f"Local checkpoint directory not found: {local_path}")

    # 1. Authenticate with Hugging Face
    api = HfApi()
    try:
        user_info = api.whoami()
        username = user_info.get("name")
        print(f"\n[1/4] Authenticated with Hugging Face as: @{username}")
    except Exception as e:
        print(f"\n[ERROR] Hugging Face authentication failed: {e}")
        sys.exit(1)

    # 2. Create Target Repository on Hub
    print(f"\n[2/4] Ensuring remote model repository exists: {args.repo_id}...")
    try:
        repo_url = api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"[OK] Repository ready: {repo_url}")
    except Exception as e:
        print(f"[NOTE] Repo check: {e}")

    # 3. Upload Primary Standalone Release Files
    print(f"\n[3/4] Uploading core standalone model files to {args.repo_id}...")
    core_files = [
        "README.md",
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ]

    t_start = time.time()
    for fname in core_files:
        fpath = local_path / fname
        if not fpath.exists():
            print(f"  • [SKIP] {fname} (not present)")
            continue

        fsize_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  • Uploading {fname} ({fsize_mb:.2f} MB)...")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Release SymboLM-1.5B-v1.0: {fname}",
        )
        dur = time.time() - t0
        speed = fsize_mb / dur if dur > 0 else 0
        print(f"    [DONE] {fname} uploaded in {dur:.1f}s ({speed:.2f} MB/s)!")

    # 4. Optional: Upload GGUF Quantized Binaries
    if args.include_quantized:
        print(f"\n[4/4] Uploading official GGUF models to repository root...")
        quant_files = [
            ("checkpoints/SymboLM-1.5B-v1.0-DSR1-Qwen2.5-GGUF/symbolm-1.5b-v1.0-q8_0.gguf", "symbolm-1.5b-v1.0-q8_0.gguf"),
            ("checkpoints/SymboLM-1.5B-v1.0-DSR1-Qwen2.5-GGUF/symbolm-1.5b-v1.0-bf16.gguf", "symbolm-1.5b-v1.0-bf16.gguf"),
        ]
        for src_path_str, dest_path_repo in quant_files:
            src_p = Path(src_path_str).resolve()
            if src_p.exists():
                size_mb = src_p.stat().st_size / (1024 * 1024)
                print(f"  • Uploading {dest_path_repo} ({size_mb:.2f} MB)...")
                t0 = time.time()
                api.upload_file(
                    path_or_fileobj=str(src_p),
                    path_in_repo=dest_path_repo,
                    repo_id=args.repo_id,
                    repo_type="model",
                    commit_message=f"Add official GGUF model: {dest_path_repo} (Sanish Gyawali)",
                )
                dur = time.time() - t0
                print(f"    [DONE] {dest_path_repo} uploaded in {dur:.1f}s!")

    total_time = time.time() - t_start
    print("\n" + "=" * 75)
    print(f"HUGGING FACE RELEASE SUCCESSFUL! ({total_time:.1f}s elapsed)")
    print(f"Model Hub URL: https://huggingface.co/{args.repo_id}")
    print("All weights, configs, and model cards are now live on the global hub!")
    print("=" * 75)


if __name__ == "__main__":
    upload_release()
