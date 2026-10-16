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


def parse_args():
    parser = argparse.ArgumentParser(description="Upload SymboLM v2.1 GGUF Binaries to Hugging Face Hub")
    parser.add_argument(
        "--model_type",
        type=str,
        default="deepseek",
        choices=["deepseek", "smollm", "all"],
        help="Target model GGUF suite to upload",
    )
    return parser.parse_args()


def upload_gguf_suite(repo_id: str, gguf_dir: Path, files: list[str], model_label: str):
    api = HfApi()
    user_info = api.whoami()
    username = user_info.get("name")
    print("=" * 75)
    print(f"SymboLM v2.1 Official GGUF Upload Pipeline — {model_label}")
    print(f"Authenticated as : @{username}")
    print(f"Target Repo      : {repo_id}")
    print(f"Author           : Sanish Gyawali (Rupandehi, Nepal)")
    print("=" * 75)

    if not gguf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {gguf_dir}")

    existing_files = set(api.list_repo_files(repo_id))

    for fname in files:
        fpath = gguf_dir / fname
        if not fpath.exists():
            print(f"[WARN] File not found: {fname}, skipping.")
            continue

        if fname in existing_files:
            print(f"[SKIP] {fname} already exists on {repo_id}. Skipping.")
            continue

        fsize_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"\n[UPLOAD] Uploading {fname} ({fsize_mb:.2f} MB / {fsize_mb/1024:.2f} GB) to {repo_id}...")

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"release: upload official SymboLM v2.1 GGUF {fname} (Sanish Gyawali)",
                )
                dur = time.time() - t0
                speed = fsize_mb / dur if dur > 0 else 0
                print(f"[DONE] {fname} successfully uploaded in {dur:.1f}s ({speed:.2f} MB/s)!")
                break
            except Exception as e:
                print(f"[RETRY] Upload attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    raise
                time.sleep(5 * attempt)

    print("\n" + "=" * 75)
    print(f"[SUCCESS] All {model_label} GGUF models are live on Hugging Face Hub!")
    print(f"Hub URL: https://huggingface.co/{repo_id}")
    print("=" * 75)


def main():
    args = parse_args()

    if args.model_type in ("deepseek", "all"):
        repo_id = "gyawalisanish0/SymboLM-2.1-DeepSeek-1.5B"
        gguf_dir = Path("checkpoints/SymboLM-2.1-DeepSeek-1.5B-GGUF").resolve()
        files = [
            "symbolm-2.1-deepseek-1.5b-q4_0.gguf",
            "symbolm-2.1-deepseek-1.5b-q8_0.gguf",
            "symbolm-2.1-deepseek-1.5b-bf16.gguf",
        ]
        upload_gguf_suite(repo_id, gguf_dir, files, "DeepSeek-1.5B")

    if args.model_type in ("smollm", "all"):
        repo_id = "gyawalisanish0/SymboLM-2.1-SmolLM2-1.7B"
        gguf_dir = Path("checkpoints/SymboLM-2.1-SmolLM2-1.7B-GGUF").resolve()
        files = [
            "symbolm-2.1-smollm2-1.7b-q4_0.gguf",
            "symbolm-2.1-smollm2-1.7b-q8_0.gguf",
            "symbolm-2.1-smollm2-1.7b-bf16.gguf",
        ]
        upload_gguf_suite(repo_id, gguf_dir, files, "SmolLM2-1.7B")


if __name__ == "__main__":
    main()
