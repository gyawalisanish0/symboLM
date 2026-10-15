"""
SymboLM v2.1 Official Hugging Face Release Pipeline
===================================================
Uploads the trained SymboLM v2.1 policy weights, tokenizer configs, GBNF grammar,
and verified model cards to Hugging Face Model Hub:
  - gyawalisanish0/SymboLM-2.1-DeepSeek-1.5B (Flagship)
  - gyawalisanish0/SymboLM-2.1-SmolLM2-1.7B (Cross-Architecture LLaMA)

Author & Principal Systems Architect: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Upload SymboLM v2.1 to Hugging Face Hub")
    parser.add_argument(
        "--model_type",
        type=str,
        default="deepseek",
        choices=["deepseek", "smollm", "all"],
        help="Target model release to upload",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="checkpoints/symbolm_v2_adapter",
        help="Local directory containing the v2.1 adapter weights",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate files and repo connection without uploading",
    )
    return parser.parse_args()


def upload_deepseek_release(api: HfApi, adapter_dir: Path, dry_run: bool = False):
    repo_id = "gyawalisanish0/SymboLM-2.1-DeepSeek-1.5B"
    print("\n" + "=" * 75)
    print(f"HUGGING FACE RELEASE: {repo_id}")
    print("=" * 75)

    # 1. Verify required files
    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    card_path = Path("research/HUGGING_FACE_MODEL_CARD_V2.1_DEEPSEEK.md")
    gbnf_path = Path("symbolic/symbolm.gbnf")

    assert weights_path.exists(), f"CRITICAL: {weights_path} not found!"
    assert config_path.exists(), f"CRITICAL: {config_path} not found!"
    assert card_path.exists(), f"CRITICAL: {card_path} not found!"

    print(f"[OK] Adapter Weights : {weights_path} ({weights_path.stat().st_size / 1e6:.2f} MB)")
    print(f"[OK] Adapter Config  : {config_path}")
    print(f"[OK] Model Card      : {card_path}")
    if gbnf_path.exists():
        print(f"[OK] GBNF Grammar    : {gbnf_path}")

    if dry_run:
        print("[DRY-RUN] Files validated. Skipping network upload.")
        return

    # 2. Ensure repository exists
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        print(f"[OK] Confirmed Hugging Face repository exists: {repo_id}")
    except Exception as e:
        print(f"[INFO] Repo creation check: {e}")

    # 3. Upload Model Card as README.md
    print("[UPLOADING] Model Card (README.md)...")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="docs: publish official SymboLM v2.1 model card with 1,000-question scorecard"
    )

    # 4. Upload GBNF Grammar
    if gbnf_path.exists():
        print("[UPLOADING] GBNF Grammar (symbolm.gbnf)...")
        api.upload_file(
            path_or_fileobj=str(gbnf_path),
            path_in_repo="symbolm.gbnf",
            repo_id=repo_id,
            repo_type="model",
            commit_message="feat: publish native GBNF grammar for llama.cpp / Ollama edge deployment"
        )

    # 5. Upload Adapter Files
    files_to_upload = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json"
    ]

    for fname in files_to_upload:
        fpath = adapter_dir / fname
        if fpath.exists():
            print(f"[UPLOADING] {fname} ({fpath.stat().st_size / 1e6:.2f} MB)...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"release: upload SymboLM v2.1 {fname}"
            )

    print("\n" + "=" * 75)
    print(f"[SUCCESS] SymboLM v2.1 Flagship Adapter is live on Hugging Face!")
    print(f"URL: https://huggingface.co/{repo_id}")
    print("=" * 75)


def upload_smollm_release(api: HfApi, adapter_dir: Path, dry_run: bool = False):
    repo_id = "gyawalisanish0/SymboLM-2.1-SmolLM2-1.7B"
    print("\n" + "=" * 75)
    print(f"HUGGING FACE RELEASE: {repo_id}")
    print("=" * 75)

    # 1. Verify required files
    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    card_path = Path("research/HUGGING_FACE_MODEL_CARD_V2.1_SMOLLM.md")
    gbnf_path = Path("symbolic/symbolm.gbnf")

    assert weights_path.exists(), f"CRITICAL: {weights_path} not found!"
    assert config_path.exists(), f"CRITICAL: {config_path} not found!"
    assert card_path.exists(), f"CRITICAL: {card_path} not found!"

    print(f"[OK] Adapter Weights : {weights_path} ({weights_path.stat().st_size / 1e6:.2f} MB)")
    print(f"[OK] Adapter Config  : {config_path}")
    print(f"[OK] Model Card      : {card_path}")
    if gbnf_path.exists():
        print(f"[OK] GBNF Grammar    : {gbnf_path}")

    if dry_run:
        print("[DRY-RUN] Files validated. Skipping network upload.")
        return

    # 2. Ensure repository exists
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        print(f"[OK] Confirmed Hugging Face repository exists: {repo_id}")
    except Exception as e:
        print(f"[INFO] Repo creation check: {e}")

    # 3. Upload Model Card as README.md
    print("[UPLOADING] Model Card (README.md)...")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="docs: publish official SymboLM v2.1 SmolLM2-1.7B cross-family model card"
    )

    # 4. Upload GBNF Grammar
    if gbnf_path.exists():
        print("[UPLOADING] GBNF Grammar (symbolm.gbnf)...")
        api.upload_file(
            path_or_fileobj=str(gbnf_path),
            path_in_repo="symbolm.gbnf",
            repo_id=repo_id,
            repo_type="model",
            commit_message="feat: publish native GBNF grammar for llama.cpp / Ollama edge deployment"
        )

    # 5. Upload Adapter Files
    files_to_upload = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json"
    ]

    for fname in files_to_upload:
        fpath = adapter_dir / fname
        if fpath.exists():
            print(f"[UPLOADING] {fname} ({fpath.stat().st_size / 1e6:.2f} MB)...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"release: upload SymboLM v2.1 SmolLM2-1.7B {fname}"
            )

    print("\n" + "=" * 75)
    print(f"[SUCCESS] SymboLM v2.1 SmolLM2-1.7B Adapter is live on Hugging Face!")
    print(f"URL: https://huggingface.co/{repo_id}")
    print("=" * 75)


def main():
    args = parse_args()
    api = HfApi()

    # Verify authentication
    try:
        user = api.whoami()
        print(f"[AUTH] Authenticated as Hugging Face user: {user.get('name')}")
    except Exception as e:
        print(f"[ERROR] Hugging Face authentication failed: {e}")
        sys.exit(1)

    adapter_dir = Path(args.adapter_dir)
    if args.model_type in ("deepseek", "all"):
        upload_deepseek_release(api, adapter_dir, args.dry_run)

    if args.model_type in ("smollm", "all"):
        smollm_dir = Path("eval/cross_family_llama/symbolm_llama_1_7b_adapter")
        upload_smollm_release(api, smollm_dir, args.dry_run)


if __name__ == "__main__":
    main()
