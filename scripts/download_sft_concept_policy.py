import os
import sys
import time
import tarfile
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

def download_file_resumable(url, target_path, max_retries=10):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = target_path + ".tmp"
    
    # Get total size first
    total_size = 0
    try:
        head_resp = requests.head(url, timeout=15, allow_redirects=True)
        total_size = int(head_resp.headers.get("content-length", 0))
    except Exception:
        pass

    for attempt in range(max_retries):
        existing_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
        if total_size > 0 and existing_size >= total_size:
            break

        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            print(f"\n[RESUME] Resuming {os.path.basename(target_path)} from {existing_size / (1024*1024):.1f} MB (attempt {attempt+1}/{max_retries})...")
        else:
            print(f"\n[DOWNLOAD] Starting {os.path.basename(target_path)} (attempt {attempt+1}/{max_retries})...")

        try:
            with requests.get(url, headers=headers, stream=True, timeout=(10, 30)) as r:
                if r.status_code == 416: # Range not satisfiable, file fully downloaded
                    break
                if existing_size > 0 and r.status_code != 206:
                    # Server doesn't support range, start from scratch
                    print("[WARN] Server does not support resume, restarting from 0...")
                    existing_size = 0
                    mode = "wb"
                else:
                    mode = "ab" if existing_size > 0 else "wb"

                if total_size == 0:
                    content_length = int(r.headers.get("content-length", 0))
                    total_size = existing_size + content_length

                downloaded = existing_size
                with open(temp_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pct = (downloaded / total_size * 100.0) if total_size > 0 else 0
                            mb = downloaded / (1024 * 1024)
                            t_mb = total_size / (1024 * 1024) if total_size > 0 else 0
                            sys.stdout.write(f"\r  [{mb:.1f} MB / {t_mb:.1f} MB] ({pct:.1f}%)")
                            sys.stdout.flush()

            # If we reached here without exception, verify size
            final_size = os.path.getsize(temp_path)
            if total_size == 0 or final_size >= total_size:
                break
        except (requests.exceptions.RequestException, TimeoutError) as e:
            print(f"\n[RETRY] Network glitch ({e}), retrying in 3s...")
            time.sleep(3)

    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(temp_path, target_path)
    final_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"\n[DONE] Saved {os.path.basename(target_path)} ({final_mb:.2f} MB)")

def main():
    api = KaggleApi()
    api.authenticate()

    with api.build_kaggle_client() as kaggle:
        token = None
        all_files = []
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = "sanishgyawali"
            req.kernel_slug = "symbolm-stage-2-5-concept-sft-seeding"
            req.page_size = 100
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            all_files.extend(resp.files)
            if not resp.next_page_token:
                break
            token = resp.next_page_token

    print(f"[STATUS] Total files available in Stage 2.5 kernel session: {len(all_files)}")

    dest_dir = r"c:\Users\user\Dev\symboLM\checkpoints\sft_concept_adapter"
    os.makedirs(dest_dir, exist_ok=True)

    target_map = {
        "symboLM/checkpoints/sft_concept_adapter/adapter_model.safetensors": os.path.join(dest_dir, "adapter_model.safetensors"),
        "symboLM/checkpoints/sft_concept_adapter/adapter_config.json": os.path.join(dest_dir, "adapter_config.json"),
        "symboLM/checkpoints/sft_concept_adapter/chat_template.jinja": os.path.join(dest_dir, "chat_template.jinja"),
        "symboLM/checkpoints/sft_concept_adapter/tokenizer.json": os.path.join(dest_dir, "tokenizer.json"),
        "symboLM/checkpoints/sft_concept_adapter/tokenizer_config.json": os.path.join(dest_dir, "tokenizer_config.json"),
        "symboLM_sft_concept_policy.tar.gz": os.path.join(dest_dir, "symboLM_sft_concept_policy.tar.gz"),
    }

    found = 0
    for f in all_files:
        if f.file_name in target_map:
            out_path = target_map[f.file_name]
            # Skip if already exists and non-empty
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                print(f"[EXISTS] {os.path.basename(out_path)} already downloaded.")
                found += 1
                continue
            download_file_resumable(f.url, out_path)
            found += 1

    print(f"\n[COMPLETED] Successfully synced {found} Stage 2.5 policy files to {dest_dir}.")

if __name__ == "__main__":
    main()
