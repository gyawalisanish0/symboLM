import os
import sys
import time
import tarfile
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

def fetch_fresh_file_map(kernel_slug="symbolm-stage-3-concept-grpo-training"):
    api = KaggleApi()
    api.authenticate()

    with api.build_kaggle_client() as kaggle:
        token = None
        all_files = []
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = "sanishgyawali"
            req.kernel_slug = kernel_slug
            req.page_size = 100
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            all_files.extend(resp.files)
            if not resp.next_page_token:
                break
            token = resp.next_page_token

    file_dict = {f.file_name: f.url for f in all_files}
    return file_dict

def download_file_resumable(file_name, target_path, file_map_getter, max_retries=20):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = target_path + ".tmp"

    file_map = file_map_getter()
    if file_name not in file_map:
        print(f"[ERROR] {file_name} not found in kernel session output!")
        return False
    url = file_map[file_name]

    # Get total size
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
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as r:
                if r.status_code == 416:  # Range not satisfiable, file fully downloaded
                    break
                if r.status_code == 403 or r.status_code == 401:  # URL expired
                    print("[REFRESH] URL expired, fetching fresh signed URL from Kaggle...")
                    file_map = file_map_getter()
                    url = file_map[file_name]
                    continue

                if existing_size > 0 and r.status_code != 206:
                    print("[WARN] Server did not return partial 206, restarting from 0...")
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

            final_size = os.path.getsize(temp_path)
            if total_size == 0 or final_size >= total_size:
                break
        except (requests.exceptions.RequestException, TimeoutError, Exception) as e:
            print(f"\n[RETRY] Network glitch ({type(e).__name__}: {e}), retrying in 3s...")
            time.sleep(3)
            # Re-fetch fresh map in case URL needs refresh
            try:
                file_map = file_map_getter()
                url = file_map.get(file_name, url)
            except Exception:
                pass

    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(temp_path, target_path)
    final_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"\n[DONE] Saved {os.path.basename(target_path)} ({final_mb:.2f} MB)")
    return True

def main():
    dest_dir = r"c:\Users\user\Dev\symboLM\checkpoints\stage3_concept_adapter"
    os.makedirs(dest_dir, exist_ok=True)

    print("[STATUS] Querying files for symbolm-stage-3-concept-grpo-training...")
    file_map_cache = {}
    def get_map():
        nonlocal file_map_cache
        file_map_cache = fetch_fresh_file_map("symbolm-stage-3-concept-grpo-training")
        return file_map_cache

    current_map = get_map()
    print(f"[STATUS] Total files available in session: {len(current_map)}")

    targets = [
        ("symboLM/checkpoints/stage3_adapter/adapter_config.json", os.path.join(dest_dir, "adapter_config.json")),
        ("symboLM/checkpoints/stage3_adapter/chat_template.jinja", os.path.join(dest_dir, "chat_template.jinja")),
        ("symboLM/checkpoints/stage3_adapter/tokenizer_config.json", os.path.join(dest_dir, "tokenizer_config.json")),
        ("symboLM/checkpoints/stage3_adapter/tokenizer.json", os.path.join(dest_dir, "tokenizer.json")),
        ("symboLM/checkpoints/stage3_adapter/adapter_model.safetensors", os.path.join(dest_dir, "adapter_model.safetensors")),
    ]

    for remote_name, local_path in targets:
        # Check if already fully downloaded
        if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
            print(f"[SKIP] {os.path.basename(local_path)} already exists ({os.path.getsize(local_path)/(1024*1024):.1f} MB)")
            continue
        download_file_resumable(remote_name, local_path, get_map)

    print(f"\n[COMPLETED] Stage 3.5 Concept GRPO policy downloaded cleanly to {dest_dir}!")

if __name__ == "__main__":
    main()
