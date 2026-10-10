import os
import sys
import time
import tarfile
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

def download_file(url, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = target_path + ".tmp"
    print(f"\n[DOWNLOAD] Starting {os.path.basename(target_path)}...")
    t0 = time.time()
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100.0
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r  [{mb:.1f} MB / {total_mb:.1f} MB] ({pct:.1f}%)")
                    else:
                        mb = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\r  [{mb:.1f} MB downloaded]")
                    sys.stdout.flush()
    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(temp_path, target_path)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(target_path) / (1024 * 1024)
    speed = size_mb / elapsed if elapsed > 0 else 0
    print(f"\n[DONE] Saved {os.path.basename(target_path)} ({size_mb:.2f} MB in {elapsed:.1f}s, {speed:.2f} MB/s)")

def main():
    api = KaggleApi()
    api.authenticate()

    with api.build_kaggle_client() as kaggle:
        token = None
        all_files = []
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = "sanishgyawali"
            req.kernel_slug = "symbolm-stage-3-concept-grpo-training"
            req.page_size = 100
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            all_files.extend(resp.files)
            if not resp.next_page_token:
                break
            token = resp.next_page_token

    print(f"[STATUS] Total files available in Stage 3 kernel session: {len(all_files)}")

    dest_dir = r"c:\Users\user\Dev\symboLM\checkpoints\stage3_adapter"
    os.makedirs(dest_dir, exist_ok=True)

    target_map = {
        "symboLM/checkpoints/stage3_adapter/adapter_model.safetensors": os.path.join(dest_dir, "adapter_model.safetensors"),
        "symboLM/checkpoints/stage3_adapter/adapter_config.json": os.path.join(dest_dir, "adapter_config.json"),
        "symboLM/checkpoints/stage3_adapter/chat_template.jinja": os.path.join(dest_dir, "chat_template.jinja"),
        "symboLM/checkpoints/stage3_adapter/tokenizer.json": os.path.join(dest_dir, "tokenizer.json"),
        "symboLM/checkpoints/stage3_adapter/tokenizer_config.json": os.path.join(dest_dir, "tokenizer_config.json"),
        "symboLM_stage3_final_policy.tar.gz": os.path.join(dest_dir, "symboLM_stage3_final_policy.tar.gz"),
    }

    found = 0
    for f in all_files:
        if f.file_name in target_map:
            out_path = target_map[f.file_name]
            download_file(f.url, out_path)
            found += 1

    print(f"\n[COMPLETED] Successfully downloaded {found} Stage 3 policy files to {dest_dir}.")

    tarball = os.path.join(dest_dir, "symboLM_stage3_final_policy.tar.gz")
    safetensor = os.path.join(dest_dir, "adapter_model.safetensors")
    if os.path.exists(tarball) and (not os.path.exists(safetensor) or os.path.getsize(safetensor) == 0):
        print(f"[EXTRACT] Extracting {tarball} into {dest_dir}...")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        print("[EXTRACT] Done.")

if __name__ == "__main__":
    main()
