import os
import sys
import time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

def download_file(url, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = target_path + ".tmp"
    print(f"Downloading to {target_path}...")
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
            req.kernel_slug = "symbolm-stage-2-grpo-training-dual-tesla-t4"
            req.page_size = 100
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            all_files.extend(resp.files)
            if not resp.next_page_token:
                break
            token = resp.next_page_token

    print(f"Total files available on Kaggle: {len(all_files)}")

    dest_dir = r"c:\Users\user\Dev\symboLM\checkpoints\grpo_download"
    target_names = {
        "symboLM_grpo_final_policy.tar.gz": os.path.join(dest_dir, "symboLM_grpo_final_policy.tar.gz"),
        "symboLM/checkpoints/grpo_adapter/adapter_model.safetensors": os.path.join(dest_dir, "adapter_model.safetensors"),
        "symboLM/checkpoints/grpo_adapter/adapter_config.json": os.path.join(dest_dir, "adapter_config.json"),
        "symboLM/checkpoints/grpo_adapter/chat_template.jinja": os.path.join(dest_dir, "chat_template.jinja"),
        "symboLM/checkpoints/grpo_adapter/tokenizer.json": os.path.join(dest_dir, "tokenizer.json"),
        "symboLM/checkpoints/grpo_adapter/tokenizer_config.json": os.path.join(dest_dir, "tokenizer_config.json"),
    }

    for f in all_files:
        if f.file_name in target_names:
            out_path = target_names[f.file_name]
            download_file(f.url, out_path)

if __name__ == "__main__":
    main()
