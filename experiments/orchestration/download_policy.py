import json
import urllib.request
import os
import base64
import sys

def main():
    kaggle_path = os.path.expanduser('~/.kaggle/kaggle.json')
    with open(kaggle_path) as f:
        creds = json.load(f)

    user = creds['username']
    key = creds['key']
    kernel = "sanishgyawali/symbolm-stage-2-grpo-training-dual-tesla-t4"
    url = f"https://www.kaggle.com/api/v1/kernels/output?userName=sanishgyawali&kernelSlug=symbolm-stage-2-grpo-training-dual-tesla-t4"

    req = urllib.request.Request(url)
    auth_str = f"{user}:{key}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    req.add_header('Authorization', f"Basic {b64_auth}")

    print("Fetching file list from Kaggle API...")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    
    files = data.get('files', [])
    print(f"Found {len(files)} output files.")
    for f in files:
        fname = f.get('fileName')
        furl = f.get('url')
        print(f"File: {fname} -> {furl[:80]}...")

    # Look for symboLM_grpo_final_policy.tar.gz or adapter files
    target_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "grpo_download")
    os.makedirs(target_dir, exist_ok=True)

    for f in files:
        fname = f.get('fileName')
        furl = f.get('url')
        if fname in ['symboLM_grpo_final_policy.tar.gz', 'adapter_model.safetensors', 'adapter_config.json']:
            dest = os.path.join(target_dir, fname)
            print(f"Downloading {fname} to {dest}...")
            urllib.request.urlretrieve(furl, dest)
            print(f"Downloaded {fname} ({os.path.getsize(dest)} bytes)")

if __name__ == "__main__":
    main()
