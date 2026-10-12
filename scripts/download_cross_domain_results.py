"""
Downloads and parses the cross-domain benchmark results from Kaggle.
Author: Sanish Gyawali
"""
import os
import sys
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    kernel_slug = "sanishgyawali/symbolm-cross-domain-200-benchmark"
    dest_dir = Path("eval_cross_domain")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checking status for: {kernel_slug}")
    status = api.kernels_status(kernel_slug)
    status_str = getattr(status, "status", str(status))
    print(f"Status: {status_str}")
    
    if str(status_str).lower() == "complete":
        print(f"Downloading output files to {dest_dir}...")
        api.kernels_output(kernel_slug, path=str(dest_dir))
        print("Download complete! Files:")
        for f in dest_dir.iterdir():
            print(f"  {f.name} ({f.stat().st_size} bytes)")
    else:
        print(f"Kernel is still: {status_str}")

if __name__ == "__main__":
    main()
