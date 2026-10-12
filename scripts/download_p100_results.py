"""
Downloads the 1,000-question multi-year benchmark results from Kaggle Tesla P100 run.
Author: Sanish Gyawali
"""
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    kernel_slug = "sanishgyawali/symbolm-1000-p100-benchmark"
    dest_dir = Path("eval")
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checking status for: {kernel_slug}")
    status = api.kernels_status(kernel_slug)
    status_str = getattr(status, "status", str(status))
    print(f"Status: {status_str}")

    if "complete" in str(status_str).lower():
        print(f"Downloading output files to {dest_dir}...")
        api.kernels_output(kernel_slug, path=str(dest_dir))
        print("Download complete! Files in eval directory:")
        for f in dest_dir.iterdir():
            if f.is_file():
                print(f"  • {f.name} ({f.stat().st_size} bytes)")
    else:
        print(f"Kernel is still: {status_str}")

if __name__ == "__main__":
    main()
