import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

def upload_stage3():
    api = HfApi()
    repo_id = "gyawalisanish0/symboLM-checkpoints"
    local_dir = Path("checkpoints/stage3_adapter")
    
    files_to_upload = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "README.md",
    ]
    
    print(f"Uploading Stage 3 Concept GRPO Adapter to {repo_id} (folder: stage3_concept_adapter)...")
    
    for f in files_to_upload:
        p = local_dir / f
        if p.exists():
            print(f"  Uploading {f} ({p.stat().st_size / (1024*1024):.2f} MB)...")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=f"stage3_concept_adapter/{f}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload SymboLM Stage 3 Concept GRPO adapter weights: {f}"
            )
            print(f"  [DONE] {f}")
        else:
            print(f"  [SKIP] {f} (not found)")
            
    print("\n[COMPLETE] Stage 3 Concept GRPO adapter successfully persisted to Hugging Face Hub!")

if __name__ == "__main__":
    upload_stage3()
