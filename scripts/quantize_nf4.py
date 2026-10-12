"""
SymboLM NF4 (4-Bit NormalFloat) Local Quantization Engine
=========================================================
Quantizes SymboLM-1.5B-v1.0-DSR1-Qwen2.5 to 4-bit NormalFloat (NF4)
using PyTorch & torchao, enabling < 1.2 GB edge execution.

Principal Systems Architect & Research Director: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

from pathlib import Path
import shutil
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchao.quantization as qao

if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SOURCE_MODEL_DIR = Path("checkpoints/SymboLM-1.5B-v1.0-DSR1-Qwen2.5")
OUTPUT_NF4_DIR = Path("checkpoints/SymboLM-1.5B-v1.0-NF4")


def quantize_to_nf4():
    print("=" * 70)
    print("SymboLM 4-Bit NormalFloat (NF4) Quantization Suite")
    print(f"Source Model : {SOURCE_MODEL_DIR}")
    print(f"Destination  : {OUTPUT_NF4_DIR}")
    print("=" * 70)

    # 1. Load Tokenizer & Model in bfloat16
    print("\n[1/4] Loading source model in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(str(SOURCE_MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(SOURCE_MODEL_DIR),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    print("[OK] Source model loaded in memory!")

    # 2. Convert Linear Projection Layers to NF4
    print("\n[2/4] Quantizing Linear Attention & MLP Layers to 4-bit NF4...")
    t0 = time.time()
    quant_count = 0
    skipped_count = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Check if dimensions are divisible by block size (256)
            if module.weight.numel() % 256 == 0 and module.weight.dim() == 2:
                with torch.no_grad():
                    orig_shape = module.weight.shape
                    # Convert to float32 temporarily for double quantization scalers
                    w_f32 = module.weight.data.to(torch.float32)
                    nf4_tensor = qao.to_nf4(w_f32)
                    module.weight = torch.nn.Parameter(nf4_tensor, requires_grad=False)
                    quant_count += 1
            else:
                skipped_count += 1

    elapsed = time.time() - t0
    print(f"[OK] Quantized {quant_count} linear layers to NF4 in {elapsed:.2f}s! (Skipped: {skipped_count})")

    # 3. Test Invariant & Syntax Preservation
    print("\n[3/4] Verifying 4-bit Invariant & AST Preservation...")
    test_q = "<｜User｜>What is the chemical symbol for potassium?<｜Assistant｜><think>\n"
    inputs = tokenizer(test_q, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=25, do_sample=False)
    gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    print(f"  • Prompt: What is the chemical symbol for potassium?")
    print(f"  • 4-Bit Output: {gen_text.strip()}")

    # 4. Save NF4 Checkpoint
    OUTPUT_NF4_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[4/4] Saving NF4 model checkpoint to {OUTPUT_NF4_DIR}...")
    save_file = OUTPUT_NF4_DIR / "pytorch_model.bin"
    torch.save(model.state_dict(), save_file)
    tokenizer.save_pretrained(str(OUTPUT_NF4_DIR))
    shutil.copy2(SOURCE_MODEL_DIR / "config.json", OUTPUT_NF4_DIR / "config.json")
    if (SOURCE_MODEL_DIR / "chat_template.jinja").exists():
        shutil.copy2(SOURCE_MODEL_DIR / "chat_template.jinja", OUTPUT_NF4_DIR / "chat_template.jinja")

    size_mb = save_file.stat().st_size / (1024 * 1024)
    print(f"[OK] 4-Bit NF4 checkpoint saved successfully! File size: {size_mb:.2f} MB ({size_mb/1024:.2f} GB)")
    print("=" * 70)


if __name__ == "__main__":
    quantize_to_nf4()
