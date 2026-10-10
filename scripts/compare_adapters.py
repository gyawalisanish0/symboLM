import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from inference.generate import load_symbo_model

prompts = [
    "What is the atomic number of Carbon?",
    "Solve 3*x + 6 = 21.",
    "Calculate 100 - 20 + 35.",
]

for adapter_name, adapter_path in [("Stage 2 GRPO", "checkpoints/grpo_adapter"), ("Stage 3 Concept", "checkpoints/stage3_adapter")]:
    print(f"\n==================== EVALUATING {adapter_name} ====================")
    model, tok = load_symbo_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", adapter_path)
    model.eval()
    for q in prompts:
        p = f"<｜User｜>{q}<｜Assistant｜><think>\n"
        inputs = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=48,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
            )
        raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        print(f"Q: {q}")
        print(f"A: {repr(raw)}")
