**Project Name:** SymboLM (Version 1.1) — Adaptive Symbolic Reasoning & General Intelligence Engine  
**Author & Principal Architect:** Sanish Gyawali  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**GitHub Repository:** [https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM)  
**Target Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
**Current Active Compute:** Kaggle Dual NVIDIA Tesla T4 (2x 16 GB = 32 GB VRAM)  
**Active Cloud Kernel:** `sanishgyawali/symbolm-stage-2-grpo-training-dual-tesla-t4` (Version 5, Status: `KernelWorkerStatus.RUNNING`)  
**Core Objective:** Reasoning compression across multiple cognitive registers—replacing verbose English CoT with an ultra-dense symbolic DSL (`→`, `∴`, `|`, `hyp`, `verify`) for deductive tasks (60–75% token reduction), while dynamically bypassing thinking for direct factual QA, managing structured state for questionnaires, and using 10-token Intent Scratchpads for fluent, empathetic human conversation.

---

## ⚡ Stage 2 GRPO Diagnostics & Resolution History (Updated Oct 9, 2026, 09:47)

### 1. Root Cause Analysis Summary
- **Run 1 (TPU VM v3-8):** Path mismatch in `/kaggle/input` mount; container had Python 3.13 without `torch_xla`; dynamic generation loop caused XLA graph recompilations.
- **Run 2/3 (Dual Tesla T4 - v3):** Outdated `torchao 0.10.0` pre-installed on Kaggle collided with PEFT dispatcher. Fixed via `pip uninstall -y torchao`.
- **Run 4 (Dual Tesla T4 - v4):** Embedding shape mismatch: SFT adapter weights contained `[151712, 1536]` embeddings (extended symbolic vocabulary), but `grpo_train.py` initialized tokenizer from base model (`[151665, 1536]`).
- **Run 5 (Dual Tesla T4 - v5):** **RESOLVED.** `training/grpo_train.py` now loads tokenizer directly from `sft_adapter/` and sets `target_vocab_size = max(len(tokenizer), 151712)`. Cell 4 mounts both `sft_adapter` and `tokenizer_extended`.

---

## 📌 Next Steps

1. Monitor live execution of Kernel Version 5 via recurring listener hook (`task-1770`).
2. Verify policy checkpoints (`symboLM_grpo_final_policy.tar.gz`) generated in Cell 6 & 8.
3. Run local efficiency & accuracy audit (`eval/efficiency.py`).
4. Export to GGUF format for local deployment.
