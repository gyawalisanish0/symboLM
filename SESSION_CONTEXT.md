**Project Name:** SymboLM (Version 1.1) — Adaptive Symbolic Reasoning & General Intelligence Engine  
**Author & Principal Architect:** Sanish Gyawali  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**GitHub Repository:** [https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM)  
**Target Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
**Current Active Compute:** Kaggle Dual NVIDIA Tesla T4 (2x 16 GB = 32 GB VRAM) / Cloud TPU  
**Active Cloud Kernel:** `sanishgyawali/symbolm-stage-2-grpo-training-tpu-v3-8`  
**Core Objective:** Reasoning compression across multiple cognitive registers—replacing verbose English CoT with an ultra-dense symbolic DSL (`→`, `∴`, `|`, `hyp`, `verify`) for deductive tasks (60–75% token reduction), while dynamically bypassing thinking for direct factual QA, managing structured state for questionnaires, and using 10-token Intent Scratchpads for fluent, empathetic human conversation.

---

## ⚡ Stage 2 GRPO RL Milestones & Root Cause Diagnostics (Updated Oct 9, 2026, 09:25)

### 1. Root Cause Analysis of First Cloud Run
- **Observed Error in Log:** `ValueError: Can't find 'adapter_config.json' at './checkpoints/sft_adapter'` originating from PEFT loader in Cell 6.
- **Root Cause:** In Kaggle kernel source mounting, kernel sources are mounted at `/kaggle/input/notebooks/sanishgyawali/symbolm-stage-1-sft-training/` rather than `/kaggle/input/symbolm-stage-1-sft-training/`. Cell 4 checked a non-existent direct path, leaving the target `./checkpoints/sft_adapter` empty.
- **Hardware Finding:** Kaggle's default Python 3.13 image lacks pre-installed `torch_xla`, and Hugging Face `trl.GRPOTrainer` performs dynamic autoregressive generation that causes frequent graph recompilations on TPU. Switched accelerator to Dual Tesla T4 (2x 16 GB = 32 GB VRAM) for native, stable CUDA acceleration.
- **Fixes Applied & Pushed to Git (`03e780c`):**
  1. [`kaggle_grpo/symboLM_grpo_kaggle.ipynb`](file:///c:/Users/user/Dev/symboLM/kaggle_grpo/symboLM_grpo_kaggle.ipynb): Dynamic search and copy across `/kaggle/input/**/sft_adapter`.
  2. [`training/grpo_train.py`](file:///c:/Users/user/Dev/symboLM/training/grpo_train.py): Added defensive `adapter_config.json` existence guard and native unquantized `fp16`/`bf16` loading.
  3. [`kaggle_grpo/kernel-metadata.json`](file:///c:/Users/user/Dev/symboLM/kaggle_grpo/kernel-metadata.json): Configured for Dual Tesla T4 GPU.

---

## 📌 Next Steps

1. Push updated kernel to Kaggle via CLI.
2. Monitor live training steps via recurring schedule hook.
3. Download final weights (`symboLM_grpo_final_policy.tar.gz`) and verify 0% trailing syntax noise and atomic self-correction.
4. Export to GGUF (`Q5_K_M`) for fast local deployment.
