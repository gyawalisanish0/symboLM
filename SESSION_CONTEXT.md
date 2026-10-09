**Project Name:** SymboLM (Version 1.1) — Adaptive Symbolic Reasoning & General Intelligence Engine  
**Author & Principal Architect:** Sanish Gyawali  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**GitHub Repository:** [https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM)  
**Target Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
**Current Active Compute:** Kaggle Google Cloud TPU VM v3-8 (8 TPU cores, 128 GB HBM memory) in native `bfloat16` via PyTorch/XLA  
**Active Cloud Kernel:** `sanishgyawali/symbolm-stage-2-grpo-training-tpu-v3-8` (Status: `KernelWorkerStatus.RUNNING`)  
**Core Objective:** Reasoning compression across multiple cognitive registers—replacing verbose English CoT with an ultra-dense symbolic DSL (`→`, `∴`, `|`, `hyp`, `verify`) for deductive tasks (60–75% token reduction), while dynamically bypassing thinking for direct factual QA, managing structured state for questionnaires, and using 10-token Intent Scratchpads for fluent, empathetic human conversation.

---

## ⚡ Stage 2 GRPO RL Milestones (Updated Oct 9, 2026, 09:05)

### 1. Tri-Register Curriculum Dataset Compiled
- **Builder:** [`data/build_grpo_dataset.py`](file:///c:/Users/user/Dev/symboLM/data/build_grpo_dataset.py)
- **Dataset File:** `data/grpo_curriculum.jsonl` (1,200 curated prompts)
- **Distribution:**
  - **60% (720 prompts):** Symbolic Deductive Reasoning (Multi-step math, algebra, and discrete logic)
  - **20% (240 prompts):** Zero-Shot Factual Bypass (Direct QA with 0 thinking tokens)
  - **20% (240 prompts):** Intent-Scratchpad Dialogue (Interpersonal & empathetic communication)

### 2. Multi-Component Reward Verification Engine
- **Module:** [`training/reward.py`](file:///c:/Users/user/Dev/symboLM/training/reward.py)
- **Reward Functions:**
  - $R_{\text{correct}}$ (+1.0 / -0.6): Ground truth normalization match.
  - $R_{\text{syntax\_noise}}$ (+0.5 / -1.0): SRL v1.0 AST compiler verification; heavy -1.0 penalty for trailing noise tokens (`={...`, `?>"...`).
  - $R_{\text{self\_correct}}$ (+0.4): Bonus for verified hypothesis testing (`verify(...)`) or backtracking (`✗` $\rightarrow$ `✓`).
  - $R_{\text{efficiency\_bypass}}$ (+0.8 / -0.8): Token compression reward for math; +0.8 bonus for 0-token factual bypass and penalty for overthinking.

### 3. Kaggle TPU VM v3-8 Cloud Deployment Active
- **Metadata:** [`kaggle_grpo/kernel-metadata.json`](file:///c:/Users/user/Dev/symboLM/kaggle_grpo/kernel-metadata.json) (`enable_tpu: true`, `kernel_sources: ["sanishgyawali/symbolm-stage-1-sft-training"]`).
- **Notebook:** [`kaggle_grpo/symboLM_grpo_kaggle.ipynb`](file:///c:/Users/user/Dev/symboLM/kaggle_grpo/symboLM_grpo_kaggle.ipynb).
- **Live Status:** **`KernelWorkerStatus.RUNNING`** on Google Cloud TPU VM v3-8.
- **Tracking URL:** [https://www.kaggle.com/code/sanishgyawali/symbolm-stage-2-grpo-training-tpu-v3-8](https://www.kaggle.com/code/sanishgyawali/symbolm-stage-2-grpo-training-tpu-v3-8)

---

## 📌 Next Steps

1. Monitor live execution on Kaggle TPU VM v3-8.
2. Upon completion, download and verify `symboLM_grpo_final_policy.tar.gz`.
3. Run local efficiency & accuracy audit to verify:
   - 0% trailing syntax noise.
   - Successful compact self-correction (`verify` / `✗`).
   - Factual bypass and dialogue preservation.
4. Convert final policy to GGUF (`Q5_K_M`) for 40+ tok/s CPU deployment.
