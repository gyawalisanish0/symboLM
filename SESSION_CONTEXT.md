**Project Name:** SymboLM (Version 1.1) — Adaptive Symbolic Reasoning & General Intelligence Engine  
**Author & Principal Architect:** Sanish Gyawali  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**GitHub Repository:** [https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM)  
**Target Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
**Current Active Compute:** Kaggle NVIDIA Tesla T4 (Isolated to Primary GPU `cuda:0`, 16 GB VRAM)  
**Active Cloud Kernel:** `sanishgyawali/symbolm-stage-2-grpo-training-dual-tesla-t4` (Version 7, Status: `KernelWorkerStatus.RUNNING`)  
**Core Objective:** Reasoning compression across multiple cognitive registers—replacing verbose English CoT with an ultra-dense symbolic DSL (`→`, `∴`, `|`, `hyp`, `verify`) for deductive tasks (60–75% token reduction), while dynamically bypassing thinking for direct factual QA, managing structured state for questionnaires, and using 10-token Intent Scratchpads for fluent, empathetic human conversation.

---

## ⚡ Stage 2 GRPO Diagnostics & Resolution History (Updated Oct 9, 2026, 10:07)

### 1. Root Cause Analysis Summary
- **Run 1 (TPU VM v3-8):** Path mismatch in `/kaggle/input` mount; container had Python 3.13 without `torch_xla`; dynamic generation loop caused XLA graph recompilations.
- **Run 2/3 (Dual Tesla T4 - v3):** Outdated `torchao 0.10.0` pre-installed on Kaggle collided with PEFT dispatcher. Fixed via `pip uninstall -y torchao`.
- **Run 4 (Dual Tesla T4 - v4):** Embedding shape mismatch: SFT adapter weights contained `[151712, 1536]` embeddings (extended symbolic vocabulary), but `grpo_train.py` initialized tokenizer from base model (`[151665, 1536]`). Fixed by prioritizing `sft_adapter/tokenizer.json` and setting `target_vocab_size = 151712`.
- **Run 5 (Dual Tesla T4 - v5):** Failed at `GRPOConfig(max_prompt_length=...)` because modern `trl` removed `max_prompt_length`. Fixed via dynamic signature filtering.
- **Run 6 (Dual Tesla T4 - v6):** Setup & rollouts ran! At step 1 loss calculation, `device_map="auto"` split the 1.5B model across `cuda:0` and `cuda:1`, causing `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!` when computing `per_token_loss1 = coef_1 * advantages`.
- **Run 7 (Tesla T4 cuda:0 - v7):** **RESOLVED.** Pinned model to single primary GPU (`CUDA_VISIBLE_DEVICES=0` and `device_map={"": "cuda:0"}`). The 3.0 GB model fits in 16 GB VRAM with 13 GB remaining, guaranteeing unified tensor placement. Pushed Kernel Version 7.

---

## 📌 Next Steps

1. Monitor live execution of Kernel Version 7 via recurring listener hook (`task-1770`).
2. Verify policy checkpoints (`symboLM_grpo_final_policy.tar.gz`) generated upon completion.
3. Run local efficiency & accuracy audit (`eval/efficiency.py`).
4. Export to GGUF format for local deployment.
