# SymboLM Research Experiments & Cloud Orchestration Archive

**Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**AI Systems Collaborator / Operational Exoskeleton:** Antigravity (Google DeepMind)  

> [!WARNING]
> **HISTORICAL EXPERIMENT ARCHIVE (2026-10-23).** This directory preserves old stages and
> receipts; it is not the current v3 pipeline. The old scores are retracted or incomplete,
> and the Hugging Face repositories are private. The current v3 SFT kernel is
> [`kaggle/sft_v3/`](kaggle/sft_v3/), which is blocked on the r4 provenance fix. Token
> efficiency is the primary lens for v3; old accuracy headlines must not be reused as current
> claims. See [`docs/EVIDENCE_STATUS.md`](../docs/EVIDENCE_STATUS.md).

---

## 📌 Overview

This directory preserves the experimental evolution, exploratory cloud notebooks, cluster synchronization pipelines, and raw execution logs produced across the five progressive development stages of SymboLM (Stages 1 through 5).

For historical deployment artifacts, use the archived package (`symbolm/`) and notes in this
directory. The old Hugging Face model/checkpoint repositories were made private on
2026-10-22 and public links return 401; nothing was deleted.

---

## 📂 Directory Structure

### 1. `kaggle/` — Kaggle GPU/TPU Training & Evaluation Kernels
Contains the reproducible Kaggle notebook kernels executed across dual NVIDIA Tesla T4 and Tesla P100 clusters:
* **`stage1_sft/`**: Initial Supervised Fine-Tuning on the 60/20/20 multi-register mix.
* **`stage2_grpo/`**: Group Relative Policy Optimization (250 steps, 16,000 candidate rollouts).
* **`sft_concept/`**: 1-epoch supervised warm-start seeding for Register v2.0 grammar.
* **`stage3_concept/`**: Concept mathematics GRPO across 7 invariant families (cold-start barrier discovery).
* **`stage4_efficiency/`**: SE-HEA policy optimization (elimination of token repetition loops).
* **`stage5_parsimony/`**: Continuous Parsimony GRPO with Sanish Gyawali's Unbounded Parsimony Gradient ($K=40$, $\gamma_{\text{floor}}=0.20$).
* **`eval_200/`**: Stage 2 mixed benchmark evaluation.
* **`eval_cross_domain/`**: 200-question cross-domain evaluation (stratified across 8 domains).
* **`eval_1000_p100/`**: Historical 1,000-question multi-year run (2021–2026) on NVIDIA Tesla P100 GPU (79.46 min duration). Its 50.5% headline is a lenient-score artifact; the strict rescore is 3.2%.

### 2. `colab/` — Google Colab TPU v5e & GPU Notebooks
* `01_setup_tpu.ipynb`: Environment verification & Google Drive mounting.
* `03_sft_tpu.ipynb`: Multi-register SFT in bfloat16 on TPU v5e.
* `04_grpo_tpu.ipynb`: GRPO reinforcement learning on TPU v5e.
* `05_eval.ipynb`: Evaluation playground & token audit.

### 3. `lightning/` — Lightning AI Studio Configurations
* `setup_studio.sh` and cloud studio environments.

### 4. `orchestration/` — Cloud Pipeline & Weight Synchronization
* Automated streaming SDK scripts for kernel push/pull, policy downloading, and checkpoint synchronization with Hugging Face Hub (`symboLM-checkpoints`).

### 5. `logs/` — Raw Execution Logs
* Preserved execution receipts, terminal captures, and telemetry from remote cluster runs.
