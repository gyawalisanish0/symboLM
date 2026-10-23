# SymboLM on Lightning AI & Kaggle

> [!WARNING]
> **HISTORICAL INFRASTRUCTURE GUIDE (2026-10-23).** This page describes the old SFT/GRPO
> pipeline, not the v3 rescue. The old adapters are not the v3 artifact, and the Hugging Face
> repositories are private. The current v3 SFT path is
> [`../kaggle/sft_v3/`](../kaggle/sft_v3/), currently blocked on the r4 CRLF/LF provenance fix.
> Do not use the old checkpoint paths below as a current result.

This directory contains automation scripts to run and resume SymboLM SFT and GRPO training across **Lightning AI** and **Kaggle**.

---

## ⚡ 1. Lightning AI Setup

### A. Local Setup (Already completed on your PC)
- Installed: `lightning` CLI (`C:\Users\user\.local\bin\lightning.exe`) via `uv`.
- To log in:
  ```powershell
  lightning login
  ```

### B. Launching a Lightning Studio
1. In the [Lightning AI Console](https://lightning.ai) or via CLI:
   ```powershell
   lightning studio create --name symbolm
   lightning studio switch --machine-type T4  # or L4 for faster Ada Lovelace GPU
   ```
2. Copy your checkpoint from local to the studio:
   ```powershell
   lightning studio cp ./checkpoints/sft_adapter/checkpoint-200 symbolm:~/symboLM/checkpoints/sft_adapter/checkpoint-200
   ```
3. Inside the Studio terminal, run:
   ```bash
   bash lightning/setup_studio.sh
   ```

---

## 📊 2. Kaggle Setup

### A. Local Setup (Already completed on your PC)
- Installed: `kaggle` CLI (`C:\Users\user\.local\bin\kaggle.exe`) via `uv`.
- API Key:
  1. Go to [kaggle.com/settings](https://www.kaggle.com/settings).
  2. Under the **API** section, click **Create New Token**.
  3. Move the downloaded `kaggle.json` into `C:\Users\user\.kaggle\kaggle.json`.

### B. Running on Kaggle
1. Open [kaggle.com](https://www.kaggle.com) $\rightarrow$ **New Notebook**.
2. Click **File** $\rightarrow$ **Import Notebook** and select `kaggle/symboLM_sft_kaggle.ipynb`.
3. Set Accelerator to **GPU T4 x2** or **GPU P100**.
4. Enable **Internet Access** in notebook settings.
5. Click **Run All**!
