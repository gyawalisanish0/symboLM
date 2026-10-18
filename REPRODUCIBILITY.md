# SymboLM Minimal Reproducibility Frame & Execution Guide

**Author & Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**AI Systems Collaborator / Operational Exoskeleton:** Antigravity (Google DeepMind)  
**Repository:** [`https://github.com/gyawalisanish0/symboLM`](https://github.com/gyawalisanish0/symboLM)  

---

## 📌 Epistemic Principles

SymboLM adheres to the highest scientific standards of computational reproducibility. Any researcher, evaluator, or open-source contributor can independently reproduce every claim, benchmark metric, and training stage using the exact instructions below.

---

## 1. Environment Setup & Exact Dependencies

### Option A: Install via PyPI / Local Source Package (Recommended)
```bash
# Clone the repository
git clone https://github.com/gyawalisanish0/symboLM.git
cd symboLM

# Create clean virtual environment (Python 3.10, 3.11, or 3.12)
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install core runtime in editable mode
pip install -e .

# Or install with evaluation & training dependencies:
pip install -e ".[eval,train]"
```

### Option B: Install via Pinned Requirements
```bash
pip install -r requirements.txt
```

---

## 2. Verifying Dataset Integrity (SHA-256 Hashes)

Before running benchmarks or training, verify that your local copies of all datasets and curricula match the author's cryptographic checksums:

```bash
# On Linux / macOS:
sha256sum -c data/CHECKSUMS.sha256

# On Windows PowerShell:
Get-ChildItem -Path data -Filter "*.jsonl" | ForEach-Object { 
    $h = Get-FileHash $_.FullName -Algorithm SHA256; 
    "$($h.Hash)  $($_.Name)" 
}
```

### Author Checksums:
| Dataset File | SHA-256 Cryptographic Hash |
| :--- | :--- |
| `data/benchmark_1000_multi_year.jsonl` | `3993760fddf46e96b082f3d7e3f75e849d45d5960d5d69da4ac43f6a25761249` |
| `data/concept_math_curriculum.jsonl` | `405dbe564b85fffccd2958c755d1320af881d18ea2bfc0bac9c5d11551634979` |
| `data/eval_200_cross_domain_benchmark.jsonl` | `acb088d2a7767c53e6f681ff6a32f4c6d45cf7a463b8eb98835d8816fb18bfd4` |
| `data/eval_200_mixed_benchmark.jsonl` | `a06b30e66a058e3df4006878239e2ddc92f69ff24665699bdd2974706b4efb55` |
| `data/grpo_curriculum.jsonl` | `651d83615865b2633730e53b9fcc6221a51409d5f5c2a7c8f99f36b7ed1fef3e` |
| `data/stage3_unified_curriculum.jsonl` | `29b07365b5dd2d687c2b683a090db94c9e099179f5a37084d3f5789ac8b7e4ae` |
| `data/stage5_parsimony_curriculum.jsonl` | `8918e3d018d415c582d4f51031ff9c211bfaf3fc8135175aea5dcb809cd77305` |

---

## 3. Reproducing Local Edge Inference (34.3 tok/s on CPU)

### One-Line Execution with Ollama
```bash
# Pull and run the official 4-bit edge binary (999 MB) directly from Hugging Face:
ollama run hf.co/gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5:q4_0
```

### Running the Automated Local CPU Benchmark
To benchmark generation speed, prompt evaluation throughput, and memory footprint on your own CPU:
```bash
python scripts/benchmark_local_4bit.py
```
*Expected Output on Zen 3 AVX2 (Ryzen 7 5700G):* **~34.3 tok/s generation**, **~216.3 tok/s prompt eval**, full proof delivery in **1.6 to 1.8 seconds**.

---

## 4. Reproducing Empirical Benchmark Audits

### 1,000-Question Multi-Year Benchmark (Tesla P100 Receipts)
To verify the 67.6% accuracy and 50.65% sequence compression across 2021–2026:
```bash
# Audit verified P100 receipts:
python eval/run_1000_benchmark.py --mode audit

# Or run live sample evaluation against local Ollama endpoint:
python eval/run_1000_benchmark.py --mode live --max_samples 20
```

### Cross-Domain 200 Benchmark (Register Routing Fidelity)
To verify 100% routing fidelity to `<reg:bypass>` and `<reg:intent>` across 8 domains:
```bash
python eval/run_cross_domain_200.py
```

### Compiler AST Parser, FSM, and Tiny Terminal Test Suite
To verify Register v2.1 AST compilation, Tiny Terminal mathematical assertions, and parsimony reward behavior:
```bash
uv run --with pytest pytest tests/
```
*Expected Result:* **56 passed (100% passing)**.

---

## 5. Reproducing Training Pipelines

All 5 progressive stages of SymboLM training are governed by declarative YAML configs located in [`configs/`](file:///c:/Users/user/Dev/symboLM/configs):

| Stage | Config File | Objective & Dataset | Compute Target |
| :--- | :--- | :--- | :--- |
| **Stage 1: SFT** | `configs/stage1_sft.yaml` | 60/20/20 multi-register grounding on `data/grpo_curriculum.jsonl` | Google Colab TPU v5e / Dual Tesla T4 |
| **Stage 2: GRPO** | `configs/stage2_grpo.yaml` | R1-style candidate rollouts ($G=8$, 250 steps, 16,000 rollouts) | Dual Tesla T4 (CUDA) |
| **Stage 3: Concept GRPO** | `configs/stage3_concept_grpo.yaml` | Register v2.0 concept alignment across 7 invariant families | Dual Tesla T4 (CUDA) |
| **Stage 4: Efficiency GRPO** | `configs/stage4_efficiency_grpo.yaml` | SE-HEA anti-looping penalty (crushing repetitive n-grams) | Single Tesla T4 |
| **Stage 5: Parsimony GRPO** | `configs/stage5_parsimony_grpo.yaml` | Continuous hyperbolic parsimony gradient ($K=40$, $\gamma_{\text{floor}}=0.20$) | Single Tesla T4 |

### Executing Training from Python
```bash
# Stage 1: SFT
python training/sft_train.py

# Stage 2: GRPO
python training/grpo_train.py

# Stage 4: SE-HEA Efficiency GRPO
python training/stage4_efficiency_train.py

# Stage 5: Unbounded Parsimony GRPO
python training/stage5_parsimony_train.py
```

Historical Kaggle executed notebooks and Colab TPU v5e pipelines are fully preserved for inspection under [`experiments/kaggle/`](file:///c:/Users/user/Dev/symboLM/experiments/kaggle) and [`experiments/colab/`](file:///c:/Users/user/Dev/symboLM/experiments/colab).
