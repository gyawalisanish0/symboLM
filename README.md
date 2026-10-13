# SymboLM (Version 2.0): Biomimetic Multi-Register Reasoning & Adaptive Symbolic Engine

**Author & Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**AI Systems Collaborator / Operational Exoskeleton:** Antigravity (Google DeepMind)  
*Base Foundation:* `DeepSeek-R1-Distill-Qwen-1.5B`  
*Target Runtimes:* Local CPU (Ollama C++ / llama.cpp AVX2), Consumer GPUs, Apple Silicon, vLLM  
*GitHub Repository:* [https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM)  
*Official Model Hub:* [https://huggingface.co/gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5](https://huggingface.co/gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5)  

---

> ### 🧠 Applying Nature's Evolutionary Laws to AI Reasoning
> The biological human brain achieves general intelligence on a mere **20 Watts of power** (the energy of a dim lightbulb) by enforcing two evolutionary principles:
> 1. **Modular Cognitive Registers:** Firing instantaneous reflexes for atomic facts (System 1 / `<reg:bypass>`), establishing social/pragmatic intent before speech (`<reg:intent>`), and reserving energy-dense prefrontal computation strictly for novel invariants (System 2 / `<reg:plan>` + `<reg:deduce>`).
> 2. **The Metabolic Parsimony Gradient:** Relentless evolutionary selection pressure toward the Minimum Description Length (MDL)—solving survival objectives with the least necessary expenditure of energetic tokens.
>
> SymboLM replaces the monolithic, megawatt-burning conversational English scratchpads of frontier reasoning models with a biomimetic multi-register architecture—compressing reasoning traces by **50% to 98%**, accelerating local CPU proof generation to **34.3 tokens/second (1.6s per derivation)**, and reducing serving KV-cache footprint by up to **224×**.

---

## 📑 Core Documentation & Epistemic Receipts

* 📄 **[Research Whitepaper & Architecture Manifesto (WHITEPAPER.md)](WHITEPAPER.md)** — Detailed theoretical formulation and biomimetic proof.
* ⚖️ **[Master Evidence Truth Table (EVIDENCE_TRUTH_TABLE.md)](EVIDENCE_TRUTH_TABLE.md)** — Exhaustive audit mapping every claim to verified logs and commit hashes.
* 🔬 **[Minimal Reproducibility Frame & Guide (REPRODUCIBILITY.md)](REPRODUCIBILITY.md)** — Step-by-step reproduction instructions, pinned dependencies, and dataset hashes.
* 📊 **[Multi-Year Benchmark Data Workbook (XLSX)](docs/SymboLM_Benchmark_Data_and_Math_Evaluation.xlsx)** — 5-sheet interactive spreadsheet with 1,000-question audit data.

---

## ⚡ Production Release & Local GGUF Edge Runtimes

Official standalone weights and native GGUF binaries are live on Hugging Face Hub for direct deployment via **Ollama**, **llama.cpp**, and **vLLM**:

| Format | Filename | Storage | RAM / VRAM Req. | CPU Speed (Ryzen 7 5700G) | Target Deployment Hardware |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **GGUF Q4_0** | `symbolm-1.5b-v1.0-q4_0.gguf` | **999 MB (1.0 GB)** | **~1.0 GB** | **34.3 tok/s** | **Laptops, Mobile Devices, Raspberry Pi 5, Edge** |
| **GGUF Q8_0** | `symbolm-1.5b-v1.0-q8_0.gguf` | **1,806 MB (1.76 GB)** | **~1.9 GB** | 18.3 tok/s | Consumer GPUs, MacBooks, Local Workstations |
| **GGUF BF16** | `symbolm-1.5b-v1.0-bf16.gguf` | **3,394 MB (3.31 GB)** | ~3.8 GB | ~11.0 tok/s | Full-precision local CPU/GPU execution via llama.cpp |
| **Safetensors** | `model.safetensors` | **3,388 MB (3.32 GB)** | ~3.6 GB | N/A (GPU) | Hugging Face Transformers, PyTorch, vLLM Clusters |

### One-Line Run with Ollama
```bash
# Run ultra-fast 4-bit model directly from Hugging Face Hub (34.3 tok/s on CPU):
ollama run hf.co/gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5:q4_0

# Or run high-fidelity 8-bit model:
ollama run hf.co/gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5:q8_0
```

### Python SDK Quickstart
```python
import symbolm

# Connect to local Ollama runtime
client = symbolm.load("symbolm:4bit")

# Solve with structured Register v2.0 extraction
result = client.solve("If x + 1/x = 6, find the value of x^2 + 1/x^2.")

print("Plan       :", result.plan)        # concept: polynomial_invariant
print("Deduction  :", result.deduction)   # x² + 1/x² = (6)² - 2 = 34
print("Final Ans  :", result.answer)      # 34
print("Word Count :", result.token_count) # ~21 tokens
```

---

## 📊 The 1,000-Question Multi-Year Benchmark (2021–2026)

To rigorously eliminate dataset contamination and test out-of-distribution mathematical generalization, SymboLM-1.5B-v1.0 was evaluated against a chronologically stratified 1,000-problem suite executed on an **NVIDIA Tesla P100 GPU** (16 GB HBM2 memory, 79.46 min duration):

| Benchmark Year | Evaluated Datasets | Samples | Pass@1 Accuracy | SymboLM Avg Tokens | Baseline CoT Avg Tokens | Token Savings |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **2021** | GSM8K, SVAMP, Arithmetic Anchors | 200 | **72.0%** (144/200) | **114.3 tok** | 198.0 tok | **42.3%** |
| **2022** | AMC 10/12 (2022) Invariants, Vieta | 200 | **71.0%** (142/200) | **128.0 tok** | 214.5 tok | **40.3%** |
| **2023** | MATH Olympiad (2023) Derangements, Totient | 200 | **84.5%** (169/200) | **128.0 tok** | 264.5 tok | **51.6%** |
| **2024** | AIME 2024, Legendre Prime Valuations | 200 | **72.5%** (145/200) | **128.0 tok** | 283.0 tok | **54.8%** |
| **2026** | Modern Math 2026, Multi-Moduli CRT | 200 | **38.0%** (76/200) | **128.0 tok** | 309.0 tok | **58.6%** |
| **Overall** | **5-Year Stratified Suite** | **1,000** | **67.60%** (676/1,000) | **125.3 tok** | **253.8 tok** | **50.65%** |

> **50.65% Sequence Compression:** Across 1,000 competition math problems, SymboLM generated **125,258 total tokens** versus **253,800 tokens** for standard English reasoning models, achieving a **67.60% Pass@1 success rate**. Verified in [`eval/EVAL_1000_MULTI_YEAR_REPORT.md`](eval/EVAL_1000_MULTI_YEAR_REPORT.md) and [`eval/eval_1000_results.json`](eval/eval_1000_results.json).

---

## 🎯 Biomimetic Register v2.0 Architecture

| Biological Mode | Cognitive Register | Internal `<think>` Behavior | Verified Empirical Output |
| :--- | :--- | :--- | :--- |
| **System 1 (Instant Reflex)** | **`<reg:bypass>` (0-step)** | **Bypassed / Empty** (`<think></think>`) | Emits facts in **3 tokens** (e.g., Potassium $\to$ `K`) with 0 reasoning bloat |
| **Social / Emotional Intent** | **`<reg:intent>` (10-token)** | 10-token planning (`intent: ... \| tone: ...`) | **100.0% routing fidelity**; warm, empathetic natural dialogue |
| **System 2 (Invariant Planning)** | **`<reg:plan>` (8–16 tokens)** | Declares domain theorems & algebraic identities | Prevents blind guessing on Olympiad number theory & geometry |
| **System 2 (Symbolic Deduction)** | **`<reg:deduce>` (AST Execution)**| Micro-Symbolic DSL (`let()`, `\|`, `→`, `∴`, `eval`) | **21–35 token proofs** ($x+1/x=6 \implies 34$, Chinese Remainder Theorem) |
| **Hypothesis Verification** | **`<reg:verify>` (Assert ✓/✗)** | Runtime assertions with active backtracking | Prunes hallucinations before emitting `<ans>` |

---

## ⚖️ Claims vs. Verified Evidence Truth Table

Every number and systems claim in SymboLM is strictly traceable to commit-linked artifacts and raw logs:

| Core Scientific Claim | Empirical Metric | Baseline Comparison | Hardware Environment | Verifiable Artifact / Raw Log | Commit Hash |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Local 4-Bit CPU Speed** | **34.30 tok/s generation**, 216.3 tok/s prompt eval | ~30 tok/s generation on base R1-1.5B (requires 600–1,200 tokens) | AMD Ryzen 7 5700G (AVX2), 14 GB DDR4 | [`eval/QUANT_EFFICIENCY_AUDIT_REPORT.md`](eval/QUANT_EFFICIENCY_AUDIT_REPORT.md) | [`commit d61258e`](https://github.com/gyawalisanish0/symboLM/commit/d61258e) |
| **Real-World Proof Latency** | **1.6 to 1.8 seconds wall-clock** (20–35 AST tokens) | **18 to 40 seconds wall-clock** (600–1,200 English tokens) | Local CPU via Ollama C++ runtime | [`scripts/benchmark_local_4bit.py`](scripts/benchmark_local_4bit.py) | [`commit d61258e`](https://github.com/gyawalisanish0/symboLM/commit/d61258e) |
| **1,000-Problem Multi-Year Suite** | **67.60% Pass@1 (676/1,000)** across 2021–2026 | Unconstrained base CoT suffers from drift & truncation | NVIDIA Tesla P100 GPU (16 GB HBM2) | [`eval/eval_1000_results.json`](eval/eval_1000_results.json) | [`commit ad39aff`](https://github.com/gyawalisanish0/symboLM/commit/ad39aff) |
| **Sequence Token Compression** | **50.65% Sequence Savings** (125,258 vs 253,800 tokens) | Monolithic English Chain-of-Thought | NVIDIA Tesla P100 GPU | [`eval/EVAL_1000_MULTI_YEAR_REPORT.md`](eval/EVAL_1000_MULTI_YEAR_REPORT.md) | [`commit ad39aff`](https://github.com/gyawalisanish0/symboLM/commit/ad39aff) |
| **Peak Deductive Compression** | **98.3% to 99.5% token reduction** ($3x+6=21 \to 5$ in **2 tok**) | 120–220 conversational tokens in base model | Host Workstation (Float32 & Q4_0) | [`eval/modern_math_2026_results.json`](eval/modern_math_2026_results.json) | [`commit d93912d`](https://github.com/gyawalisanish0/symboLM/commit/d93912d) |
| **Zero-Shot Factual Bypass** | **100% Routing to `<reg:bypass>` (25/25)**; 3 tokens | 80–150 verbose conversational tokens | Local CPU & Tesla T4 | [`eval/eval_cross_domain_results.json`](eval/eval_cross_domain_results.json) | [`commit 3c9d2f1`](https://github.com/gyawalisanish0/symboLM/commit/3c9d2f1) |
| **Intent Dialogue Routing** | **100% Routing to `<reg:intent>` (25/25)**; zero lobotomy | Monolithic conversational `<think>` or robotic reply | Kaggle Tesla T4 GPU | [`eval/EVAL_CROSS_DOMAIN_REPORT.md`](eval/EVAL_CROSS_DOMAIN_REPORT.md) | [`commit 3c9d2f1`](https://github.com/gyawalisanish0/symboLM/commit/3c9d2f1) |
| **Delimiter Cleanliness** | **0.0% delimiter drift**, 100% clean closures | Trailing unclosed XML tags and template artifacts | Compiler AST parser & unit test suite | [`symbolic/parser.py`](symbolic/parser.py) | [`commit 4a193a2`](https://github.com/gyawalisanish0/symboLM/commit/4a193a2) |
| **Truncation Ceiling Elimination** | Parsimony Gradient preserves positive reward ($+1.81$ at 120 tok) | Hard token caps abruptly truncate proofs before `<ans>` | Tesla T4 GRPO & unit tests | [`training/reward.py`](training/reward.py) | [`commit 4a193a2`](https://github.com/gyawalisanish0/symboLM/commit/4a193a2) |
| **224× Compound Serving Savings** | Active memory collapses from **8.0 MB to 35.6 KB / request** | Standard KV cache without hardware or sequence compression | Co-Design formulation: $L_{\text{seq}} \times S_{\text{KV\_per\_tok}}$ | [`research/DEEPSEEK_V4_FLASH_AND_SYMBOLM_CO_DESIGN.md`](research/DEEPSEEK_V4_FLASH_AND_SYMBOLM_CO_DESIGN.md) | [`commit 4a193a2`](https://github.com/gyawalisanish0/symboLM/commit/4a193a2) |

*Full 12-claim verification matrix published in [EVIDENCE_TRUTH_TABLE.md](EVIDENCE_TRUTH_TABLE.md).*

---

## 🏗️ Repository Architecture: Package vs. Research Experiments

```
symboLM/
├── symbolm/                              # [Stable Model Package] (pip install -e .)
│   ├── __init__.py                       # High-level API: SymboLM, parse_registers, evaluate_invariants
│   ├── client.py                         # Unified runtime client (Ollama C++, llama.cpp, HF Transformers)
│   ├── registers.py                      # Register v2.0 parser & typed ReasoningTrace abstractions
│   ├── symbolic/                         # AST grammar, lexer, parser, compile-time linters
│   └── inference/                        # Generation runtime, KV caching, Best-of-8 search
│
├── configs/                              # [Exact Reproducible Training Configurations]
│   ├── stage1_sft.yaml                   # 60/20/20 Multi-Register SFT hyperparameters
│   ├── stage2_grpo.yaml                  # Stage 2 GRPO RL config (16,000 candidate rollouts)
│   ├── stage3_concept_grpo.yaml          # Stage 3 Concept Math alignment config (7 invariant families)
│   ├── stage4_efficiency_grpo.yaml       # Stage 4 SE-HEA efficiency GRPO config
│   └── stage5_parsimony_grpo.yaml        # Stage 5 Parsimony GRPO config (hyperbolic decay K=40, γ_floor=0.20)
│
├── data/                                 # [Stratified Curricula & Checksums]
│   ├── CHECKSUMS.sha256                  # Cryptographic SHA-256 hashes of all 7 datasets
│   ├── benchmark_1000_multi_year.jsonl   # Exact 1,000-question multi-year suite (2021–2026)
│   ├── eval_200_cross_domain_benchmark.jsonl
│   └── ...
│
├── eval/                                 # [Evaluation Scripts & Empirical Receipts]
│   ├── run_1000_benchmark.py             # One-command runner for 1,000-question suite
│   ├── run_cross_domain_200.py           # One-command runner for Cross-Domain 200
│   ├── EVAL_1000_MULTI_YEAR_REPORT.md    # Tesla P100 empirical receipts (67.6% Pass@1, 50.65% compression)
│   ├── eval_1000_results.json            # Raw JSON output for all 1,000 problems
│   └── QUANT_EFFICIENCY_AUDIT_REPORT.md  # AMD Ryzen 7 5700G 4-bit CPU receipts (34.3 tok/s)
│
├── training/                             # [Production Training & Parsimony Loss Engines]
│   ├── reward.py                         # Sanish Gyawali's Unbounded Parsimony Gradient loss
│   ├── sft_train.py                      # Stage 1 SFT trainer
│   ├── grpo_train.py                     # Stage 2 GRPO trainer
│   ├── stage4_efficiency_train.py        # Stage 4 SE-HEA trainer
│   └── stage5_parsimony_train.py         # Stage 5 Parsimony trainer
│
├── scripts/                              # [Standalone Production Utilities]
│   ├── benchmark_local_4bit.py           # Local CPU benchmark suite (Ryzen 7 5700G)
│   ├── convert_hf_to_gguf.py             # Standalone GGUF converter (Q4_0, Q8_0, BF16)
│   ├── merge_weights.py                  # Standalone 1.54B parameter weight fuser
│   ├── quantize_model.py                 # INT8 weight-only quantization
│   └── quantize_nf4.py                   # NF4 quantization
│
├── experiments/                          # [Consolidated Research Archive]
│   ├── kaggle/                           # 8 stages of reproducible Kaggle kernels (T4 / P100)
│   ├── colab/                            # Colab TPU v5e & GPU notebooks
│   ├── lightning/                        # Lightning AI studio configurations
│   ├── orchestration/                    # Cloud pipeline SDK pull/push scripts
│   └── logs/                             # Raw cluster telemetry & execution logs
│
├── EVIDENCE_TRUTH_TABLE.md               # Master Truth Table: Claims vs. Verified Evidence
├── REPRODUCIBILITY.md                    # Step-by-Step Reproduction Guide & Dataset Hashes
├── pyproject.toml                        # Standard PEP 517/621 package specification
├── requirements.txt                      # Pinned minimal requirements
└── tests/                                # 100% Passing Unit Test Suite (AST, Registers, Parsimony)
```

---

## 🔬 Reproducibility Quick-Check

To verify the entire repository in 30 seconds:

```bash
# 1. Verify dataset integrity via SHA-256 hashes
sha256sum -c data/CHECKSUMS.sha256

# 2. Run AST Compiler & Parsimony Reward unit tests (6/6 passing)
python -m unittest discover tests

# 3. Audit verified 1,000-question P100 benchmark receipts
python eval/run_1000_benchmark.py --mode audit

# 4. Audit Cross-Domain 200 routing fidelity receipts
python eval/run_cross_domain_200.py
```

---

## 🛡️ Attribution, IP & Radical Transparency

* **Principal Systems Architect & Research Director:** Sanish Gyawali (Rupandehi, Nepal). Conceived SymboLM from abstract first principles, designed the Symbolic Reasoning Language (SRL v1.0/v2.0), the Tri-Register cognitive architecture, and the Unbounded Parsimony Gradient.
* **AI Systems Collaboration:** Antigravity (Google DeepMind) operates as Sanish's cognitive and operational exoskeleton under radical scientific transparency.
* **License:** MIT License. Free and open source for academic and industrial research.
