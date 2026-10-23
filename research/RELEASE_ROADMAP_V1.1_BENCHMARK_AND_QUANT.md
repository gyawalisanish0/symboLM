> [!WARNING]
> **HISTORICAL / NON-v3 RECEIPT - NOT A CURRENT RESULT (2026-10-23).** This report predates the v3 strict declared-answer scorer and the token-efficiency protocol. Its scores are retained for audit, not as current claims. The old six-arm flagship run stopped at 2,119/4,800 cells because of Kaggle quota/infrastructure limits; it is incomplete, not a confirmatory or negative result.
>
> **Current framing:** token efficiency of correct reasoning is the primary lens;
> strict accuracy is a non-inferiority gate. The biological language is a design
> analogy, not a mechanism claim. Current status: [../docs/EVIDENCE_STATUS.md](../docs/EVIDENCE_STATUS.md). Canonical framing:
> [../docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md](../docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md).

# SymboLM v1.1 Open-Source Release & Multi-Year Benchmark Roadmap
## Standalone Weight Merge, 1,000-Question Temporal Evaluation, and Multi-Bit Quantization Audit

**Principal Systems Architect & Research Director:** Sanish Gyawali  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**Location:** Rupandehi, Nepal  
**Target Release Hub:** Hugging Face Model Hub (`gyawalisanish0/symboLM-1.5B-v1.1`)  
**Base Foundation Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
**Active Champion Policy:** `checkpoints/stage5_parsimony_adapter/` (Parsimony GRPO)  

---

## 1. Executive Summary & Strategic Rationale

Following the completion of Stage 5 Parsimony GRPO (which verified 3-token atomic fact retrieval, 21-token linear invariant reduction, and 35-token Chinese Remainder Theorem derivations without truncation), the project transitions from an experimental research checkpoint to a **universally deployable open-source release**.

This roadmap establishes the four-pillar release strategy designed by Sanish Gyawali:
1. **Standalone Weight Merge:** Fusing the LoRA adapter into the base foundation weights to eliminate adapter loading friction and create a native, 1-click drop-in model for the global developer ecosystem.
2. **The 1,000-Question Multi-Year Benchmark (2021–2026):** Stratifying 200 problems across 5 consecutive years to definitively eliminate "data contamination" and "memorization" critiques by proving temporal generalization on recent 2024–2026 mathematics.
3. **Multi-Bit Quantization Hierarchy:** Producing native INT8 and INT4 (AWQ / GGUF Q4_K_M) variants to enable consumer hardware and edge execution on mobile phones, Raspberry Pis, and Ollama/llama.cpp environments with < 1.2 GB RAM footprint.
4. **Empirical Precision vs. Latency Trade-Off Audit:** Measuring throughput (tokens/sec), inference latency (seconds/problem), memory footprint, and symbolic AST syntax integrity across FP16 $\to$ INT8 $\to$ INT4.

---

## 2. The 6-Phase Execution Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│               SymboLM v1.1 RELEASE PIPELINE ARCHITECTURE               │
├────────────────────────────────────────────────────────────────────────┤
│ PHASE 1: Standalone Weight Merge                                       │
│ • Merge Stage 5 LoRA weights into DeepSeek-R1-Distill-Qwen-1.5B        │
│ • Export unified 16-bit safetensors checkpoint (checkpoints/merged/)   │
├────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: Multi-Year 1,000-Question Stratified Benchmark Suite          │
│ • Curate 200 verified questions each: 2021, 2022, 2023, 2024, 2025/26 │
│ • Blend: Elementary Math, Competition Math, Physics, CS, and Logic     │
├────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: 1,000-Batch FP16 Baseline & Accuracy Benchmark                │
│ • Run full 1,000 evaluation run on Kaggle GPU / Local                  │
│ • Measure Pass@1, Token Compression %, and Register Routing            │
├────────────────────────────────────────────────────────────────────────┤
│ PHASE 4: INT8 & INT4 Quantization (AWQ / BitsAndBytes / GGUF)          │
│ • Export 8-bit & 4-bit quantized versions                              │
│ • Generate GGUF files for Ollama / llama.cpp users                     │
├────────────────────────────────────────────────────────────────────────┤
│ PHASE 5: Latency, Throughput & Edge Audit                              │
│ • Benchmark tokens/sec and RAM footprint across FP16 vs INT8 vs INT4   │
├────────────────────────────────────────────────────────────────────────┤
│ PHASE 6: Hugging Face Model Card & Public Draft Release                │
│ • Draft Model Card (gyawalisanish0/symboLM-1.5B-v1.1) with receipts    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase Details & Technical Specifications

### Phase 1: Standalone Weight Merging (`scripts/merge_weights.py`)
* **Objective:** Merge the LoRA delta matrices $\Delta W = \frac{\alpha}{r} (B \cdot A)$ into the base transformer layers ($W_{\text{merged}} = W_{\text{base}} + \Delta W$).
* **Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` with embedding matrix resized to 151,712 tokens.
* **Adapter Source:** `checkpoints/stage5_parsimony_adapter/`.
* **Target Output:** `checkpoints/symbolm_v1.1_merged/` containing:
  - `model.safetensors` (~3.0 GB in FP16 / BF16).
  - `config.json` with updated vocabulary and architecture metadata.
  - `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja`.
* **Execution Environment:** Host Workstation (requires ~3.5 GB system RAM, executable in 90–120 seconds).

---

### Phase 2: Multi-Year 1,000-Question Stratified Suite (`data/build_multi_year_1000_benchmark.py`)
* **Objective:** Construct a rigorous, tamper-proof 1,000-problem evaluation suite stratified chronologically across 5 distinct years:

| Year Slice | Sample Count | Target Datasets & Problem Sources | Evaluation Objective |
| :---: | :---: | :--- | :--- |
| **2021** | 200 | GSM8K (Classic 2021), SVAMP, early MATH Level 1–2 | Historical baseline alignment |
| **2022** | 200 | AMC 10/12 (2022), GSM-Hard, MATH Level 3 | Algebraic manipulation verification |
| **2023** | 200 | MATH 2023, Olympiad Invariants, ARC-AGI Logic | Structural invariance across reasoning |
| **2024** | 200 | AIME 2024, AMC 12 (2024), Putnam 2024 | Complex multi-hop number theory |
| **2025/2026**| 200 | Modern Math 2026, Chinese Remainder Theorem, Legendre valuations | Unseen, non-contaminated frontier evaluation |
| **TOTAL** | **1,000** | **Balanced across 5 years & 7 difficulty tiers** | **Universal Temporal Generalization** |

* **Data Format:** `data/benchmark_1000_multi_year.jsonl` with fields:
  `{"id": str, "year": int, "difficulty": str, "prompt": str, "ground_truth": str, "category": str}`.

---

### Phase 3: Comprehensive Batch Evaluation
* **Objective:** Execute inference across all 1,000 questions using the merged FP16 model.
* **Metrics Recorded:**
  - **Pass@1 Accuracy (%)** stratified per year (2021, 2022, 2023, 2024, 2025/26).
  - **Average Tokens Emitted vs. Baseline English CoT.**
  - **Token Compression Ratio.**
  - **Register Routing Fidelity** (`<reg:bypass>` vs. `<reg:plan>` + `<reg:deduce>`).
  - **Trailing Syntax Noise Rate (%)**.
* **Output Artifact:** `eval/EVAL_1000_MULTI_YEAR_REPORT.md` and `eval/eval_1000_results.json`.

---

### Phase 4: Multi-Bit Quantization Hierarchy (`scripts/quantize_model.py`)
* **Objective:** Compress the merged 1.5B model for ultra-low memory footprints and edge execution:
  1. **8-Bit Quantization (INT8):**
     - Using `bitsandbytes` (LLM.int8()) and native PyTorch dynamic quantization.
     - Expected footprint: **~1.6 GB VRAM/RAM**.
  2. **4-Bit Quantization (INT4 / NF4):**
     - Using NormalFloat4 (NF4) with double quantization.
     - Expected footprint: **~0.95 GB – 1.1 GB VRAM/RAM**.
  3. **GGUF Export (llama.cpp):**
     - Quantization formats: `Q4_K_M` (balanced), `Q5_K_M` (higher accuracy), `Q8_0` (near-lossless).
     - Target destinations: Direct compatibility with **Ollama**, **LM Studio**, and **llama.cpp** on CPU and Apple Silicon.

---

### Phase 5: Latency, Throughput & Precision Audit (`eval/quant_efficiency_audit.py`)
* **Objective:** Evaluate the trade-offs between quantization and symbolic reasoning fidelity:
  - Does 4-bit quantization preserve the exact AST delimiters (`<reg:plan>`, `<reg:deduce>`, `∴`, `→`) without syntax corruption?
  - Throughput audit: Tokens generated per second across FP16 vs INT8 vs INT4.
  - Time-to-First-Token (TTFT) and total query latency.
  - Accuracy retention: $\frac{\text{Pass@1}_{\text{INT4}}}{\text{Pass@1}_{\text{FP16}}} \ge 95\%$.

---

### Phase 6: Hugging Face Model Hub Release & Documentation
* **Repository ID:** `gyawalisanish0/symboLM-1.5B-v1.1`
* **Release Artifacts:**
  - Merged 16-bit weights (`model.safetensors`).
  - INT8 and INT4 quantized variants.
  - Standalone `.gguf` binaries for Ollama integration.
  - Model Card documenting:
    * Principal Architecture & Author Attribution (**Sanish Gyawali, Rupandehi, Nepal**).
    * Biomimetic Multi-Register Philosophy (Nature's 20-Watt brain vs. datacenter token bloat).
    * 1,000-question multi-year benchmark receipts.
    * Latency & efficiency audit receipts.

---

## 4. Status Tracking

| Milestone | Deliverable | Status |
| :--- | :--- | :---: |
| **Stage 5 Parsimony GRPO** | Champion Policy Weights Verified | **COMPLETED** |
| **Phase 1** | Standalone Weight Merge (`scripts/merge_weights.py`) | **PLANNED** |
| **Phase 2** | 1,000-Question Multi-Year Benchmark Dataset | **PLANNED** |
| **Phase 3** | 1,000-Question FP16 Benchmark Execution | **PLANNED** |
| **Phase 4** | INT8 & INT4 Quantization + GGUF Export | **PLANNED** |
| **Phase 5** | Latency, Throughput & Precision Audit | **PLANNED** |
| **Phase 6** | Hugging Face Hub Model Card & Release | **PLANNED** |

---

*Authored by Sanish Gyawali from Rupandehi, Nepal. All research and systems rights reserved.*
