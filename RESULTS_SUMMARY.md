# SymboLM Empirical Audit: Comprehensive Results Summary (Task 7)

> [!WARNING]
> **HISTORICAL AUDIT SNAPSHOT — SUPERSEDED (2026-10-22).** This document was written while the old Kaggle run was still active. That run stopped at **2,119/4,800 cells**; the 37.0% / 39.7% headline is retracted. The old scoring path is not the v3 strict scorer. Current status: [`docs/EVIDENCE_STATUS.md`](docs/EVIDENCE_STATUS.md). New protocol: [`PREREG_VALUATION_V2.md`](PREREG_VALUATION_V2.md). The Hugging Face model repositories are now private.

> [!IMPORTANT]
> **Framing correction (2026-10-23).** Read this document through the token-efficiency lens:
> the dependent variable is generated tokens and joules per **correct** answer, while strict
> accuracy is a non-inferiority constraint. The SmolLM2 pilot is a Tier B measurement, not
> the v3 result. The six-arm run stopped because the free Kaggle quota/infrastructure limit
> ended the session; it is **incomplete**, not a negative or confirmatory result. “Biomimetic”
> and “System 1/2” language in the historical sections is a design analogy, not a claim of
> biological fidelity. Canonical framing:
> [`docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`](docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md).

**Principal Systems Architect & Research Director:** **Sanish Gyawali** (Rupandehi, Nepal)  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**Adversarial Peer Reviewer:** Claude (Haiku 5.5, Anthropic)  
**Repository Branch:** `audit/empirical-hardening`  
**Date:** October 17, 2026  
**Status:** HISTORICAL — the old run stopped at 2,119/4,800 cells; results below are audit material, not current claims.  

---

## 1. Executive Summary & Audit Background

Following an adversarial scientific peer review by Claude (Haiku 5.5, Anthropic), SymboLM underwent a comprehensive, zero-compromise empirical audit. The objective was to replace every early simulation, constant baseline assumption, and lenient scoring fallback with strict, reproducible measurements on real hardware under free compute quotas (Kaggle dual Tesla T4 GPUs and host CPU).

This document synthesizes the empirical findings across all audit tasks (Tasks 0 through 7) and provides the definitive scientific record for research grant evaluations (Emergent Ventures / Dr. Tyler Cowen, 1517 Fund, Nat Friedman & Daniel Gross).

---

## 2. Master Audit Findings (Tasks 0, 1, 3, 4)

### A. Task 0: Complete Claims & Number Classification ([`AUDIT.md`](AUDIT.md))
- Every numerical claim in early documentation was forensically cataloged into:
  - `MEASURED`: Edge GGUF speeds (34.3 tok/s), memory sizes (999 MB Q4 GGUF, 3.55 GB BF16), unit tests (26/26 passing), atomic bypass latency (3 tokens).
  - `SIMULATED`: Early 5-arm ablation (`eval/ablation_5_arm_suite.py` which utilized synthetic random distributions). Replaced by `eval/ablation_real.py`.
  - `CONSTANT`: Early baseline comparison tokens (e.g., fixed token lengths in benchmark scripts).
  - `UNVERIFIED`: The preliminary 50.5% accuracy claim on the 1,000-question multi-year benchmark.

### B. Task 1: Honest Rescoring & Forensic Breakdown ([`eval/RESCORING_AUDIT_REPORT.md`](eval/RESCORING_AUDIT_REPORT.md))
- **Finding:** Rescoring `eval/eval_v2_1000_results.json` under strict criteria (zero regex tail guessing, unclosed truncation = FAIL) shifted raw Pass@1 accuracy from **50.5% down to 3.2% (32 / 1,000)**.
- **Root Cause Forensic:**
  1. **The 512-Token Guillotine:** 917 out of 1,000 outputs (91.7%) reached the 512-token limit and were truncated mid-sentence (e.g. `</thi`).
  2. **Degenerate Token Loop:** Without a repetition penalty, the model entered a repetitive cycle (`"bands: <val> bands: <val>"`, occurring in 823 outputs).
  3. **Scorer Fallback:** The legacy scorer's Step 5 extracted trailing integers from these looped traces, awarding false-positive passes.
  4. **The Factual Reflex Sub-System:** The 32 problems that passed honest scoring were pure `<reg:bypass>` atomic scientific facts, achieving **80.0% Pass@1 accuracy in exactly 3 tokens**, proving that the System 1 reflex routing mechanism functions with high fidelity.
- **Architectural Solution:**
  - Eradicated the 512-token cap across the entire codebase (unbounded 4,096 headroom with `AnswerStoppingCriteria`).
  - Added `repetition_penalty = 1.1`.
  - Eliminated regex tail-guessing fallbacks permanently.

### C. Task 3: Verifier Falsification Test ([`eval/VERIFIER_FALSIFICATION_REPORT.md`](eval/VERIFIER_FALSIFICATION_REPORT.md))
- **Finding:** Corrupted arithmetic operations injected into 100 step-by-step mathematical derivations yielded an empirical rejection rate of:
  $$P(✗) = 0.0\% \quad (0 / 100)$$
- **Scientific Implication:** Static autoregressive generation in small parameter models ($\le 1.7\text{B}$) is unable to detect its own arithmetic errors post-hoc without external feedback. The model emits `verify(...) ✓` tautologically.
- **Architectural Resolution:** Validates Sanish Gyawali's directive to decouple arithmetic verification from autoregressive generation into the **Sandboxed SymPy Tiny Terminal** (`symbolm/tiny_terminal.py`), executing verified deterministic checks in $<5$ ms.

### D. Task 4: Benchmark Contamination Audit ([`eval/CONTAMINATION_AUDIT_REPORT.md`](eval/CONTAMINATION_AUDIT_REPORT.md))
- **Audit Target:** Overlap between `data/concept_math_contrastive_v2_1.jsonl` (training curriculum, 5,000 items) and `data/benchmark_1000_multi_year.jsonl` (evaluation benchmark, 1,000 items).
- **Exact Verbatim Prompt Matches:** **Exactly 47 items (4.7%)**, confirming Claude Haiku 5.5's finding #7.
- **8-Gram Partial Overlap:** 49 items (4.9%).
- **Strictly Clean Out-of-Distribution Problems:** **904 items (90.4%)**.
- **Action Taken:** Extracted a pristine, stratified, 100% clean evaluation set of 200 problems: [`data/held_out_eval_200_clean.jsonl`](data/held_out_eval_200_clean.jsonl) (SHA-256 at the time, CRLF checkout: `8369BD8CAF919F672D8EC87FFF2295FF6380AE9E451F668F9EEEE908F83CF07B`; current LF bytes: `575B247066B65E5B8514990280C534BAA9448014E100ADCFE67D385E43B9AA8E`).

---

## 3. Real 4-Arm Ablation Suite (Task 2 Receipts)

- **Harness:** [`eval/ablation_real.py`](eval/ablation_real.py)
- **Model:** `HuggingFaceTB/SmolLM2-1.7B-Instruct` + `gyawalisanish0/SymboLM-2.1-SmolLM2-1.7B` fused weights.
- **Dataset:** 200 clean held-out problems (`data/held_out_eval_200_clean.jsonl`).
- **Receipts:** [`eval/ablation_real_results.json`](eval/ablation_real_results.json) & [`eval/ABLATION_REAL_REPORT.md`](eval/ABLATION_REAL_REPORT.md).
- **Ablation Scorecard (Pass@1, Honest Scorer):**

| Ablation Arm | Accuracy (Pass@1) | Wilson 95% CI | Mean Tokens | Compression vs CoT | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Arm A: Standard CoT** | **7.00%** | [4.2%, 11.4%] | 167.6 | Baseline (1.0×) | `MEASURED` |
| **Arm B: Answer Only** | **15.00%** | [10.7%, 20.6%] | 8.1 | 20.67× (-95.2%) | `MEASURED` |
| **Arm C: Full Registers** | **10.00%** | [6.6%, 14.9%] | 125.0 | 1.34× (-25.4%) | `MEASURED` |
| **Arm D: Shuffled Plan (Causal Control)** | **9.00%** | [5.8%, 13.8%] | 136.4 | 1.23× (-18.6%) | `MEASURED` |

- **Causal Register Impact (Lanham Intervention):** Arm C (10.00%) vs. Arm D (9.00%) yields $\Delta_{\text{causal}} = +1.00\%$.

---

## 4. Hardware Valuation Protocol & Real Silicon Energy (Task 2/5 Receipts)

- **Harness:** [`eval/power_profiler.py`](eval/power_profiler.py) on Kaggle Dual NVIDIA Tesla T4 GPUs (100ms NVML power polling).
- **Dataset:** 200 clean held-out problems (`data/held_out_eval_200_clean.jsonl`).
- **Baseline Idle Power:** Measured at **32.60 Watts**.
- **Receipts:** [`results/valuation/valuation_results.json`](results/valuation/valuation_results.json) & [`VALUATION.md`](VALUATION.md).
- **Physical Silicon Energy Scorecard:**

| Valuation Arm | Accuracy | Dynamic J / Correct | Latency / Correct | Mean Tokens | Energy Type | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A: Ordinary CoT** | **7.00%** | **2052.41 J** | 62.172 s | 167.6 | `MEASURED (NVML)` | `VERIFIED` |
| **Arm B: Answer Only** | **15.00%** | **25.84 J** | 1.461 s | 8.1 | `MEASURED (NVML)` | `VERIFIED` |
| **Arm C: Full Registers** | **9.50%** | **1265.75 J** | 38.528 s | 141.3 | `MEASURED (NVML)` | `VERIFIED` |
| **Arm D: Router Only (Control)** | **10.50%** | **1301.41 J** | 39.412 s | 159.8 | `MEASURED (NVML)` | `VERIFIED` |
| **Arm E: CoT + Python Sandbox** | **23.00%** | **444.30 J** | 13.571 s | 120.2 | `MEASURED (NVML)` | `VERIFIED` |

- **Physical Silicon Efficiency Gains (pilot only):**
  - Arm C consumes **38.3% less dynamic energy per correct solve** than Ordinary CoT ($1265.75\text{ J}$ vs. $2052.41\text{ J}$).
  - Arm E (Tiny Terminal emulation) delivers **78.4% lower energy per correct solve** ($444.30\text{ J}$ vs. $2052.41\text{ J}$) with a **3.3× accuracy boost** (23.0% vs. 7.0%).
  - These are **Tier B pilot readings**, not v3 results. They do not establish format value, because C is not better than the router-only D.
- **Pre-Registered Hypothesis Test (Arm C vs. Arm D on SmolLM2):**
  - $\Delta J = +24.26\text{ J}$, Paired Bootstrap 95% CI: $[-844.57, +861.30]$.
  - As Claude Haiku 5.5 noted, SmolLM2 is a small non-reasoning base model, so router-dominated savings are expected on this transfer check.
  - The correct conclusion is **no format value established**. The old “Format Value Confirmed” phrase is retracted.

### 4.1 Token-efficiency reading (derived, Tier B)

Using the same fixed 200-problem set, `tokens per correct = mean tokens ÷ strict accuracy`:

| Arm | Strict accuracy | Mean tokens | Implied tokens/correct | J/correct |
|---|---:|---:|---:|---:|
| A: ordinary CoT | 7.0% | 167.6 | ~2,394 | 2,052.41 |
| B: answer-only | 15.0% | 8.1 | ~54 | 25.84 |
| C: full registers | 9.5% | 141.3 | ~1,487 | 1,265.75 |
| D: router-only | 10.5% | 159.8 | ~1,522 | 1,301.41 |
| E: CoT + Python sandbox | 23.0% | 120.2 | ~523 | 444.30 |

All token values are **total generation**, not a separate thinking-token count. B is the cheap
floor; E is the strongest engineering signal; C is not better than D. The old flagship’s
apparent token win is not comparable because its Arm C traces lacked `<ans>` and its scoring
path was different.

---

## 5. Unfinished Flagship Valuation: DeepSeek-R1-Distill-Qwen-1.5B Multi-Budget Grid

- **Kernel:** [`sanishgyawali/symbolm-deepseek-1-5b-valuation-grid`](https://www.kaggle.com/code/sanishgyawali/symbolm-deepseek-1-5b-valuation-grid)
- **Base Architecture:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (Flagship reasoning model).
- **Adapter:** `gyawalisanish0/SymboLM-2.1-DeepSeek-1.5B`.
- **Required Budget Grid:** $\{256, 512, 1024, 2048\}$ tokens to eliminate truncation bias and map the complete Accuracy-Energy Pareto frontier.
- **Hypothesis Decision Rule:** Arm C must beat Arm D on $J/\text{correct}$ at every budget where Arm D is within 2% of peak accuracy.
- **Status:** Not completed. The run stopped at 2,119/4,800 cells; its 37.0% / 39.7% claims are retracted. A fresh v3 run is pending under `PREREG_VALUATION_V2.md`.

---

## 5. Branch Governance (updated 2026-10-22)

- **Old `main` snapshot:** Historical only. Do not treat commit `011b2c0` or its edge measurements as a current v3 result. The old audit notice is preserved in the archive.
- **Old audit branch (`audit/empirical-hardening`):** Superseded by `research/rescue-v3`. Its partial six-arm receipts and lenient scores are audit material, not current claims.
- **Current rescue branch (`research/rescue-v3`):** Contains the canonical SFT split, v2 benchmark, strict scorer, repaired evaluator, and the v3 preregistration. It must not be merged to `main` until the SFT generation check and the full strict evaluation are complete and reviewed by Sanish Gyawali.
- **Hugging Face:** All four model repositories were made private on 2026-10-22. Public links now return 401; no repositories were deleted.
