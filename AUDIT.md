# AUDIT.md: Full Empirical Audit of All Claims in SymboLM

**Audit Protocol:** Claude Haiku 5.5 Empirical Auditing Protocol  
**Author & Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**Branch:** `audit/empirical-hardening`  
**Date:** October 17, 2026  

---

## 1. Epistemic Classification Scheme

Every number, benchmark score, and systems multiplier cited in `README.md` and related public documents has been audited against the codebase and assigned one of four epistemic labels:

- **`MEASURED`**: Produced by an executable script from live forward passes on genuine models, with an immutable raw JSON receipt recording per-sample prompt, token count, latency, and full generation.
- **`SIMULATED`**: Generated via synthetic functions, mock probes, or pseudo-random distributions. Must NOT be presented as empirical evidence.
- **`CONSTANT`**: An assumed, fixed constant, theoretical calculation, or literature estimate rather than an observed measurement on the identical evaluation set.
- **`UNVERIFIED`**: A claim lacking an exact reproducing script, using a lenient scorer that credits truncated outputs, or conflating baseline regimes.

---

## 2. Exhaustive Audit of `README.md` Claims

| # | Claim in `README.md` | Cited Value | Epistemic Status | Producing Script / File | Raw Results Artifact | Audit Finding & Forensic Ground Truth |
| :-: | :--- | :--- | :-: | :--- | :--- | :--- |
| **1** | **Local Edge CPU Inference Speed (Q4_0)** | `34.3 tok/s` | **`MEASURED`** | `scripts/benchmark_local_4bit.py` | `eval/QUANT_EFFICIENCY_AUDIT_REPORT.md` | Verified on AMD Ryzen 7 5700G (AVX2, Zen 3) via Ollama C++ `llama.cpp` runtime. |
| **2** | **Proof Derivation Latency (CPU)** | `1.6s to 1.8s` | **`MEASURED`** | `scripts/benchmark_local_4bit.py` | `eval/QUANT_EFFICIENCY_AUDIT_REPORT.md` | Verified wall-clock execution time for 20–35 token derivations on local CPU. |
| **3** | **Overall Pass@1 Accuracy (1,000 Suite)** | `50.50% (505/1,000)` | **`UNVERIFIED`** | `eval/run_200_mixed_benchmark.py` | `eval/eval_v2_1000_results.json` | **Lenient Scorer:** Step 5 regex fallback extracted trailing numbers from truncated generations, crediting answers when the model failed to close `<ans>`. Requires `eval/rescore.py`. |
| **4** | **"Zero Premature Derivation Cuts"** | `0 premature cuts` | **`UNVERIFIED` (REFUTED)** | `eval/run_200_mixed_benchmark.py` | `eval/eval_v2_1000_results.json` | **Refuted by raw data:** **917 out of 1,000 outputs (91.7%)** reached `tokens_emitted = 512` and were truncated mid-sentence (e.g. `</thi`). Must be removed from README. |
| **5** | **"3.9× Increase in Accuracy"** | `3.9× vs baseline` | **`UNVERIFIED` (CONFLATED)** | N/A (Comparative claim) | `eval/EVAL_V2_1000_REPORT.md` | **Conflated Baseline:** Compared SymboLM v2.1 against SymboLM v2.0 (which was artificially strangled by a 192-token cap), NOT against an unconstrained standard CoT baseline. |
| **6** | **Sequence Compression vs. CoT** | `50% to 98%` | **`CONSTANT` / `UNVERIFIED`** | `data/build_multi_year_1000_benchmark.py` | `eval/eval_v2_1000_results.json` | **Assumed Baseline:** 98% is measured on elementary arithmetic ($3x+6=21$, 2 tokens). But the 50% suite-wide claim relies on `english_baseline_tokens` hardcoded as uniform constants per item rather than measured CoT rollouts. |
| **7** | **Information Efficiency Ratio (IER)** | `10.21 (17× to 25×)` | **`CONSTANT` / `UNVERIFIED`** | Derived formula | `eval/eval_v2_1000_results.json` | Baseline CoT IER was assumed from constant estimates ($0.40 - 0.60$) rather than empirical measurements on the same split. |
| **8** | **Trailing Syntax Noise** | `1.1%` | **`MEASURED`** | `eval/run_200_mixed_benchmark.py` | `eval/eval_v2_1000_results.json` | 11 / 1,000 items contained scraper or template debris. |
| **9** | **Verifier Emission of `✓` / `✗`** | `Hypothesis Verification ✓/✗` | **`UNVERIFIED` (REFUTED)** | `eval/run_200_mixed_benchmark.py` | `eval/eval_v2_1000_results.json` | **Zero emissions:** In `eval_v2_1000_results.json`, `<reg:verify>` emitted `✓` 0 times and `✗` 0 times. Instead, `"bands:"` repeated in 823 raw generations. |
| **10** | **Causal 5-Arm Register Ablation** | `-65.0% drop on shuffle` | **`SIMULATED`** | `eval/ablation_5_arm_suite.py` | `eval/ablation_5_arm_results.json` | **Simulation Only:** Script lines 36–60 used `random.randint` and hard-coded accuracy floats. Must be renamed to `eval/SIMULATED_ablation_5_arm_suite.py` and replaced with live model runs. |
| **11** | **Adaptive Allocation Curve** | `96.05% compression` | **`CONSTANT`** | `eval/plot_adaptive_allocation.py` | `eval/adaptive_allocation_curve.json` | **Hardcoded Constants:** R1 baselines were hardcoded to `580`, `720`, `940`, and `1250` tokens rather than run dynamically. |
| **12** | **Cross-Architecture (SmolLM2) Transfer** | `100.0% fidelity` | **`UNVERIFIED`** | `eval/cross_family_llama/test_smollm.py` | `eval/cross_family_llama/llama_1_7b_transfer_results.json` | **Sample Size $n=5$:** Based on only 5 hand-picked prompts, two of which truncated. Must be evaluated on full held-out split. |
| **13** | **Compound KV Cache Footprint** | `224× reduction` | **`CONSTANT`** | Theoretical analytical model | `research/DEEPSEEK_V4_FLASH_AND_SYMBOLM_CO_DESIGN.md` | Theoretical co-design formula ($\text{Tokens} \times \text{KV Bytes/Tok}$). Valid architectural derivation, but an analytical projection, not an empirical cluster benchmark. |
| **14** | **GGUF Binary Suite Sizes** | `999MB / 1.8GB / 3.3GB` | **`MEASURED`** | `scripts/convert_hf_to_gguf.py` | Local disk & Hugging Face Hub | Verified byte-for-byte on Hugging Face repositories `gyawalisanish0/SymboLM-2.1-DeepSeek-1.5B` and `SmolLM2-1.7B`. |
| **15** | **Dedicated Delimiters (3 tokens)** | `3 tokens (<reg:bypass>)`| **`MEASURED`** | `symbolm/tokenizer.py` | `tests/test_tokenizer_wrapper.py` | Verified token count across Qwen and LLaMA tokenizers with added special tokens. |
| **16** | **Unit Test Suite Pass Rate** | `26/26 tests passing` | **`MEASURED`** | `tests/` | Pytest logs | 45 / 45 unit tests currently pass (100% OK) under `uv run --with pytest pytest tests/`. |

---

## 3. Mandatory Remediation Directives

1. **Retract or Re-label All `SIMULATED` & `CONSTANT` Claims:**
   - Move `eval/ablation_5_arm_suite.py` to `eval/SIMULATED_ablation_5_arm_suite.py` with an unmistakable header disclaiming any evidential status.
   - Clarify `plot_adaptive_allocation.py` as an illustrative model, not measured empirical data.
2. **Execute Task 1 (Honest Scorer):**
   - Eliminate regex fallbacks and enforce strict truncation penalties via `eval/rescore.py`.
   - Publish the rescored accuracy delta transparently.
3. **Execute Task 2 (Real Ablation):**
   - Conduct actual forward-pass ablation on `HuggingFaceTB/SmolLM2-1.7B-Instruct` across $\ge 200$ held-out problems within free Kaggle GPU limits.
4. **Update `README.md` & `EVIDENCE_TRUTH_TABLE.md`:**
   - Remove "zero premature derivation cuts" and "3.9× increase in accuracy".
   - Replace with honest, audited empirical figures.
