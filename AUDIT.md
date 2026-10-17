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
   - Renamed `eval/ablation_5_arm_suite.py` to `eval/SIMULATED_ablation_5_arm_suite.py` with an unmistakable disclaimer header.
   - Replaced simulation with real forward-pass harness `eval/ablation_real.py`.
2. **Execute Task 1 (Honest Scorer):**
   - Eliminated regex fallbacks and enforced strict truncation penalties via `eval/rescore.py`.
   - Published rescored accuracy delta transparently in `eval/RESCORING_AUDIT_REPORT.md`.
3. **Execute Task 2 (Real Ablation):**
   - Conducted live forward passes on 200 clean held-out problems via `eval/ablation_real.py`.
4. **Update `README.md` & `EVIDENCE_TRUTH_TABLE.md`:**
   - Removed unverified claims and replaced with audited, measured physical receipts.

---

## 4. Completed Remediation Status & Audited Receipts

| Remediation Task | Description | Status | Verifiable Receipt Artifact | Key Empirical Finding |
| :--- | :--- | :---: | :--- | :--- |
| **Task 0: Audit** | Full claims & numbers audit | **`COMPLETE`** | [`AUDIT.md`](AUDIT.md) | Cataloged all claims into 4 epistemic classes. |
| **Task 1: Honest Scorer** | Eradicate regex tail-guessing & penalize truncation | **`COMPLETE`** | [`eval/RESCORING_AUDIT_REPORT.md`](eval/RESCORING_AUDIT_REPORT.md) | True strict accuracy is 3.2%; 91.7% truncated at 512 cap; 80% reflex bypass verified. |
| **Task 2: Real Ablation** | Forward-pass 4-arm ablation on clean split | **`COMPLETE`** | [`eval/ABLATION_REAL_REPORT.md`](eval/ABLATION_REAL_REPORT.md) | CoT 7.0%, Answer-Only 15.0%, Full Reg 10.0%, Shuffled 9.0% ($\Delta_{\text{causal}} = +1.00\%$). |
| **Task 3: Verifier Falsification** | Measure rejection rate on corrupted math | **`COMPLETE`** | [`eval/VERIFIER_FALSIFICATION_REPORT.md`](eval/VERIFIER_FALSIFICATION_REPORT.md) | Autoregressive $P(✗) = 0.0\%$; validates sandboxed SymPy Tiny Terminal. |
| **Task 4: Contamination Audit** | Measure prompt and n-gram overlap | **`COMPLETE`** | [`eval/CONTAMINATION_AUDIT_REPORT.md`](eval/CONTAMINATION_AUDIT_REPORT.md) | 47 prompt leaks (4.7%); extracted 200 clean held-out problems (`data/held_out_eval_200_clean.jsonl`). |
| **Hardware Valuation** | NVML power profiling on Dual Tesla T4 | **`COMPLETE`** | [`VALUATION.md`](VALUATION.md) | Idle baseline 32.60 W; Arm C saves 38.3% Joules vs CoT; Arm E saves 78.4% Joules with 3.3× accuracy. |
| **Pre-Registration** | Lock hypothesis & decision rule | **`LOCKED`** | [`PREREG_VALUATION.md`](PREREG_VALUATION.md) | Committed before physical silicon measurements; locked paired bootstrap 95% CI rule. |
| **Task 7: Master Summary** | Public-facing report for research grants | **`COMPLETE`** | [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) | Consolidated truth table and hardware receipts for Emergent Ventures & 1517 Fund. |

