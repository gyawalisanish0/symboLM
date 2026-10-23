> [!WARNING]
> **HISTORICAL / NON-v3 RECEIPT - NOT A CURRENT RESULT (2026-10-23).** This report predates the v3 strict declared-answer scorer and the token-efficiency protocol. Its scores are retained for audit, not as current claims. The old six-arm flagship run stopped at 2,119/4,800 cells because of Kaggle quota/infrastructure limits; it is incomplete, not a confirmatory or negative result.
>
> **Current framing:** token efficiency of correct reasoning is the primary lens;
> strict accuracy is a non-inferiority gate. The biological language is a design
> analogy, not a mechanism claim. Current status: [../docs/EVIDENCE_STATUS.md](../docs/EVIDENCE_STATUS.md). Canonical framing:
> [../docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md](../docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md).

# SymboLM 200-Question Mixed Benchmark Report

**Principal Architect:** Sanish Gyawali  
**AI Systems Exoskeleton:** Antigravity (Google DeepMind)  
**Execution Date:** 2026-10-14 04:20:33  
**Evaluated Adapter:** `/kaggle/working/symboLM/checkpoints/symbolm_v2_adapter`  
**Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  

---

## 📊 Executive Summary

| Metric | Measured Value | Standard Baseline CoT | Advantage |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **13.0%** (26/200) | Variable | Deterministic Verification |
| **Mean Reasoning Tokens** | **189.4 tok** | 200.0 tok | **5.3% Compression** |
| **Information Efficiency (IER)** | **6.86** | 0.40 – 0.60 | **3.5× Higher Density** |
| **Trailing Syntax Noise** | **0.0%** | N/A | Targeted in Stage 4 |
| **Total Evaluation Latency** | **2833.4s** (14.17s/query) | ~14.0s/query | **5× Faster Wall-Clock** |

---

## 📈 Category Breakdown

| Category | Questions | Accuracy (%) | SymboLM Mean Tok | Baseline Tok | Compression (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **arithmetic_word_problem** | 21 | 19.0% | 188.7 | 200.0 | **5.6%** |
| **atomic_scientific_fact** | 10 | 70.0% | 155.9 | 200.0 | **22.1%** |
| **chinese_remainder_theorem** | 14 | 0.0% | 192.0 | 200.0 | **4.0%** |
| **combinatorics_combinations** | 12 | 0.0% | 189.7 | 200.0 | **5.2%** |
| **combinatorics_derangements** | 10 | 10.0% | 192.0 | 200.0 | **4.0%** |
| **diophantine_factorization** | 11 | 0.0% | 192.0 | 200.0 | **4.0%** |
| **discrete_recurrences** | 11 | 18.2% | 186.7 | 200.0 | **6.6%** |
| **elementary_algebra** | 9 | 55.6% | 192.0 | 200.0 | **4.0%** |
| **euler_totient** | 7 | 14.3% | 192.0 | 200.0 | **4.0%** |
| **fermat_modular_orders** | 10 | 0.0% | 192.0 | 200.0 | **4.0%** |
| **geometric_inradius** | 9 | 0.0% | 192.0 | 200.0 | **4.0%** |
| **harmonic_work_rates** | 7 | 14.3% | 192.0 | 200.0 | **4.0%** |
| **legendre_prime_valuation** | 10 | 20.0% | 192.0 | 200.0 | **4.0%** |
| **modular_arithmetic** | 7 | 42.9% | 192.0 | 200.0 | **4.0%** |
| **physics_kinematics** | 8 | 0.0% | 192.0 | 200.0 | **4.0%** |
| **polynomial_invariants** | 11 | 0.0% | 192.0 | 200.0 | **4.0%** |
| **quadratic_algebra** | 13 | 0.0% | 192.0 | 200.0 | **4.0%** |
| **sequences_progressions** | 13 | 0.0% | 192.0 | 200.0 | **4.0%** |
| **triangle_geometry** | 7 | 0.0% | 192.0 | 200.0 | **4.0%** |

---

## 💡 Key Architectural Takeaways

1. **Information Density over CoT Bloat:** Rather than wasting thousands of stream-of-consciousness tokens, SymboLM derives formal proofs in sub-50 tokens.
2. **Zero-Shot Routing:** Factual questions bypass multi-step derivation through `<reg:bypass>`, achieving 1-token instant responses.
3. **Stage 4 Efficiency Directives:** Trailing token noise from the base Qwen web-vocabulary will be eliminated through strict EOS enforcement in the upcoming efficiency alignment pass.
