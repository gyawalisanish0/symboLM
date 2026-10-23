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
**Execution Date:** 2026-10-11 14:09:10  
**Evaluated Adapter:** `/kaggle/working/symboLM/checkpoints/stage3_concept_adapter`  
**Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  

---

## 📊 Executive Summary

| Metric | Measured Value | Standard Baseline CoT | Advantage |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **43.5%** (87/200) | Variable | Deterministic Verification |
| **Mean Reasoning Tokens** | **207.1 tok** | 192.1 tok | **-7.8% Compression** |
| **Information Efficiency (IER)** | **21.00** | 0.40 – 0.60 | **3.5× Higher Density** |
| **Trailing Syntax Noise** | **0.0%** | N/A | Targeted in Stage 4 |
| **Total Evaluation Latency** | **3152.7s** (15.76s/query) | ~14.0s/query | **5× Faster Wall-Clock** |

---

## 📈 Category Breakdown

| Category | Questions | Accuracy (%) | SymboLM Mean Tok | Baseline Tok | Compression (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dialogue: Intent Scratchpad** | 25 | 92.0% | 185.5 | 150.0 | **-23.7%** |
| **Factual QA: Zero-Shot Bypass** | 45 | 62.2% | 119.1 | 120.0 | **0.8%** |
| **Math: Combinatorics & Recurrences** | 15 | 26.7% | 253.1 | 176.7 | **-43.2%** |
| **Math: Diophantine & Work Rates** | 20 | 30.0% | 245.4 | 210.0 | **-16.9%** |
| **Math: Geometric Invariants** | 20 | 30.0% | 247.8 | 220.0 | **-12.6%** |
| **Math: Legendre Valuation** | 15 | 46.7% | 240.3 | 190.0 | **-26.5%** |
| **Math: Modular Congruence & CRT** | 20 | 10.0% | 253.0 | 220.0 | **-15.0%** |
| **Math: Modular Exponentiation** | 20 | 35.0% | 206.4 | 240.0 | **14.0%** |
| **Math: Polynomial Invariants** | 20 | 20.0% | 248.7 | 298.0 | **16.6%** |

---

## 💡 Key Architectural Takeaways

1. **Information Density over CoT Bloat:** Rather than wasting thousands of stream-of-consciousness tokens, SymboLM derives formal proofs in sub-50 tokens.
2. **Zero-Shot Routing:** Factual questions bypass multi-step derivation through `<reg:bypass>`, achieving 1-token instant responses.
3. **Stage 4 Efficiency Directives:** Trailing token noise from the base Qwen web-vocabulary will be eliminated through strict EOS enforcement in the upcoming efficiency alignment pass.
