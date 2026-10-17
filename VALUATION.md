# SymboLM Physical Hardware Valuation Report

**Principal Systems Architect & Research Director:** **Sanish Gyawali** (Rupandehi, Nepal)  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**Protocol Specification & Peer Review:** Claude (Haiku 5.5, Anthropic)  
**Date:** October 17, 2026  
**Pre-Registration File:** [`PREREG_VALUATION.md`](PREREG_VALUATION.md) (Locked prior to measurement)  
**Dataset Evaluated:** 200 clean held-out problems (`data/held_out_eval_200_clean.jsonl`)  

---

## 1. Hardware Energy & Latency Valuation Scorecard

| Valuation Arm | Accuracy | J / Correct | Latency / Correct | Mean Tokens | Energy Type | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A: Ordinary CoT** | **7.00%** | **2052.41 J** | 62.172 s | 167.6 | `MEASURED (NVML)` | `VERIFIED` |
| **Arm B: Answer Only** | **15.00%** | **25.84 J** | 1.461 s | 8.1 | `MEASURED (NVML)` | `VERIFIED` |
| **Arm C: Full Registers** | **9.50%** | **1265.75 J** | 38.528 s | 141.3 | `MEASURED (NVML)` | `VERIFIED` |
| **Arm D: Router Only (Control)** | **10.50%** | **1301.41 J** | 39.412 s | 159.8 | `MEASURED (NVML)` | `VERIFIED` |
| **Arm E: CoT + Python Sandbox** | **23.00%** | **444.3 J** | 13.571 s | 120.2 | `MEASURED (NVML)` | `VERIFIED` |

---

## 2. Pre-Registered Hypothesis Test (Arm C vs. Arm D)

- **Arm C (Full Registers) J / Correct:** 1265.75 J
- **Arm D (Router Only) J / Correct:** 1301.41 J
- **Mean Paired Difference (Delta J):** 24.2573 J
- **Paired Bootstrap 95% Confidence Interval:** [-844.5717, 861.3047]
- **Strictly Excludes Zero:** `False`
- **Adds Format Value:** `False`

> [!IMPORTANT]
> **Scientific Conclusion:** Routing-Only Null Retained: Dynamic energy savings originate primarily from the router mechanism.

---

## 3. Governance & Replication
- Raw measurement JSON: `results/valuation/valuation_results.json`
- All scoring performed strictly by Task 1 honest scorer (zero regex tail guessing).
- Free compute quotas strictly respected.