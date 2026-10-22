# SymboLM Physical Hardware Valuation Report

**Principal Systems Architect & Research Director:** **Sanish Gyawali** (Rupandehi, Nepal)  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**Protocol Specification & Peer Review:** Claude (Haiku 5.5, Anthropic)  
**Date:** October 17, 2026  
**Pre-Registration File:** [`PREREG_VALUATION.md`](PREREG_VALUATION.md) (Locked prior to measurement)  
**Dataset Evaluated:** 200 clean held-out problems (`data/held_out_eval_200_clean.jsonl`)  

> [!WARNING]
> **Historical SmolLM2-1.7B pilot — partially retracted (2026-10-22).** This file predates the v3 strict declared-answer scorer and the v2 benchmark. It is a SmolLM2-1.7B pilot, not a DeepSeek flagship result and not the v3 rerun. The note below previously said “Format Value Confirmed,” which contradicted the table: the paired 95% CI includes zero and the table says `Adds Format Value: False`. The correct conclusion is **no format value established**. Current status: [`docs/EVIDENCE_STATUS.md`](docs/EVIDENCE_STATUS.md). New protocol: [`PREREG_VALUATION_V2.md`](PREREG_VALUATION_V2.md).

> [!NOTE]
> **Multi-Budget Grid Valuation:** The old 6-arm × 4-budget run stopped at 2,119 of 4,800 cells. Its 1,955 receipts are partial and were scored before the v3 strict scorer existed. The old “Format Value Confirmed” phrase is retracted; the report's own paired interval includes zero.

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
- Scoring used the Task 1 audit path available at the time. It is **not** the v3 strict declared-answer scorer and must not be pooled with v3 results.
- Free compute quotas strictly respected.