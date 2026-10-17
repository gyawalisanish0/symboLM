# SymboLM Real 4-Arm Register Ablation Audit Report (Task 2)

**Principal Systems Architect & Research Director:** **Sanish Gyawali** (Rupandehi, Nepal)  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**Adversarial Peer Reviewer:** Claude (Haiku 5.5, Anthropic)  
**Date:** October 17, 2026  
**Total Problems Evaluated:** 200 clean held-out problems  
**Scoring Engine:** Task 1 Strict Honest Scorer (Zero regex tail fallbacks, truncation = FAIL)  

---

## 1. Executive Summary & Causal Scorecard

| Ablation Arm | Accuracy (Pass@1) | Wilson 95% CI | Mean Tokens | Compression vs CoT | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Arm A: Standard CoT** | **7.00%** | [4.2%, 11.4%] | 167.6 | Baseline (1.0×) | `MEASURED` |
| **Arm B: Answer Only** | **15.00%** | [10.7%, 20.6%] | 8.1 | 20.67× (-95.2%) | `MEASURED` |
| **Arm C: Full Registers** | **10.00%** | [6.6%, 14.9%] | 125.0 | 1.34× (-25.4%) | `MEASURED` |
| **Arm D: Shuffled Plan (Causal Control)** | **9.00%** | [5.8%, 13.8%] | 136.4 | 1.23× (-18.6%) | `MEASURED` |

---

## 2. Causal Register Analysis (Arm C vs. Arm D)

- **Arm C (Canonical Plan) Accuracy:** 10.00%
- **Arm D (Corrupted Plan) Accuracy:** 9.00%
- **Causal Accuracy Delta (Delta_causal):** +1.00%

> [!NOTE]
> In the Lanham et al. (2023) causal intervention framework, if injecting a corrupted concept tag collapses deductive performance, the model is causally dependent on `<reg:plan>`. If accuracy remains identical, the register is purely cosmetic formatting.

---

## 3. Methodological Integrity & Free Compute Compliance
- **Zero Simulation:** Every figure derived directly from live model forward passes.
- **Honest Scorer:** Zero trailing number regex guessing (`re.findall(r'[-+]?\d+')` eradicated).
- **Clean Split:** Evaluated on `data/held_out_eval_200_clean.jsonl` with 0.0% training contamination.
