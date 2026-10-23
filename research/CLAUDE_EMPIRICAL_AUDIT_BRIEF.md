> [!WARNING]
> **HISTORICAL / NON-v3 RECEIPT - NOT A CURRENT RESULT (2026-10-23).** This report predates the v3 strict declared-answer scorer and the token-efficiency protocol. Its scores are retained for audit, not as current claims. The old six-arm flagship run stopped at 2,119/4,800 cells because of Kaggle quota/infrastructure limits; it is incomplete, not a confirmatory or negative result.
>
> **Current framing:** token efficiency of correct reasoning is the primary lens;
> strict accuracy is a non-inferiority gate. The biological language is a design
> analogy, not a mechanism claim. Current status: [../docs/EVIDENCE_STATUS.md](../docs/EVIDENCE_STATUS.md). Canonical framing:
> [../docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md](../docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md).

# Claude Haiku 5.5 Empirical Review & Adversarial Audit Brief

**Reviewer:** Claude (Haiku 5.5, Anthropic)  
**Author & Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**Target Repository:** [`github.com/gyawalisanish0/symboLM`](https://github.com/gyawalisanish0/symboLM)  
**Date:** October 17, 2026  

---

## 1. Executive Summary & Review Intent

Claude Haiku 5.5 conducted an in-depth adversarial peer review of the SymboLM codebase, empirical logs, and benchmark claims. The audit established that while the core architectural vision (Tri-Register reasoning, grammar constraints, SymPy tiny terminal, and token parsimony) is conceptually brilliant, several published figures, ablation scripts, and benchmark claims in early reports relied on simulations, constant baseline assumptions, or lenient scoring fallbacks rather than strictly isolated empirical measurements.

---

## 2. Non-Negotiable Operational Constraints

1. **Free Compute Only:** All experiments must execute within Kaggle free GPU quotas (Dual Tesla T4 / P100, 30h/week), Colab free tier, or local Host CPU. Never assume paid GPUs, paid APIs, or cloud credits.
2. **Absolute Empirical Provenance:** Do not hard-code, estimate, or copy any number into a report unless an executable script produced it and a raw results JSON records it.
3. **Receipt Immutability:** Do not overwrite existing result JSONs or receipts. Write new files with versioned names.
4. **Branch Governance:** Work on a dedicated audit branch (`audit/empirical-hardening`). Do not merge to `main` without Sanish Gyawali's approval.
5. **Data Protection:** Ask Sanish before deleting any file.

---

## 3. Findings & Forensic Discrepancies

1. **`eval/ablation_5_arm_suite.py` was a Simulation:**
   - Lines 36–60 utilized `simulate_arm_execution` with `random.randint` and hard-coded accuracy scalars (e.g., `0.70`, `0.65`, `0.95`).
   - `eval/ablation_5_arm_results.json` reflects pseudo-random outputs rather than live model forwards.
2. **`eval/plot_adaptive_allocation.py` Used Constant Baselines:**
   - Baselines for standard Long-CoT models were hard-coded constants (`580`, `720`, `940`, `1250` tokens) rather than measured runs on identical questions.
3. **Constant Assumptions in Multi-Year Benchmark Generator:**
   - `data/build_multi_year_1000_benchmark.py` assigned `english_baseline_tokens` as fixed constants per item.
4. **1,000-Question Token Guillotine & "Zero Premature Cuts" Discrepancy:**
   - In `eval/eval_v2_1000_results.json`, **917 out of 1,000 outputs (91.7%)** reached the `max_tokens=512` cap.
   - The README claim of "zero premature cuts" directly conflicted with raw output cutoffs at `</thi`.
5. **Scorer Lenience (Trailing Number Fallbacks):**
   - Fallback logic extracted trailing integers from truncated generations, crediting correct answers even when the model never closed its derivation or reached `<ans>`.
6. **Verifier Register Degeneracy in v2.0/v2.1 Evaluation:**
   - In the 1,000-question run, the string `"bands:"` repeated in **823 of 1,000 raw outputs**, and `<reg:verify>` emitted zero instances of `✓` or `✗`.
7. **Curriculum & Benchmark Contamination:**
   - 47 benchmark prompts appeared verbatim in the 5,000-sample training set (`concept_math_contrastive_v2_1.jsonl`).
8. **Relative Comparison Conflation:**
   - The "3.9× accuracy" claim compared v2.1 against v2.0 (which was artificially strangled by a 192-token Guillotine), rather than comparing SymboLM directly against an unconstrained CoT baseline.
9. **Sample Size Discrepancies:**
   - Preliminary MATH-500 comparison had $n=5$.
10. **Cross-Architecture Claims:**
    - SmolLM2 "100% register fidelity" was asserted based on 5 curated test prompts.

---

## 4. Priority Execution Queue

- **Task 0 (Audit):** Author `AUDIT.md` cataloging every number in the README, labeled as `MEASURED`, `SIMULATED`, `CONSTANT`, or `UNVERIFIED`.
- **Task 1 (Honest Scorer):** Implement `eval/rescore.py` rejecting truncated outputs, removing trailing-number guesses, and calculating true accuracy delta.
- **Task 2 (Real Ablation):** Rename simulated script to `eval/SIMULATED_ablation_5_arm_suite.py` with disclaimer. Implement real forward-pass ablation `eval/ablation_real.py` on SmolLM2 with 4 arms on held-out math split ($\ge 200$ items).
- **Task 3 (Verifier Falsification):** Inject arithmetic corruption into 100 deduction steps and measure empirical $P(✗)$ emission rate.
- **Task 4 (Contamination Audit):** Measure exact and n-gram overlap between benchmark and training curriculum; report clean vs. contaminated accuracy.
- **Task 5 (Cross-Family Retest):** Run SmolLM2 adapter across full held-out benchmark split.
- **Task 6 (Environment & Tests):** Pin dependencies and verify native unit tests pass 100%.
- **Task 7 (Honest Public Facing):** Author `RESULTS_SUMMARY.md` and update `README.md` with an unshakeable Evidence Truth Table.

---

## 5. Hardware Valuation Protocol (Joules per Correct Answer)

**Objective:** Measure what the register format is worth on real hardware in physical energy and latency per correct answer ($J/\text{correct}$), comparing against a router-only baseline to separate the value of routing from the value of the format.

### A. Pre-Registration Mandate (`PREREG_VALUATION.md`)
Prior to running any physical measurement, commit `PREREG_VALUATION.md` declaring:
- **Primary Metric:** Dynamic energy per correct answer ($J/\text{correct}$).
- **Required Token Budget Grid:** Evaluate every arm at budgets **256, 512, 1024, and 2048 tokens**. Truncated outputs without formal closures scored strictly as **FAIL (0)**.
- **Pre-Registered Decision Rule:** Arm C (Full Registers) counts as adding value over Arm D (Router-Only) ONLY IF:
  1. Arm C beats Arm D on $J/\text{correct}$ at every budget where D's accuracy is within 2 points of its own peak.
  2. The paired bootstrap 95% CI on $(J/\text{correct}_D - J/\text{correct}_C)$ strictly excludes zero on identical problems.
  - *If C and D are within noise, the conclusion is strictly that the value is routing alone.*
  - *Report all budgets transparently, including those where C loses.*

### B. Base Model Hierarchy (In Order of Priority)
1. **Primary Flagship:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. Foundation of all headline results; maintains repo continuity. Target: phone and local CPU.
2. **Scale Point:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (Q4_K_M on laptop CPU, plus 4-bit LoRA on Kaggle T4 if quota allows). Target: laptop.
3. **Cross-Family Reasoning Control:** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (4-bit). Different architecture (Meta LLaMA), same reasoning training. *Do not use SmolLM2 as the control, because it is not a reasoning model.*
- *Quota Rule:* If quota is tight, run all 5 arms on the 1.5B only, and arms A and C on the 7B.

### C. The 5 Valuation Arms
- **Arm A:** Ordinary CoT, greedy.
- **Arm B:** Answer-only.
- **Arm C:** Full Register format (Router + Plan + Deduce + Verify).
- **Arm D (Causal Router Control):** Router-only. Shares identical router weights with Arm C. Easy queries get `<reg:bypass>`; hard queries get plain CoT with zero registers. Isolates the format value from routing.
- **Arm E:** CoT with Python calculator. Generated code executed in a sandboxed subprocess with timeout and zero network access.

### D. Hardware Targets & Sampling
- **Kaggle T4 GPU:** Query `nvidia-smi --query-gpu=power.draw` every 100 ms during generation; subtract idle baseline to compute dynamic Joules.
- **Local Host CPU:** Microsecond timing; report execution time and label energy as `ESTIMATE` with stated TDP.
- **Outputs:** `results/valuation/` raw logs, `VALUATION.md` tables and Pareto frontier plots (Accuracy vs. $J/\text{correct}$ across token budgets).


