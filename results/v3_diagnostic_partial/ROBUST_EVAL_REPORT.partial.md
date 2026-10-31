> **INCOMPLETE — NOT A RESULT.** This report was generated from 2,280 of 2,400 expected cells. It was stopped by Kaggle infrastructure limits, has no wrapper receipt, and must not be used as a benchmark or claim. See PARTIAL_STATUS.md.

# SymboLM Robust Evaluation Report

**Run ID:** `robust_eval_20261031T022945Z`  
**Run kind:** `diagnostic`  
**Commit:** `17d62d1bc8e26f6a02bee67e777ebf8984a0ca02`  
**Dataset SHA-256 verified:** `True`  
**Dataset SHA-256:** `F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707`  
**Scoring:** strict declared answer (`<ans>` / `<reg:bypass>`; truncation without EOS = fail; no deduce-equality fallback)  
**Ground-truth overrides:** `/kaggle/working/symboLM/data/ground_truth_overrides.json`  
**Adapter SHA-256:** `204823BC216BBF7329D6C7C5159ACF8F6688FD321AD1652757C2A9A9D066B31C` (match: `True`)  
**Subset:** `proportional_stratified_sha256` n=`100` seed=`20261025`  
**Expected cells:** 2400  
**Energy:** measured NVML dynamic power  
**Duplicate-prompt rows:** 408/2280  
**Timestamp (UTC):** 2026-10-31T02:29:55.902951+00:00  
**GPU:** Tesla T4  
**Arms evaluated:** A, B, C, D, E, F  
**Budgets:** [256, 512, 1024, 2048]  

> **Diagnostic only.** This is a deterministic subset run, not the 200-problem x 4-budget x 6-arm preregistered experiment. It cannot be reported as the v3 benchmark result.


## 1. Correctness–Token–Energy Pareto Frontier

Format: `Acc% [CI-lo, CI-hi] | tok/correct | J/correct | mean-tok | trunc`

| Budget | Arm A: Ordinary CoT | Arm B: Answer-Only | Arm C: Full Registers | Arm D: Router-Only Control | Arm E: CoT + Python Sandbox | Arm F: Vanilla DeepSeek (No Adapter) |
|--------|---|---|---|---|---|---|
| **256** | 15.8% [9.8,24.4] \| 1427.3 tok/✓ \| 3728.2 J/✓ \| 225tok \| 63trunc | 29.5% [21.2,39.3] \| 725.3 tok/✓ \| 1852.1 J/✓ \| 214tok \| 52trunc | 62.1% [52.1,71.2] \| 181.4 tok/✓ \| 508.9 J/✓ \| 113tok \| 0trunc | 18.9% [12.3,28.0] \| 1169.2 tok/✓ \| 2951.7 J/✓ \| 222tok \| 63trunc | 20.0% [13.2,29.1] \| 938.9 tok/✓ \| 2430.9 J/✓ \| 188tok \| 35trunc | 2.1% [0.6,7.3] \| 12122.5 tok/✓ \| 25834.2 J/✓ \| 255tok \| 93trunc |
| **512** | 50.5% [40.6,60.4] \| 665.9 tok/✓ \| 1541.4 J/✓ \| 336tok \| 19trunc | 64.2% [54.2,73.1] \| 481.4 tok/✓ \| 1146.7 J/✓ \| 309tok \| 16trunc | 62.1% [52.1,71.2] \| 181.4 tok/✓ \| 429.2 J/✓ \| 113tok \| 0trunc | 53.7% [43.7,63.4] \| 619.6 tok/✓ \| 1478.7 J/✓ \| 333tok \| 19trunc | 34.7% [25.9,44.7] \| 666.5 tok/✓ \| 1573.2 J/✓ \| 232tok \| 8trunc | 31.6% [23.1,41.5] \| 1475.0 tok/✓ \| 2892.4 J/✓ \| 466tok \| 63trunc |
| **1024** | 62.1% [52.1,71.2] \| 598.5 tok/✓ \| 1438.9 J/✓ \| 372tok \| 0trunc | 75.8% [66.3,83.3] \| 459.3 tok/✓ \| 1124.3 J/✓ \| 348tok \| 2trunc | 62.1% [52.1,71.2] \| 181.4 tok/✓ \| 429.4 J/✓ \| 113tok \| 0trunc | 65.3% [55.3,74.1] \| 563.7 tok/✓ \| 1369.5 J/✓ \| 368tok \| 0trunc | 39.0% [29.8,49.0] \| 643.3 tok/✓ \| 1558.5 J/✓ \| 250tok \| 3trunc | 49.5% [39.6,59.4] \| 1538.2 tok/✓ \| 3053.0 J/✓ \| 761tok \| 45trunc |
| **2048** | 62.1% [52.1,71.2] \| 598.5 tok/✓ \| 1445.7 J/✓ \| 372tok \| 0trunc | 75.8% [66.3,83.3] \| 460.4 tok/✓ \| 1127.9 J/✓ \| 349tok \| 0trunc | 62.1% [52.1,71.2] \| 181.4 tok/✓ \| 427.8 J/✓ \| 113tok \| 0trunc | 65.3% [55.3,74.1] \| 563.7 tok/✓ \| 1372.8 J/✓ \| 368tok \| 0trunc | 39.0% [29.8,49.0] \| 726.3 tok/✓ \| 1868.0 J/✓ \| 283tok \| 3trunc | 75.8% [66.3,83.3] \| 1502.2 tok/✓ \| 2975.8 J/✓ \| 1138tok \| 20trunc |

### 1.0 Strict Score Diagnostics

Strict accuracy is the primary metric. Lenient accuracy is the old fallback-based number and is shown only to expose the gap. EOS and format rates are computed only where the run recorded them.

| Budget | Arm | Strict Acc | Lenient Acc | EOS rate | Valid format | Declared answer | Truncated |
|--------|-----|-----------|-------------|----------|--------------|-----------------|-----------|
| 256 | A | 15.8% | 24.2% | 33.7% | 20.0% | 20.0% | 63 |
| 256 | B | 29.5% | 29.5% | 45.3% | 35.8% | 35.8% | 52 |
| 256 | C | 62.1% | 62.1% | 100.0% | 100.0% | 100.0% | 0 |
| 256 | D | 18.9% | 27.4% | 33.7% | 23.2% | 23.2% | 63 |
| 256 | E | 20.0% | 35.8% | 63.2% | 29.5% | 29.5% | 35 |
| 256 | F | 2.1% | 2.1% | 2.1% | 2.1% | 2.1% | 93 |
| 512 | A | 50.5% | 57.9% | 80.0% | 61.1% | 61.1% | 19 |
| 512 | B | 64.2% | 65.3% | 83.2% | 73.7% | 73.7% | 16 |
| 512 | C | 62.1% | 62.1% | 100.0% | 100.0% | 100.0% | 0 |
| 512 | D | 53.7% | 61.1% | 80.0% | 64.2% | 64.2% | 19 |
| 512 | E | 34.7% | 48.4% | 91.6% | 49.5% | 49.5% | 8 |
| 512 | F | 31.6% | 31.6% | 33.7% | 32.6% | 32.6% | 63 |
| 1024 | A | 62.1% | 69.5% | 100.0% | 80.0% | 80.0% | 0 |
| 1024 | B | 75.8% | 75.8% | 97.9% | 88.4% | 88.4% | 2 |
| 1024 | C | 62.1% | 62.1% | 100.0% | 100.0% | 100.0% | 0 |
| 1024 | D | 65.3% | 72.6% | 100.0% | 83.2% | 83.2% | 0 |
| 1024 | E | 39.0% | 52.6% | 96.8% | 53.7% | 53.7% | 3 |
| 1024 | F | 49.5% | 50.5% | 52.6% | 50.5% | 50.5% | 45 |
| 2048 | A | 62.1% | 69.5% | 100.0% | 80.0% | 80.0% | 0 |
| 2048 | B | 75.8% | 75.8% | 100.0% | 90.5% | 90.5% | 0 |
| 2048 | C | 62.1% | 62.1% | 100.0% | 100.0% | 100.0% | 0 |
| 2048 | D | 65.3% | 72.6% | 100.0% | 83.2% | 83.2% | 0 |
| 2048 | E | 39.0% | 52.6% | 96.8% | 53.7% | 53.7% | 3 |
| 2048 | F | 75.8% | 83.2% | 79.0% | 76.8% | 76.8% | 20 |

### 1.1 Test-Time Compute & Overthinking Metrics (OptimalThinkingBench & LLMTHINKBENCH)

Overthinking-Adjusted Accuracy (OAA), Area Under Curve (AUC_OAA), and Harmonic Overthinking Score (O).
Formulation: O = 2 * (Acc * TokenEff) / (Acc + TokenEff). Penalizes runaway token bloat and rewards edge efficiency.

| Arm | Budget | Pass@1 | Mean Tok | Tok/correct | Correct/1k tok | OAA@256 | AUC_OAA | TokenEff | Overthinking Score (O) |
|-----|--------|--------|----------|-------------|----------------|---------|---------|----------|------------------------|
| Arm A: Ordinary CoT | 256 | 15.8% | 225.4tok | 1427.3 | 0.701 | 15.8% | 0.144 | 1.00 | **0.273** |
| Arm A: Ordinary CoT | 512 | 50.5% | 336.5tok | 665.9 | 1.502 | 15.8% | 0.426 | 0.76 | **0.607** |
| Arm A: Ordinary CoT | 1024 | 62.1% | 371.7tok | 598.5 | 1.671 | 15.8% | 0.498 | 0.69 | **0.653** |
| Arm A: Ordinary CoT | 2048 | 62.1% | 371.7tok | 598.5 | 1.671 | 15.8% | 0.498 | 0.69 | **0.653** |
| Arm B: Answer-Only | 256 | 29.5% | 213.8tok | 725.3 | 1.379 | 29.5% | 0.274 | 1.00 | **0.455** |
| Arm B: Answer-Only | 512 | 64.2% | 309.1tok | 481.4 | 2.077 | 29.5% | 0.556 | 0.83 | **0.723** |
| Arm B: Answer-Only | 1024 | 75.8% | 348.1tok | 459.3 | 2.177 | 29.5% | 0.628 | 0.74 | **0.746** |
| Arm B: Answer-Only | 2048 | 75.8% | 348.9tok | 460.4 | 2.172 | 29.5% | 0.628 | 0.73 | **0.746** |
| Arm C: Full Registers | 256 | 62.1% | 112.6tok | 181.4 | 5.514 | 62.1% | 0.588 | 1.00 | **0.766** |
| Arm C: Full Registers | 512 | 62.1% | 112.6tok | 181.4 | 5.514 | 62.1% | 0.588 | 1.00 | **0.766** |
| Arm C: Full Registers | 1024 | 62.1% | 112.6tok | 181.4 | 5.514 | 62.1% | 0.588 | 1.00 | **0.766** |
| Arm C: Full Registers | 2048 | 62.1% | 112.6tok | 181.4 | 5.514 | 62.1% | 0.588 | 1.00 | **0.766** |
| Arm D: Router-Only Control | 256 | 18.9% | 221.5tok | 1169.2 | 0.855 | 18.9% | 0.176 | 1.00 | **0.319** |
| Arm D: Router-Only Control | 512 | 53.7% | 332.6tok | 619.6 | 1.614 | 18.9% | 0.458 | 0.77 | **0.632** |
| Arm D: Router-Only Control | 1024 | 65.3% | 367.9tok | 563.7 | 1.774 | 18.9% | 0.530 | 0.70 | **0.674** |
| Arm D: Router-Only Control | 2048 | 65.3% | 367.9tok | 563.7 | 1.774 | 18.9% | 0.530 | 0.70 | **0.674** |
| Arm E: CoT + Python Sandbox | 256 | 20.0% | 187.8tok | 938.9 | 1.065 | 20.0% | 0.183 | 1.00 | **0.333** |
| Arm E: CoT + Python Sandbox | 512 | 34.7% | 231.5tok | 666.5 | 1.500 | 20.0% | 0.303 | 1.00 | **0.516** |
| Arm E: CoT + Python Sandbox | 1024 | 39.0% | 250.5tok | 643.3 | 1.555 | 20.0% | 0.330 | 1.00 | **0.561** |
| Arm E: CoT + Python Sandbox | 2048 | 39.0% | 282.9tok | 726.3 | 1.377 | 20.0% | 0.330 | 0.91 | **0.545** |
| Arm F: Vanilla DeepSeek (No Adapter) | 256 | 2.1% | 255.2tok | 12122.5 | 0.082 | 2.1% | 0.019 | 1.00 | **0.041** |
| Arm F: Vanilla DeepSeek (No Adapter) | 512 | 31.6% | 465.8tok | 1475.0 | 0.678 | 2.1% | 0.259 | 0.55 | **0.401** |
| Arm F: Vanilla DeepSeek (No Adapter) | 1024 | 49.5% | 761.0tok | 1538.2 | 0.650 | 2.1% | 0.370 | 0.34 | **0.401** |
| Arm F: Vanilla DeepSeek (No Adapter) | 2048 | 75.8% | 1138.5tok | 1502.2 | 0.666 | 2.1% | 0.436 | 0.22 | **0.347** |

### SymPy Dual-Layer Bonus

Additional answers rescued by Layer 2 symbolic equality (would be FAIL in string-only mode):

| Arm | Budget | SymPy-rescued correct answers |
|-----|--------|-------------------------------|

## 2. Effect Sizes (Cohen's d + CLES)

Cohen's d on J/correct per arm-pair. CLES = P(C cheaper than comparison arm).
Positive d = arm C has lower J/correct (better). |d| > 0.8 = large effect.

| Budget | Comparison | Cohen's d (C vs X) | CLES | Bootstrap 95% CI (J/correct_X - J/correct_C) | Excludes Zero |
|--------|-----------|-------------------|------|---------------------------------------------|---------------|
| 256 | Arm A: Ordinary CoT | -1.5867 | 0.8847 | [1971.66, 6195.13] | ✓ |
| 256 | Arm B: Answer-Only | -0.7058 | 0.6568 | [862.87, 2224.49] | ✓ |
| 256 | Arm D: Router-Only Control | -0.7669 | 0.6667 | [1526.37, 4460.96] | ✓ |
| 256 | Arm E: CoT + Python Sandbox | -0.9292 | 0.7119 | [1182.44, 3488.57] | ✓ |
| 256 | Arm F: Vanilla DeepSeek (No Adapter) | -2.0445 | 0.8983 | [9600.21, 52694.67] | ✓ |
| 512 | Arm A: Ordinary CoT | -2.5605 | 0.9718 | [823.98, 1512.06] | ✓ |
| 512 | Arm B: Answer-Only | -1.6218 | 0.8536 | [537.62, 957.43] | ✓ |
| 512 | Arm D: Router-Only Control | -2.0061 | 0.9 | [779.86, 1418.09] | ✓ |
| 512 | Arm E: CoT + Python Sandbox | -1.7939 | 0.8701 | [798.85, 1718.87] | ✓ |
| 512 | Arm F: Vanilla DeepSeek (No Adapter) | -4.4913 | 0.9977 | [1747.63, 3745.59] | ✓ |
| 1024 | Arm A: Ordinary CoT | -2.1479 | 0.9776 | [779.22, 1304.57] | ✓ |
| 1024 | Arm B: Answer-Only | -1.5278 | 0.8757 | [530.67, 899.46] | ✓ |
| 1024 | Arm D: Router-Only Control | -1.8711 | 0.918 | [724.17, 1208.03] | ✓ |
| 1024 | Arm E: CoT + Python Sandbox | -1.7222 | 0.8846 | [779.88, 1673.93] | ✓ |
| 1024 | Arm F: Vanilla DeepSeek (No Adapter) | -2.4462 | 0.9982 | [1952.52, 3603.92] | ✓ |
| 2048 | Arm A: Ordinary CoT | -2.1824 | 0.9767 | [788.05, 1314.86] | ✓ |
| 2048 | Arm B: Answer-Only | -1.5311 | 0.8752 | [532.90, 910.83] | ✓ |
| 2048 | Arm D: Router-Only Control | -1.8656 | 0.9183 | [727.52, 1215.14] | ✓ |
| 2048 | Arm E: CoT + Python Sandbox | -1.7348 | 0.8823 | [850.09, 2375.73] | ✓ |
| 2048 | Arm F: Vanilla DeepSeek (No Adapter) | -1.7716 | 0.9991 | [2007.11, 3229.65] | ✓ |

## 3. Per-Category Accuracy with Wilson 95% CIs

### Budget = 256 tokens

| Category | N | A: Acc [CI] | B: Acc [CI] | C: Acc [CI] | D: Acc [CI] | E: Acc [CI] | F: Acc [CI] |
|----------|---|---|---|---|---|---|---|
| `arithmetic_word_problem` | 9 | 11% [2,44] | 33% [12,65] | 56% [27,81] | 11% [2,44] | 44% [19,73] | 0% [0,30] |
| `atomic_scientific_fact` | 4 | 25% [5,70] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N | 100% [51,100] ⚠️LOW_N | 100% [51,100] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N |
| `chinese_remainder_theorem` | 5 | 20% [4,62] | 20% [4,62] | 40% [12,77] | 20% [4,62] | 0% [0,43] | 0% [0,43] |
| `combinatorics_combinations` | 5 | 0% [0,43] | 20% [4,62] | 100% [57,100] | 0% [0,43] | 20% [4,62] | 0% [0,43] |
| `combinatorics_derangements` | 4 | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 75% [30,95] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N |
| `diophantine_factorization` | 5 | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
| `discrete_recurrences` | 5 | 60% [23,88] | 80% [38,96] | 100% [57,100] | 60% [23,88] | 0% [0,43] | 0% [0,43] |
| `elementary_algebra` | 5 | 80% [38,96] | 100% [57,100] | 100% [57,100] | 80% [38,96] | 60% [23,88] | 0% [0,43] |
| `euler_totient` | 4 | 0% [0,49] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N |
| `fermat_modular_orders` | 4 | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N |
| `geometric_inradius` | 5 | 0% [0,43] | 40% [12,77] | 100% [57,100] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
| `harmonic_work_rates` | 5 | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
| `legendre_prime_valuation` | 5 | 0% [0,43] | 0% [0,43] | 80% [38,96] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
| `modular_arithmetic` | 5 | 20% [4,62] | 40% [12,77] | 40% [12,77] | 20% [4,62] | 0% [0,43] | 20% [4,62] |
| `physics_kinematics` | 5 | 40% [12,77] | 0% [0,43] | 100% [57,100] | 40% [12,77] | 0% [0,43] | 0% [0,43] |
| `polynomial_invariants` | 5 | 0% [0,43] | 20% [4,62] | 60% [23,88] | 0% [0,43] | 40% [12,77] | 0% [0,43] |
| `quadratic_algebra` | 5 | 0% [0,43] | 100% [57,100] | 60% [23,88] | 0% [0,43] | 60% [23,88] | 0% [0,43] |
| `sequences_progressions` | 5 | 40% [12,77] | 0% [0,43] | 80% [38,96] | 40% [12,77] | 40% [12,77] | 0% [0,43] |
| `triangle_geometry` | 5 | 0% [0,43] | 20% [4,62] | 40% [12,77] | 0% [0,43] | 20% [4,62] | 0% [0,43] |

### Budget = 512 tokens

| Category | N | A: Acc [CI] | B: Acc [CI] | C: Acc [CI] | D: Acc [CI] | E: Acc [CI] | F: Acc [CI] |
|----------|---|---|---|---|---|---|---|
| `arithmetic_word_problem` | 9 | 89% [56,98] | 78% [45,94] | 56% [27,81] | 89% [56,98] | 56% [27,81] | 89% [56,98] |
| `atomic_scientific_fact` | 4 | 25% [5,70] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N | 100% [51,100] ⚠️LOW_N | 100% [51,100] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N |
| `chinese_remainder_theorem` | 5 | 20% [4,62] | 20% [4,62] | 40% [12,77] | 20% [4,62] | 0% [0,43] | 0% [0,43] |
| `combinatorics_combinations` | 5 | 60% [23,88] | 80% [38,96] | 100% [57,100] | 60% [23,88] | 40% [12,77] | 0% [0,43] |
| `combinatorics_derangements` | 4 | 0% [0,49] ⚠️LOW_N | 75% [30,95] ⚠️LOW_N | 75% [30,95] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N |
| `diophantine_factorization` | 5 | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
| `discrete_recurrences` | 5 | 60% [23,88] | 80% [38,96] | 100% [57,100] | 60% [23,88] | 0% [0,43] | 40% [12,77] |
| `elementary_algebra` | 5 | 100% [57,100] | 100% [57,100] | 100% [57,100] | 100% [57,100] | 60% [23,88] | 100% [57,100] |
| `euler_totient` | 4 | 25% [5,70] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N |
| `fermat_modular_orders` | 4 | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N |
| `geometric_inradius` | 5 | 100% [57,100] | 80% [38,96] | 100% [57,100] | 100% [57,100] | 100% [57,100] | 0% [0,43] |
| `harmonic_work_rates` | 5 | 80% [38,96] | 100% [57,100] | 0% [0,43] | 80% [38,96] | 0% [0,43] | 0% [0,43] |
| `legendre_prime_valuation` | 5 | 0% [0,43] | 40% [12,77] | 80% [38,96] | 0% [0,43] | 40% [12,77] | 0% [0,43] |
| `modular_arithmetic` | 5 | 80% [38,96] | 60% [23,88] | 40% [12,77] | 80% [38,96] | 0% [0,43] | 80% [38,96] |
| `physics_kinematics` | 5 | 40% [12,77] | 20% [4,62] | 100% [57,100] | 40% [12,77] | 0% [0,43] | 20% [4,62] |
| `polynomial_invariants` | 5 | 40% [12,77] | 80% [38,96] | 60% [23,88] | 40% [12,77] | 40% [12,77] | 0% [0,43] |
| `quadratic_algebra` | 5 | 0% [0,43] | 100% [57,100] | 60% [23,88] | 0% [0,43] | 60% [23,88] | 100% [57,100] |
| `sequences_progressions` | 5 | 100% [57,100] | 100% [57,100] | 80% [38,96] | 100% [57,100] | 40% [12,77] | 60% [23,88] |
| `triangle_geometry` | 5 | 80% [38,96] | 100% [57,100] | 40% [12,77] | 80% [38,96] | 80% [38,96] | 0% [0,43] |

### Budget = 1024 tokens

| Category | N | A: Acc [CI] | B: Acc [CI] | C: Acc [CI] | D: Acc [CI] | E: Acc [CI] | F: Acc [CI] |
|----------|---|---|---|---|---|---|---|
| `arithmetic_word_problem` | 9 | 89% [56,98] | 78% [45,94] | 56% [27,81] | 89% [56,98] | 56% [27,81] | 89% [56,98] |
| `atomic_scientific_fact` | 4 | 25% [5,70] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N | 100% [51,100] ⚠️LOW_N | 100% [51,100] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N |
| `chinese_remainder_theorem` | 5 | 20% [4,62] | 60% [23,88] | 40% [12,77] | 20% [4,62] | 0% [0,43] | 0% [0,43] |
| `combinatorics_combinations` | 5 | 80% [38,96] | 80% [38,96] | 100% [57,100] | 80% [38,96] | 40% [12,77] | 80% [38,96] |
| `combinatorics_derangements` | 4 | 100% [51,100] ⚠️LOW_N | 100% [51,100] ⚠️LOW_N | 75% [30,95] ⚠️LOW_N | 100% [51,100] ⚠️LOW_N | 75% [30,95] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N |
| `diophantine_factorization` | 5 | 0% [0,43] | 20% [4,62] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
| `discrete_recurrences` | 5 | 60% [23,88] | 80% [38,96] | 100% [57,100] | 60% [23,88] | 0% [0,43] | 100% [57,100] |
| `elementary_algebra` | 5 | 100% [57,100] | 100% [57,100] | 100% [57,100] | 100% [57,100] | 60% [23,88] | 100% [57,100] |
| `euler_totient` | 4 | 25% [5,70] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N |
| `fermat_modular_orders` | 4 | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N |
| `geometric_inradius` | 5 | 100% [57,100] | 100% [57,100] | 100% [57,100] | 100% [57,100] | 100% [57,100] | 40% [12,77] |
| `harmonic_work_rates` | 5 | 100% [57,100] | 100% [57,100] | 0% [0,43] | 100% [57,100] | 0% [0,43] | 100% [57,100] |
| `legendre_prime_valuation` | 5 | 80% [38,96] | 100% [57,100] | 80% [38,96] | 80% [38,96] | 60% [23,88] | 0% [0,43] |
| `modular_arithmetic` | 5 | 80% [38,96] | 100% [57,100] | 40% [12,77] | 80% [38,96] | 0% [0,43] | 100% [57,100] |
| `physics_kinematics` | 5 | 40% [12,77] | 20% [4,62] | 100% [57,100] | 40% [12,77] | 0% [0,43] | 40% [12,77] |
| `polynomial_invariants` | 5 | 40% [12,77] | 100% [57,100] | 60% [23,88] | 40% [12,77] | 40% [12,77] | 20% [4,62] |
| `quadratic_algebra` | 5 | 0% [0,43] | 100% [57,100] | 60% [23,88] | 0% [0,43] | 60% [23,88] | 100% [57,100] |
| `sequences_progressions` | 5 | 100% [57,100] | 100% [57,100] | 80% [38,96] | 100% [57,100] | 40% [12,77] | 60% [23,88] |
| `triangle_geometry` | 5 | 100% [57,100] | 100% [57,100] | 40% [12,77] | 100% [57,100] | 100% [57,100] | 0% [0,43] |

### Budget = 2048 tokens

| Category | N | A: Acc [CI] | B: Acc [CI] | C: Acc [CI] | D: Acc [CI] | E: Acc [CI] | F: Acc [CI] |
|----------|---|---|---|---|---|---|---|
| `arithmetic_word_problem` | 9 | 89% [56,98] | 78% [45,94] | 56% [27,81] | 89% [56,98] | 56% [27,81] | 89% [56,98] |
| `atomic_scientific_fact` | 4 | 25% [5,70] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N | 100% [51,100] ⚠️LOW_N | 100% [51,100] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N |
| `chinese_remainder_theorem` | 5 | 20% [4,62] | 60% [23,88] | 40% [12,77] | 20% [4,62] | 0% [0,43] | 40% [12,77] |
| `combinatorics_combinations` | 5 | 80% [38,96] | 80% [38,96] | 100% [57,100] | 80% [38,96] | 40% [12,77] | 80% [38,96] |
| `combinatorics_derangements` | 4 | 100% [51,100] ⚠️LOW_N | 100% [51,100] ⚠️LOW_N | 75% [30,95] ⚠️LOW_N | 100% [51,100] ⚠️LOW_N | 75% [30,95] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N |
| `diophantine_factorization` | 5 | 0% [0,43] | 20% [4,62] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
| `discrete_recurrences` | 5 | 60% [23,88] | 80% [38,96] | 100% [57,100] | 60% [23,88] | 0% [0,43] | 100% [57,100] |
| `elementary_algebra` | 5 | 100% [57,100] | 100% [57,100] | 100% [57,100] | 100% [57,100] | 60% [23,88] | 100% [57,100] |
| `euler_totient` | 4 | 25% [5,70] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N |
| `fermat_modular_orders` | 4 | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 0% [0,49] ⚠️LOW_N | 25% [5,70] ⚠️LOW_N | 50% [15,85] ⚠️LOW_N |
| `geometric_inradius` | 5 | 100% [57,100] | 100% [57,100] | 100% [57,100] | 100% [57,100] | 100% [57,100] | 100% [57,100] |
| `harmonic_work_rates` | 5 | 100% [57,100] | 100% [57,100] | 0% [0,43] | 100% [57,100] | 0% [0,43] | 100% [57,100] |
| `legendre_prime_valuation` | 5 | 80% [38,96] | 100% [57,100] | 80% [38,96] | 80% [38,96] | 60% [23,88] | 80% [38,96] |
| `modular_arithmetic` | 5 | 80% [38,96] | 100% [57,100] | 40% [12,77] | 80% [38,96] | 0% [0,43] | 100% [57,100] |
| `physics_kinematics` | 5 | 40% [12,77] | 20% [4,62] | 100% [57,100] | 40% [12,77] | 0% [0,43] | 80% [38,96] |
| `polynomial_invariants` | 5 | 40% [12,77] | 100% [57,100] | 60% [23,88] | 40% [12,77] | 40% [12,77] | 80% [38,96] |
| `quadratic_algebra` | 5 | 0% [0,43] | 100% [57,100] | 60% [23,88] | 0% [0,43] | 60% [23,88] | 100% [57,100] |
| `sequences_progressions` | 5 | 100% [57,100] | 100% [57,100] | 80% [38,96] | 100% [57,100] | 40% [12,77] | 100% [57,100] |
| `triangle_geometry` | 5 | 100% [57,100] | 100% [57,100] | 40% [12,77] | 100% [57,100] | 100% [57,100] | 80% [38,96] |

## 4. Contamination Breakdown

| Arm | Budget | All Acc% | Clean-Only Acc% | Gap | Alert |
|-----|--------|----------|-----------------|-----|-------|
| A | 256 | 15.8% | 15.8% | 0.0pp | OK |
| A | 512 | 50.5% | 50.5% | 0.0pp | OK |
| A | 1024 | 62.1% | 62.1% | 0.0pp | OK |
| A | 2048 | 62.1% | 62.1% | 0.0pp | OK |
| B | 256 | 29.5% | 29.5% | 0.0pp | OK |
| B | 512 | 64.2% | 64.2% | 0.0pp | OK |
| B | 1024 | 75.8% | 75.8% | 0.0pp | OK |
| B | 2048 | 75.8% | 75.8% | 0.0pp | OK |
| C | 256 | 62.1% | 62.1% | 0.0pp | OK |
| C | 512 | 62.1% | 62.1% | 0.0pp | OK |
| C | 1024 | 62.1% | 62.1% | 0.0pp | OK |
| C | 2048 | 62.1% | 62.1% | 0.0pp | OK |
| D | 256 | 18.9% | 18.9% | 0.0pp | OK |
| D | 512 | 53.7% | 53.7% | 0.0pp | OK |
| D | 1024 | 65.3% | 65.3% | 0.0pp | OK |
| D | 2048 | 65.3% | 65.3% | 0.0pp | OK |
| E | 256 | 20.0% | 20.0% | 0.0pp | OK |
| E | 512 | 34.7% | 34.7% | 0.0pp | OK |
| E | 1024 | 39.0% | 39.0% | 0.0pp | OK |
| E | 2048 | 39.0% | 39.0% | 0.0pp | OK |
| F | 256 | 2.1% | 2.1% | 0.0pp | OK |
| F | 512 | 31.6% | 31.6% | 0.0pp | OK |
| F | 1024 | 49.5% | 49.5% | 0.0pp | OK |
| F | 2048 | 75.8% | 75.8% | 0.0pp | OK |

## 5. Pre-Registered Decision Rules

**Rule (from PREREG_VALUATION_V2.md):** Arm C adds format value over Arm D ONLY IF the paired bootstrap 95% CI on (J/correct_D − J/correct_C) strictly excludes zero *and* strict Acc(C) ≥ strict Acc(D) − 2.0 pp.  Both accuracies use declared answers; lenient extraction is not allowed in this decision.

| Budget | Acc_C | Acc_D | Acc_F | CI excludes zero (C vs D) | Acc condition | Verdict |
|--------|-------|-------|-------|--------------------------|---------------|---------|
| 256 | 62.1% | 18.9% | 2.1% | ✓ | ✓ | ✓ FORMAT VALUE SUPPORTED (strict) |
| 512 | 62.1% | 53.7% | 31.6% | ✓ | ✓ | ✓ FORMAT VALUE SUPPORTED (strict) |
| 1024 | 62.1% | 65.3% | 49.5% | ✓ | ✗ | ✗ NO FORMAT VALUE (strict) |
| 2048 | 62.1% | 65.3% | 75.8% | ✓ | ✗ | ✗ NO FORMAT VALUE (strict) |

## 6. Reproducibility

Full manifest: `EVAL_RUN_MANIFEST.json`  
Raw JSONL: `results/robust_eval/robust_eval_results.jsonl`  
Dataset SHA-256: `F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707`  
Commit: `17d62d1bc8e26f6a02bee67e777ebf8984a0ca02`  
Seed: `42`  
Adapter SHA-256: `204823BC216BBF7329D6C7C5159ACF8F6688FD321AD1652757C2A9A9D066B31C`  
Expected cells: `2400`  
Energy measurement: `nvml`  
SymPy: `1.14.0`  
