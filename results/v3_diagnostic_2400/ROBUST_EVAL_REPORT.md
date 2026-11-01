# SymboLM Robust Evaluation Report

**Run ID:** `robust_eval_20261031T151842Z`  
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
**Duplicate-prompt rows:** 432/2400  
**Timestamp (UTC):** 2026-10-31T15:18:53.930968+00:00  
**GPU:** Tesla T4  
**Arms evaluated:** A, B, C, D, E, F  
**Budgets:** [256, 512, 1024, 2048]  

> **Diagnostic only.** This is a deterministic subset run, not the 200-problem x 4-budget x 6-arm preregistered experiment. It cannot be reported as the v3 benchmark result.


## 1. Correctness–Token–Energy Pareto Frontier

Format: `Acc% [CI-lo, CI-hi] | tok/correct | J/correct | mean-tok | trunc`

| Budget | Arm A: Ordinary CoT | Arm B: Answer-Only | Arm C: Full Registers | Arm D: Router-Only Control | Arm E: CoT + Python Sandbox | Arm F: Vanilla DeepSeek (No Adapter) |
|--------|---|---|---|---|---|---|
| **256** | 16.0% [10.1,24.4] \| 1413.6 tok/✓ \| 3710.3 J/✓ \| 226tok \| 67trunc | 29.0% [21.0,38.5] \| 739.6 tok/✓ \| 1883.2 J/✓ \| 214tok \| 56trunc | 62.0% [52.2,70.9] \| 179.5 tok/✓ \| 501.0 J/✓ \| 111tok \| 0trunc | 19.0% [12.5,27.8] \| 1163.3 tok/✓ \| 2925.3 J/✓ \| 221tok \| 67trunc | 20.0% [13.3,28.9] \| 935.7 tok/✓ \| 2412.7 J/✓ \| 187tok \| 37trunc | 2.0% [0.5,7.0] \| 12762.5 tok/✓ \| 27034.7 J/✓ \| 255tok \| 98trunc |
| **512** | 51.0% [41.3,60.6] \| 666.3 tok/✓ \| 1546.8 J/✓ \| 340tok \| 21trunc | 64.0% [54.2,72.7] \| 489.3 tok/✓ \| 1167.6 J/✓ \| 313tok \| 18trunc | 62.0% [52.2,70.9] \| 179.5 tok/✓ \| 424.5 J/✓ \| 111tok \| 0trunc | 54.0% [44.3,63.4] \| 619.7 tok/✓ \| 1479.9 J/✓ \| 335tok \| 21trunc | 34.0% [25.5,43.7] \| 681.5 tok/✓ \| 1610.1 J/✓ \| 232tok \| 9trunc | 31.0% [22.8,40.6] \| 1504.0 tok/✓ \| 2944.3 J/✓ \| 466tok \| 67trunc |
| **1024** | 64.0% [54.2,72.7] \| 588.7 tok/✓ \| 1420.6 J/✓ \| 377tok \| 0trunc | 77.0% [67.8,84.2] \| 457.5 tok/✓ \| 1121.2 J/✓ \| 352tok \| 2trunc | 62.0% [52.2,70.9] \| 179.5 tok/✓ \| 424.9 J/✓ \| 111tok \| 0trunc | 67.0% [57.3,75.4] \| 554.7 tok/✓ \| 1350.9 J/✓ \| 372tok \| 0trunc | 39.0% [30.0,48.8] \| 648.9 tok/✓ \| 1575.8 J/✓ \| 253tok \| 3trunc | 48.0% [38.5,57.7] \| 1594.0 tok/✓ \| 3157.2 J/✓ \| 765tok \| 48trunc |
| **2048** | 64.0% [54.2,72.7] \| 588.7 tok/✓ \| 1434.2 J/✓ \| 377tok \| 0trunc | 77.0% [67.8,84.2] \| 458.4 tok/✓ \| 1122.6 J/✓ \| 353tok \| 0trunc | 62.0% [52.2,70.9] \| 179.5 tok/✓ \| 431.0 J/✓ \| 111tok \| 0trunc | 67.0% [57.3,75.4] \| 554.7 tok/✓ \| 1360.5 J/✓ \| 372tok \| 0trunc | 39.0% [30.0,48.8] \| 727.7 tok/✓ \| 1883.6 J/✓ \| 284tok \| 3trunc | 74.0% [64.6,81.6] \| 1553.2 tok/✓ \| 3077.7 J/✓ \| 1149tok \| 22trunc |

### 1.0 Strict Score Diagnostics

Strict accuracy is the primary metric. Lenient accuracy is the old fallback-based number and is shown only to expose the gap. EOS and format rates are computed only where the run recorded them.

| Budget | Arm | Strict Acc | Lenient Acc | EOS rate | Valid format | Declared answer | Truncated |
|--------|-----|-----------|-------------|----------|--------------|-----------------|-----------|
| 256 | A | 16.0% | 24.0% | 33.0% | 20.0% | 20.0% | 67 |
| 256 | B | 29.0% | 30.0% | 44.0% | 35.0% | 35.0% | 56 |
| 256 | C | 62.0% | 62.0% | 100.0% | 100.0% | 100.0% | 0 |
| 256 | D | 19.0% | 27.0% | 33.0% | 23.0% | 23.0% | 67 |
| 256 | E | 20.0% | 36.0% | 63.0% | 29.0% | 29.0% | 37 |
| 256 | F | 2.0% | 2.0% | 2.0% | 2.0% | 2.0% | 98 |
| 512 | A | 51.0% | 58.0% | 79.0% | 61.0% | 61.0% | 21 |
| 512 | B | 64.0% | 65.0% | 82.0% | 73.0% | 73.0% | 18 |
| 512 | C | 62.0% | 62.0% | 100.0% | 100.0% | 100.0% | 0 |
| 512 | D | 54.0% | 61.0% | 79.0% | 64.0% | 64.0% | 21 |
| 512 | E | 34.0% | 48.0% | 91.0% | 48.0% | 48.0% | 9 |
| 512 | F | 31.0% | 31.0% | 33.0% | 32.0% | 32.0% | 67 |
| 1024 | A | 64.0% | 71.0% | 100.0% | 81.0% | 81.0% | 0 |
| 1024 | B | 77.0% | 77.0% | 98.0% | 89.0% | 89.0% | 2 |
| 1024 | C | 62.0% | 62.0% | 100.0% | 100.0% | 100.0% | 0 |
| 1024 | D | 67.0% | 74.0% | 100.0% | 84.0% | 84.0% | 0 |
| 1024 | E | 39.0% | 53.0% | 97.0% | 53.0% | 53.0% | 3 |
| 1024 | F | 48.0% | 49.0% | 52.0% | 49.0% | 49.0% | 48 |
| 2048 | A | 64.0% | 71.0% | 100.0% | 81.0% | 81.0% | 0 |
| 2048 | B | 77.0% | 77.0% | 100.0% | 91.0% | 91.0% | 0 |
| 2048 | C | 62.0% | 62.0% | 100.0% | 100.0% | 100.0% | 0 |
| 2048 | D | 67.0% | 74.0% | 100.0% | 84.0% | 84.0% | 0 |
| 2048 | E | 39.0% | 53.0% | 97.0% | 53.0% | 53.0% | 3 |
| 2048 | F | 74.0% | 82.0% | 78.0% | 75.0% | 75.0% | 22 |

### 1.1 Test-Time Compute & Overthinking Metrics (OptimalThinkingBench & LLMTHINKBENCH)

Overthinking-Adjusted Accuracy (OAA), Area Under Curve (AUC_OAA), and Harmonic Overthinking Score (O).
Formulation: O = 2 * (Acc * TokenEff) / (Acc + TokenEff). Penalizes runaway token bloat and rewards edge efficiency.

| Arm | Budget | Pass@1 | Mean Tok | Tok/correct | Correct/1k tok | OAA@256 | AUC_OAA | TokenEff | Overthinking Score (O) |
|-----|--------|--------|----------|-------------|----------------|---------|---------|----------|------------------------|
| Arm A: Ordinary CoT | 256 | 16.0% | 226.2tok | 1413.6 | 0.707 | 16.0% | 0.145 | 1.00 | **0.276** |
| Arm A: Ordinary CoT | 512 | 51.0% | 339.8tok | 666.3 | 1.501 | 16.0% | 0.430 | 0.75 | **0.608** |
| Arm A: Ordinary CoT | 1024 | 64.0% | 376.8tok | 588.7 | 1.699 | 16.0% | 0.511 | 0.68 | **0.659** |
| Arm A: Ordinary CoT | 2048 | 64.0% | 376.8tok | 588.7 | 1.699 | 16.0% | 0.511 | 0.68 | **0.659** |
| Arm B: Answer-Only | 256 | 29.0% | 214.5tok | 739.6 | 1.352 | 29.0% | 0.270 | 1.00 | **0.450** |
| Arm B: Answer-Only | 512 | 64.0% | 313.1tok | 489.3 | 2.044 | 29.0% | 0.554 | 0.82 | **0.718** |
| Arm B: Answer-Only | 1024 | 77.0% | 352.2tok | 457.5 | 2.186 | 29.0% | 0.635 | 0.73 | **0.748** |
| Arm B: Answer-Only | 2048 | 77.0% | 353.0tok | 458.4 | 2.181 | 29.0% | 0.635 | 0.73 | **0.747** |
| Arm C: Full Registers | 256 | 62.0% | 111.3tok | 179.5 | 5.571 | 62.0% | 0.588 | 1.00 | **0.765** |
| Arm C: Full Registers | 512 | 62.0% | 111.3tok | 179.5 | 5.571 | 62.0% | 0.588 | 1.00 | **0.765** |
| Arm C: Full Registers | 1024 | 62.0% | 111.3tok | 179.5 | 5.571 | 62.0% | 0.588 | 1.00 | **0.765** |
| Arm C: Full Registers | 2048 | 62.0% | 111.3tok | 179.5 | 5.571 | 62.0% | 0.588 | 1.00 | **0.765** |
| Arm D: Router-Only Control | 256 | 19.0% | 221.0tok | 1163.3 | 0.860 | 19.0% | 0.177 | 1.00 | **0.319** |
| Arm D: Router-Only Control | 512 | 54.0% | 334.7tok | 619.7 | 1.614 | 19.0% | 0.461 | 0.77 | **0.633** |
| Arm D: Router-Only Control | 1024 | 67.0% | 371.6tok | 554.7 | 1.803 | 19.0% | 0.542 | 0.69 | **0.679** |
| Arm D: Router-Only Control | 2048 | 67.0% | 371.6tok | 554.7 | 1.803 | 19.0% | 0.542 | 0.69 | **0.679** |
| Arm E: CoT + Python Sandbox | 256 | 20.0% | 187.1tok | 935.7 | 1.069 | 20.0% | 0.183 | 1.00 | **0.333** |
| Arm E: CoT + Python Sandbox | 512 | 34.0% | 231.7tok | 681.5 | 1.467 | 20.0% | 0.297 | 1.00 | **0.507** |
| Arm E: CoT + Python Sandbox | 1024 | 39.0% | 253.1tok | 648.9 | 1.541 | 20.0% | 0.328 | 1.00 | **0.561** |
| Arm E: CoT + Python Sandbox | 2048 | 39.0% | 283.8tok | 727.7 | 1.374 | 20.0% | 0.328 | 0.90 | **0.545** |
| Arm F: Vanilla DeepSeek (No Adapter) | 256 | 2.0% | 255.2tok | 12762.5 | 0.078 | 2.0% | 0.018 | 1.00 | **0.039** |
| Arm F: Vanilla DeepSeek (No Adapter) | 512 | 31.0% | 466.2tok | 1504.0 | 0.665 | 2.0% | 0.254 | 0.55 | **0.396** |
| Arm F: Vanilla DeepSeek (No Adapter) | 1024 | 48.0% | 765.1tok | 1594.0 | 0.627 | 2.0% | 0.360 | 0.33 | **0.394** |
| Arm F: Vanilla DeepSeek (No Adapter) | 2048 | 74.0% | 1149.4tok | 1553.2 | 0.644 | 2.0% | 0.425 | 0.22 | **0.342** |

### SymPy Dual-Layer Bonus

Additional answers rescued by Layer 2 symbolic equality (would be FAIL in string-only mode):

| Arm | Budget | SymPy-rescued correct answers |
|-----|--------|-------------------------------|

## 2. Effect Sizes (Cohen's d + CLES)

Cohen's d on J/correct per arm-pair. CLES = P(C cheaper than comparison arm).
Positive d = arm C has lower J/correct (better). |d| > 0.8 = large effect.

| Budget | Comparison | Cohen's d (C vs X) | CLES | Bootstrap 95% CI (J/correct_X - J/correct_C) | Excludes Zero |
|--------|-----------|-------------------|------|---------------------------------------------|---------------|
| 256 | Arm A: Ordinary CoT | -1.5988 | 0.8881 | [2000.27, 6164.78] | ✓ |
| 256 | Arm B: Answer-Only | -0.7039 | 0.6574 | [902.26, 2253.05] | ✓ |
| 256 | Arm D: Router-Only Control | -0.6756 | 0.6409 | [1520.45, 4452.67] | ✓ |
| 256 | Arm E: CoT + Python Sandbox | -0.9128 | 0.7177 | [1199.65, 3366.10] | ✓ |
| 256 | Arm F: Vanilla DeepSeek (No Adapter) | -2.0645 | 0.9032 | [10161.95, 55091.73] | ✓ |
| 512 | Arm A: Ordinary CoT | -2.5361 | 0.9728 | [845.15, 1508.35] | ✓ |
| 512 | Arm B: Answer-Only | -1.6422 | 0.8558 | [566.09, 978.44] | ✓ |
| 512 | Arm D: Router-Only Control | -1.9312 | 0.8895 | [791.71, 1417.83] | ✓ |
| 512 | Arm E: CoT + Python Sandbox | -1.7671 | 0.871 | [829.45, 1769.86] | ✓ |
| 512 | Arm F: Vanilla DeepSeek (No Adapter) | -4.4338 | 0.9979 | [1796.07, 3789.94] | ✓ |
| 1024 | Arm A: Ordinary CoT | -2.1144 | 0.9786 | [782.78, 1265.30] | ✓ |
| 1024 | Arm B: Answer-Only | -1.5587 | 0.8798 | [542.61, 886.86] | ✓ |
| 1024 | Arm D: Router-Only Control | -1.8209 | 0.9109 | [724.57, 1174.06] | ✓ |
| 1024 | Arm E: CoT + Python Sandbox | -1.5545 | 0.8867 | [809.29, 1681.29] | ✓ |
| 1024 | Arm F: Vanilla DeepSeek (No Adapter) | -2.4457 | 0.9983 | [2041.17, 3736.67] | ✓ |
| 2048 | Arm A: Ordinary CoT | -2.1417 | 0.9723 | [791.28, 1272.98] | ✓ |
| 2048 | Arm B: Answer-Only | -1.5249 | 0.8735 | [532.05, 888.38] | ✓ |
| 2048 | Arm D: Router-Only Control | -1.7477 | 0.9056 | [723.15, 1184.70] | ✓ |
| 2048 | Arm E: CoT + Python Sandbox | -1.3898 | 0.8772 | [879.75, 2360.68] | ✓ |
| 2048 | Arm F: Vanilla DeepSeek (No Adapter) | -1.7708 | 0.9943 | [2093.97, 3327.46] | ✓ |

## 3. Per-Category Accuracy with Wilson 95% CIs

### Budget = 256 tokens

| Category | N | A: Acc [CI] | B: Acc [CI] | C: Acc [CI] | D: Acc [CI] | E: Acc [CI] | F: Acc [CI] |
|----------|---|---|---|---|---|---|---|
| `arithmetic_word_problem` | 10 | 10% [2,40] | 30% [11,60] | 50% [24,76] | 10% [2,40] | 40% [17,69] | 0% [0,28] |
| `atomic_scientific_fact` | 5 | 40% [12,77] | 40% [12,77] | 100% [57,100] | 100% [57,100] | 60% [23,88] | 20% [4,62] |
| `chinese_remainder_theorem` | 5 | 20% [4,62] | 20% [4,62] | 40% [12,77] | 20% [4,62] | 0% [0,43] | 0% [0,43] |
| `combinatorics_combinations` | 5 | 0% [0,43] | 20% [4,62] | 100% [57,100] | 0% [0,43] | 20% [4,62] | 0% [0,43] |
| `combinatorics_derangements` | 5 | 0% [0,43] | 0% [0,43] | 60% [23,88] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
| `diophantine_factorization` | 5 | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
| `discrete_recurrences` | 5 | 60% [23,88] | 80% [38,96] | 100% [57,100] | 60% [23,88] | 0% [0,43] | 0% [0,43] |
| `elementary_algebra` | 5 | 80% [38,96] | 100% [57,100] | 100% [57,100] | 80% [38,96] | 60% [23,88] | 0% [0,43] |
| `euler_totient` | 5 | 0% [0,43] | 20% [4,62] | 60% [23,88] | 0% [0,43] | 20% [4,62] | 0% [0,43] |
| `fermat_modular_orders` | 5 | 0% [0,43] | 20% [4,62] | 20% [4,62] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
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
| `arithmetic_word_problem` | 10 | 90% [60,98] | 80% [49,94] | 50% [24,76] | 90% [60,98] | 50% [24,76] | 90% [60,98] |
| `atomic_scientific_fact` | 5 | 40% [12,77] | 60% [23,88] | 100% [57,100] | 100% [57,100] | 60% [23,88] | 40% [12,77] |
| `chinese_remainder_theorem` | 5 | 20% [4,62] | 20% [4,62] | 40% [12,77] | 20% [4,62] | 0% [0,43] | 0% [0,43] |
| `combinatorics_combinations` | 5 | 60% [23,88] | 80% [38,96] | 100% [57,100] | 60% [23,88] | 40% [12,77] | 0% [0,43] |
| `combinatorics_derangements` | 5 | 0% [0,43] | 60% [23,88] | 60% [23,88] | 0% [0,43] | 20% [4,62] | 0% [0,43] |
| `diophantine_factorization` | 5 | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
| `discrete_recurrences` | 5 | 60% [23,88] | 80% [38,96] | 100% [57,100] | 60% [23,88] | 0% [0,43] | 40% [12,77] |
| `elementary_algebra` | 5 | 100% [57,100] | 100% [57,100] | 100% [57,100] | 100% [57,100] | 60% [23,88] | 100% [57,100] |
| `euler_totient` | 5 | 20% [4,62] | 20% [4,62] | 60% [23,88] | 20% [4,62] | 20% [4,62] | 0% [0,43] |
| `fermat_modular_orders` | 5 | 20% [4,62] | 20% [4,62] | 20% [4,62] | 20% [4,62] | 20% [4,62] | 0% [0,43] |
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
| `arithmetic_word_problem` | 10 | 90% [60,98] | 80% [49,94] | 50% [24,76] | 90% [60,98] | 50% [24,76] | 90% [60,98] |
| `atomic_scientific_fact` | 5 | 40% [12,77] | 60% [23,88] | 100% [57,100] | 100% [57,100] | 60% [23,88] | 40% [12,77] |
| `chinese_remainder_theorem` | 5 | 20% [4,62] | 60% [23,88] | 40% [12,77] | 20% [4,62] | 0% [0,43] | 0% [0,43] |
| `combinatorics_combinations` | 5 | 80% [38,96] | 80% [38,96] | 100% [57,100] | 80% [38,96] | 40% [12,77] | 80% [38,96] |
| `combinatorics_derangements` | 5 | 100% [57,100] | 100% [57,100] | 60% [23,88] | 100% [57,100] | 80% [38,96] | 0% [0,43] |
| `diophantine_factorization` | 5 | 0% [0,43] | 20% [4,62] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
| `discrete_recurrences` | 5 | 60% [23,88] | 80% [38,96] | 100% [57,100] | 60% [23,88] | 0% [0,43] | 100% [57,100] |
| `elementary_algebra` | 5 | 100% [57,100] | 100% [57,100] | 100% [57,100] | 100% [57,100] | 60% [23,88] | 100% [57,100] |
| `euler_totient` | 5 | 40% [12,77] | 40% [12,77] | 60% [23,88] | 40% [12,77] | 20% [4,62] | 0% [0,43] |
| `fermat_modular_orders` | 5 | 20% [4,62] | 20% [4,62] | 20% [4,62] | 20% [4,62] | 20% [4,62] | 0% [0,43] |
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
| `arithmetic_word_problem` | 10 | 90% [60,98] | 80% [49,94] | 50% [24,76] | 90% [60,98] | 50% [24,76] | 90% [60,98] |
| `atomic_scientific_fact` | 5 | 40% [12,77] | 60% [23,88] | 100% [57,100] | 100% [57,100] | 60% [23,88] | 40% [12,77] |
| `chinese_remainder_theorem` | 5 | 20% [4,62] | 60% [23,88] | 40% [12,77] | 20% [4,62] | 0% [0,43] | 40% [12,77] |
| `combinatorics_combinations` | 5 | 80% [38,96] | 80% [38,96] | 100% [57,100] | 80% [38,96] | 40% [12,77] | 80% [38,96] |
| `combinatorics_derangements` | 5 | 100% [57,100] | 100% [57,100] | 60% [23,88] | 100% [57,100] | 80% [38,96] | 40% [12,77] |
| `diophantine_factorization` | 5 | 0% [0,43] | 20% [4,62] | 0% [0,43] | 0% [0,43] | 0% [0,43] | 0% [0,43] |
| `discrete_recurrences` | 5 | 60% [23,88] | 80% [38,96] | 100% [57,100] | 60% [23,88] | 0% [0,43] | 100% [57,100] |
| `elementary_algebra` | 5 | 100% [57,100] | 100% [57,100] | 100% [57,100] | 100% [57,100] | 60% [23,88] | 100% [57,100] |
| `euler_totient` | 5 | 40% [12,77] | 40% [12,77] | 60% [23,88] | 40% [12,77] | 20% [4,62] | 20% [4,62] |
| `fermat_modular_orders` | 5 | 20% [4,62] | 20% [4,62] | 20% [4,62] | 20% [4,62] | 20% [4,62] | 60% [23,88] |
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
| A | 256 | 16.0% | 16.0% | 0.0pp | OK |
| A | 512 | 51.0% | 51.0% | 0.0pp | OK |
| A | 1024 | 64.0% | 64.0% | 0.0pp | OK |
| A | 2048 | 64.0% | 64.0% | 0.0pp | OK |
| B | 256 | 29.0% | 29.0% | 0.0pp | OK |
| B | 512 | 64.0% | 64.0% | 0.0pp | OK |
| B | 1024 | 77.0% | 77.0% | 0.0pp | OK |
| B | 2048 | 77.0% | 77.0% | 0.0pp | OK |
| C | 256 | 62.0% | 62.0% | 0.0pp | OK |
| C | 512 | 62.0% | 62.0% | 0.0pp | OK |
| C | 1024 | 62.0% | 62.0% | 0.0pp | OK |
| C | 2048 | 62.0% | 62.0% | 0.0pp | OK |
| D | 256 | 19.0% | 19.0% | 0.0pp | OK |
| D | 512 | 54.0% | 54.0% | 0.0pp | OK |
| D | 1024 | 67.0% | 67.0% | 0.0pp | OK |
| D | 2048 | 67.0% | 67.0% | 0.0pp | OK |
| E | 256 | 20.0% | 20.0% | 0.0pp | OK |
| E | 512 | 34.0% | 34.0% | 0.0pp | OK |
| E | 1024 | 39.0% | 39.0% | 0.0pp | OK |
| E | 2048 | 39.0% | 39.0% | 0.0pp | OK |
| F | 256 | 2.0% | 2.0% | 0.0pp | OK |
| F | 512 | 31.0% | 31.0% | 0.0pp | OK |
| F | 1024 | 48.0% | 48.0% | 0.0pp | OK |
| F | 2048 | 74.0% | 74.0% | 0.0pp | OK |

## 5. Pre-Registered Decision Rules

**Rule (from PREREG_VALUATION_V2.md):** Arm C adds format value over Arm D ONLY IF the paired bootstrap 95% CI on (J/correct_D − J/correct_C) strictly excludes zero *and* strict Acc(C) ≥ strict Acc(D) − 2.0 pp.  Both accuracies use declared answers; lenient extraction is not allowed in this decision.

| Budget | Acc_C | Acc_D | Acc_F | CI excludes zero (C vs D) | Acc condition | Verdict |
|--------|-------|-------|-------|--------------------------|---------------|---------|
| 256 | 62.0% | 19.0% | 2.0% | ✓ | ✓ | ✓ FORMAT VALUE SUPPORTED (strict) |
| 512 | 62.0% | 54.0% | 31.0% | ✓ | ✓ | ✓ FORMAT VALUE SUPPORTED (strict) |
| 1024 | 62.0% | 67.0% | 48.0% | ✓ | ✗ | ✗ NO FORMAT VALUE (strict) |
| 2048 | 62.0% | 67.0% | 74.0% | ✓ | ✗ | ✗ NO FORMAT VALUE (strict) |

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
