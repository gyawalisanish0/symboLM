# SymboLM Robust Evaluation Report

**Run ID:** `robust_eval_20261019T070229Z`  
**Commit:** `daaced2b4f09bbfffb778df7b3467ad87fd33b5b`  
**Dataset SHA-256 verified:** `False`  
**Timestamp (UTC):** 2026-10-19T07:02:29.046123+00:00  
**GPU:** Tesla T4  
**Arms evaluated:** A, B, C, D, E, F  
**Budgets:** [256, 512, 1024, 2048]  

## 1. Accuracy–Energy Pareto Frontier

Format: `Acc% [CI-lo, CI-hi] | J/correct | mean-tok | trunc`

| Budget | Arm A: Ordinary CoT | Arm B: Answer-Only | Arm C: Full Registers | Arm D: Router-Only Control | Arm E: CoT + Python Sandbox | Arm F: Vanilla DeepSeek (No Adapter) |
|--------|---|---|---|---|---|---|
| **256** | 8.6% [3.7,18.6] \| 4308.9 J/✓ \| 250tok \| 53trunc | 39.7% [28.1,52.5] \| 822.6 J/✓ \| 218tok \| 38trunc | 37.0% [30.6,43.9] \| 547.9 J/✓ \| 136tok \| 27trunc | 8.6% [3.7,18.6] \| 4371.6 J/✓ \| 253tok \| 54trunc | 17.2% [9.6,28.9] \| 2124.2 J/✓ \| 247tok \| 51trunc | 0.0% [0.0,6.3] \| inf J/✓ \| 256tok \| 57trunc |
| **512** | 8.6% [3.7,18.6] \| 8069.7 J/✓ \| 484tok \| 53trunc | 39.7% [28.1,52.5] \| 1396.6 J/✓ \| 382tok \| 37trunc | 37.0% [30.6,43.9] \| 652.8 J/✓ \| 171tok \| 27trunc | 8.6% [3.7,18.6] \| 8205.9 J/✓ \| 491tok \| 54trunc | 17.2% [9.6,28.9] \| 3904.6 J/✓ \| 468tok \| 50trunc | 12.3% [6.1,23.2] \| 5713.6 J/✓ \| 487tok \| 46trunc |
| **1024** | 8.6% [3.7,18.6] \| 16211.6 J/✓ \| 952tok \| 53trunc | 39.7% [28.1,52.5] \| 2626.4 J/✓ \| 709tok \| 37trunc | 37.0% [30.6,43.9] \| 966.9 J/✓ \| 240tok \| 27trunc | 8.6% [3.7,18.6] \| 16410.6 J/✓ \| 968tok \| 54trunc | 17.2% [9.6,28.9] \| 7687.9 J/✓ \| 909tok \| 50trunc | 33.3% [22.5,46.3] \| 3474.0 J/✓ \| 789tok \| 30trunc |
| **2048** | 8.6% [3.7,18.6] \| 31543.8 J/✓ \| 1888tok \| 53trunc | 39.7% [28.1,52.5] \| 4949.0 J/✓ \| 1362tok \| 37trunc | 37.0% [30.6,43.9] \| 1472.3 J/✓ \| 378tok \| 27trunc | 8.6% [3.7,18.6] \| 31945.8 J/✓ \| 1921tok \| 54trunc | 17.5% [9.8,29.4] \| 14641.5 J/✓ \| 1787tok \| 49trunc | 49.1% [36.6,61.7] \| 3774.1 J/✓ \| 1286tok \| 24trunc |

### 1.1 SOTA Test-Time Compute & Overthinking Metrics (OptimalThinkingBench & LLMTHINKBENCH)

Overthinking-Adjusted Accuracy (OAA), Area Under Curve (AUC_OAA), and Harmonic Overthinking Score (O).
Formulation: O = 2 * (Acc * TokenEff) / (Acc + TokenEff). Penalizes runaway token bloat and rewards edge efficiency.

| Arm | Budget | Pass@1 | Mean Tok | OAA@256 | AUC_OAA | TokenEff | Overthinking Score (O) |
|-----|--------|--------|----------|---------|---------|----------|------------------------|
| Arm A: Ordinary CoT | 256 | 8.6% | 250.2tok | 8.6% | 0.078 | 1.00 | **0.159** |
| Arm A: Ordinary CoT | 512 | 8.6% | 484.1tok | 0.0% | 0.070 | 0.53 | **0.148** |
| Arm A: Ordinary CoT | 1024 | 8.6% | 952.0tok | 0.0% | 0.054 | 0.27 | **0.131** |
| Arm A: Ordinary CoT | 2048 | 8.6% | 1887.7tok | 0.0% | 0.022 | 0.14 | **0.105** |
| Arm B: Answer-Only | 256 | 39.7% | 217.7tok | 39.7% | 0.363 | 1.00 | **0.568** |
| Arm B: Answer-Only | 512 | 39.7% | 382.3tok | 29.3% | 0.354 | 0.67 | **0.498** |
| Arm B: Answer-Only | 1024 | 39.7% | 708.9tok | 29.3% | 0.334 | 0.36 | **0.378** |
| Arm B: Answer-Only | 2048 | 39.7% | 1362.2tok | 29.3% | 0.295 | 0.19 | **0.255** |
| Arm C: Full Registers | 256 | 37.0% | 136.3tok | 37.0% | 0.343 | 1.00 | **0.540** |
| Arm C: Full Registers | 512 | 37.0% | 170.9tok | 30.5% | 0.337 | 1.00 | **0.540** |
| Arm C: Full Registers | 1024 | 37.0% | 240.0tok | 30.5% | 0.325 | 1.00 | **0.540** |
| Arm C: Full Registers | 2048 | 37.0% | 378.3tok | 30.5% | 0.301 | 0.68 | **0.478** |
| Arm D: Router-Only Control | 256 | 8.6% | 252.6tok | 8.6% | 0.079 | 1.00 | **0.159** |
| Arm D: Router-Only Control | 512 | 8.6% | 490.9tok | 1.7% | 0.072 | 0.52 | **0.148** |
| Arm D: Router-Only Control | 1024 | 8.6% | 967.6tok | 1.7% | 0.059 | 0.26 | **0.130** |
| Arm D: Router-Only Control | 2048 | 8.6% | 1921.0tok | 1.7% | 0.034 | 0.13 | **0.105** |
| Arm E: CoT + Python Sandbox | 256 | 17.2% | 246.7tok | 17.2% | 0.156 | 1.00 | **0.294** |
| Arm E: CoT + Python Sandbox | 512 | 17.2% | 467.7tok | 1.7% | 0.142 | 0.55 | **0.262** |
| Arm E: CoT + Python Sandbox | 1024 | 17.2% | 909.1tok | 1.7% | 0.113 | 0.28 | **0.214** |
| Arm E: CoT + Python Sandbox | 2048 | 17.5% | 1787.3tok | 1.8% | 0.055 | 0.14 | **0.158** |
| Arm F: Vanilla DeepSeek (No Adapter) | 256 | 0.0% | 256.0tok | 0.0% | 0.000 | 1.00 | **0.000** |
| Arm F: Vanilla DeepSeek (No Adapter) | 512 | 12.3% | 487.3tok | 0.0% | 0.100 | 0.53 | **0.199** |
| Arm F: Vanilla DeepSeek (No Adapter) | 1024 | 33.3% | 789.2tok | 0.0% | 0.231 | 0.32 | **0.329** |
| Arm F: Vanilla DeepSeek (No Adapter) | 2048 | 49.1% | 1286.2tok | 0.0% | 0.271 | 0.20 | **0.283** |

### SymPy Dual-Layer Bonus

Additional answers rescued by Layer 2 symbolic equality (would be FAIL in string-only mode):

| Arm | Budget | SymPy-rescued correct answers |
|-----|--------|-------------------------------|

## 2. Effect Sizes (Cohen's d + CLES)

Cohen's d on J/correct per arm-pair. CLES = P(C cheaper than comparison arm).
Positive d = arm C has lower J/correct (better). |d| > 0.8 = large effect.

| Budget | Comparison | Cohen's d (C vs X) | CLES | Bootstrap 95% CI (J/correct_X - J/correct_C) | Excludes Zero |
|--------|-----------|-------------------|------|---------------------------------------------|---------------|
| 256 | Arm A: Ordinary CoT | -1.7897 | 0.9541 | [1554.53, 20403.05] | ✓ |
| 256 | Arm B: Answer-Only | -0.3915 | 0.6416 | [-344.88, 630.69] | ✗ |
| 256 | Arm D: Router-Only Control | -1.2539 | 0.8 | [1589.99, 20921.70] | ✓ |
| 256 | Arm E: CoT + Python Sandbox | -1.8096 | 0.9257 | [601.19, 3609.55] | ✓ |
| 256 | Arm F: Vanilla DeepSeek (No Adapter) | N/A | N/A | N/A | ✗ |
| 512 | Arm A: Ordinary CoT | -2.1177 | 0.8676 | [3285.88, 38127.10] | ✓ |
| 512 | Arm B: Answer-Only | -0.3175 | 0.6481 | [-283.11, 1389.67] | ✗ |
| 512 | Arm D: Router-Only Control | -1.583 | 0.8 | [3338.75, 39335.57] | ✓ |
| 512 | Arm E: CoT + Python Sandbox | -1.925 | 0.8635 | [1451.29, 7084.95] | ✓ |
| 512 | Arm F: Vanilla DeepSeek (No Adapter) | -1.3387 | 0.8243 | [2308.90, 13063.69] | ✓ |
| 1024 | Arm A: Ordinary CoT | -2.1667 | 0.8973 | [7006.60, 76136.37] | ✓ |
| 1024 | Arm B: Answer-Only | -0.2635 | 0.6387 | [-296.01, 2962.57] | ✗ |
| 1024 | Arm D: Router-Only Control | -1.5623 | 0.7649 | [6994.53, 78232.17] | ✓ |
| 1024 | Arm E: CoT + Python Sandbox | -1.9569 | 0.873 | [3188.00, 14361.69] | ✓ |
| 1024 | Arm F: Vanilla DeepSeek (No Adapter) | -0.7942 | 0.8243 | [430.13, 4406.81] | ✓ |
| 2048 | Arm A: Ordinary CoT | -2.1463 | 0.8595 | [14016.78, 147932.39] | ✓ |
| 2048 | Arm B: Answer-Only | -0.2244 | 0.6263 | [-271.50, 6035.34] | ✗ |
| 2048 | Arm D: Router-Only Control | -1.5831 | 0.7757 | [14024.80, 152220.63] | ✓ |
| 2048 | Arm E: CoT + Python Sandbox | -1.9169 | 0.8824 | [6420.08, 27961.39] | ✓ |
| 2048 | Arm F: Vanilla DeepSeek (No Adapter) | -0.7052 | 0.834 | [-1003.24, 3498.63] | ✗ |

## 3. Per-Category Accuracy with Wilson 95% CIs

### Budget = 256 tokens

| Category | N | A: Acc [CI] | B: Acc [CI] | C: Acc [CI] | D: Acc [CI] | E: Acc [CI] | F: Acc [CI] |
|----------|---|---|---|---|---|---|---|
| `arithmetic_word_problem` | 10 | 20% [6,51] | 50% [24,76] | 45% [21,72] | 20% [6,51] | 60% [31,83] | 0% [0,28] |
| `atomic_scientific_fact` | 10 | 10% [2,40] | 60% [31,83] | 18% [5,48] | 10% [2,40] | 10% [2,40] | 0% [0,28] |
| `chinese_remainder_theorem` | 10 | 0% [0,28] | 10% [2,40] | 9% [2,38] | 0% [0,28] | 10% [2,40] | 0% [0,28] |
| `combinatorics_combinations` | 10 | 20% [6,51] | 80% [49,94] | 9% [2,38] | 20% [6,51] | 20% [6,51] | 0% [0,28] |
| `combinatorics_derangements` | 10 | 0% [0,28] | 20% [6,51] | 100% [74,100] | 0% [0,28] | 0% [0,28] | 0% [0,28] |
| `diophantine_factorization` | 8 | 0% [0,32] | 12% [2,47] | 0% [0,26] | 0% [0,32] | 0% [0,32] | 0% [0,35] |
| `discrete_recurrences` | 0 | — | — | 100% [74,100] | — | — | — |
| `elementary_algebra` | 0 | — | — | 0% [0,26] | — | — | — |
| `euler_totient` | 0 | — | — | 64% [35,85] | — | — | — |
| `fermat_modular_orders` | 0 | — | — | 18% [5,48] | — | — | — |
| `geometric_inradius` | 0 | — | — | 90% [60,98] | — | — | — |
| `harmonic_work_rates` | 0 | — | — | 30% [11,60] | — | — | — |
| `legendre_prime_valuation` | 0 | — | — | 20% [6,51] | — | — | — |
| `modular_arithmetic` | 0 | — | — | 0% [0,28] | — | — | — |
| `physics_kinematics` | 0 | — | — | 100% [72,100] | — | — | — |
| `polynomial_invariants` | 0 | — | — | 0% [0,28] | — | — | — |
| `quadratic_algebra` | 0 | — | — | 30% [11,60] | — | — | — |
| `sequences_progressions` | 0 | — | — | 70% [40,89] | — | — | — |
| `triangle_geometry` | 0 | — | — | 0% [0,28] | — | — | — |

### Budget = 512 tokens

| Category | N | A: Acc [CI] | B: Acc [CI] | C: Acc [CI] | D: Acc [CI] | E: Acc [CI] | F: Acc [CI] |
|----------|---|---|---|---|---|---|---|
| `arithmetic_word_problem` | 10 | 20% [6,51] | 50% [24,76] | 45% [21,72] | 20% [6,51] | 60% [31,83] | 70% [40,89] |
| `atomic_scientific_fact` | 10 | 10% [2,40] | 60% [31,83] | 18% [5,48] | 10% [2,40] | 10% [2,40] | 0% [0,28] |
| `chinese_remainder_theorem` | 10 | 0% [0,28] | 10% [2,40] | 9% [2,38] | 0% [0,28] | 10% [2,40] | 0% [0,28] |
| `combinatorics_combinations` | 10 | 20% [6,51] | 80% [49,94] | 9% [2,38] | 20% [6,51] | 20% [6,51] | 0% [0,28] |
| `combinatorics_derangements` | 10 | 0% [0,28] | 20% [6,51] | 100% [74,100] | 0% [0,28] | 0% [0,28] | 0% [0,28] |
| `diophantine_factorization` | 8 | 0% [0,32] | 12% [2,47] | 0% [0,26] | 0% [0,32] | 0% [0,32] | 0% [0,35] |
| `discrete_recurrences` | 0 | — | — | 100% [74,100] | — | — | — |
| `elementary_algebra` | 0 | — | — | 0% [0,26] | — | — | — |
| `euler_totient` | 0 | — | — | 64% [35,85] | — | — | — |
| `fermat_modular_orders` | 0 | — | — | 18% [5,48] | — | — | — |
| `geometric_inradius` | 0 | — | — | 90% [60,98] | — | — | — |
| `harmonic_work_rates` | 0 | — | — | 30% [11,60] | — | — | — |
| `legendre_prime_valuation` | 0 | — | — | 20% [6,51] | — | — | — |
| `modular_arithmetic` | 0 | — | — | 0% [0,28] | — | — | — |
| `physics_kinematics` | 0 | — | — | 100% [72,100] | — | — | — |
| `polynomial_invariants` | 0 | — | — | 0% [0,28] | — | — | — |
| `quadratic_algebra` | 0 | — | — | 30% [11,60] | — | — | — |
| `sequences_progressions` | 0 | — | — | 70% [40,89] | — | — | — |
| `triangle_geometry` | 0 | — | — | 0% [0,28] | — | — | — |

### Budget = 1024 tokens

| Category | N | A: Acc [CI] | B: Acc [CI] | C: Acc [CI] | D: Acc [CI] | E: Acc [CI] | F: Acc [CI] |
|----------|---|---|---|---|---|---|---|
| `arithmetic_word_problem` | 10 | 20% [6,51] | 50% [24,76] | 45% [21,72] | 20% [6,51] | 60% [31,83] | 80% [49,94] |
| `atomic_scientific_fact` | 10 | 10% [2,40] | 60% [31,83] | 18% [5,48] | 10% [2,40] | 10% [2,40] | 10% [2,40] |
| `chinese_remainder_theorem` | 10 | 0% [0,28] | 10% [2,40] | 9% [2,38] | 0% [0,28] | 10% [2,40] | 0% [0,28] |
| `combinatorics_combinations` | 10 | 20% [6,51] | 80% [49,94] | 9% [2,38] | 20% [6,51] | 20% [6,51] | 100% [72,100] |
| `combinatorics_derangements` | 10 | 0% [0,28] | 20% [6,51] | 100% [74,100] | 0% [0,28] | 0% [0,28] | 0% [0,28] |
| `diophantine_factorization` | 8 | 0% [0,32] | 12% [2,47] | 0% [0,26] | 0% [0,32] | 0% [0,32] | 0% [0,35] |
| `discrete_recurrences` | 0 | — | — | 100% [74,100] | — | — | — |
| `elementary_algebra` | 0 | — | — | 0% [0,26] | — | — | — |
| `euler_totient` | 0 | — | — | 64% [35,85] | — | — | — |
| `fermat_modular_orders` | 0 | — | — | 18% [5,48] | — | — | — |
| `geometric_inradius` | 0 | — | — | 90% [60,98] | — | — | — |
| `harmonic_work_rates` | 0 | — | — | 30% [11,60] | — | — | — |
| `legendre_prime_valuation` | 0 | — | — | 20% [6,51] | — | — | — |
| `modular_arithmetic` | 0 | — | — | 0% [0,28] | — | — | — |
| `physics_kinematics` | 0 | — | — | 100% [72,100] | — | — | — |
| `polynomial_invariants` | 0 | — | — | 0% [0,28] | — | — | — |
| `quadratic_algebra` | 0 | — | — | 30% [11,60] | — | — | — |
| `sequences_progressions` | 0 | — | — | 70% [40,89] | — | — | — |
| `triangle_geometry` | 0 | — | — | 0% [0,28] | — | — | — |

### Budget = 2048 tokens

| Category | N | A: Acc [CI] | B: Acc [CI] | C: Acc [CI] | D: Acc [CI] | E: Acc [CI] | F: Acc [CI] |
|----------|---|---|---|---|---|---|---|
| `arithmetic_word_problem` | 10 | 20% [6,51] | 50% [24,76] | 45% [21,72] | 20% [6,51] | 60% [31,83] | 80% [49,94] |
| `atomic_scientific_fact` | 10 | 10% [2,40] | 60% [31,83] | 18% [5,48] | 10% [2,40] | 10% [2,40] | 10% [2,40] |
| `chinese_remainder_theorem` | 10 | 0% [0,28] | 10% [2,40] | 9% [2,38] | 0% [0,28] | 10% [2,40] | 70% [40,89] |
| `combinatorics_combinations` | 10 | 20% [6,51] | 80% [49,94] | 9% [2,38] | 20% [6,51] | 20% [6,51] | 100% [72,100] |
| `combinatorics_derangements` | 10 | 0% [0,28] | 20% [6,51] | 100% [74,100] | 0% [0,28] | 0% [0,28] | 20% [6,51] |
| `diophantine_factorization` | 8 | 0% [0,32] | 12% [2,47] | 0% [0,26] | 0% [0,32] | 0% [0,35] | 0% [0,35] |
| `discrete_recurrences` | 0 | — | — | 100% [74,100] | — | — | — |
| `elementary_algebra` | 0 | — | — | 0% [0,26] | — | — | — |
| `euler_totient` | 0 | — | — | 64% [35,85] | — | — | — |
| `fermat_modular_orders` | 0 | — | — | 18% [5,48] | — | — | — |
| `geometric_inradius` | 0 | — | — | 90% [60,98] | — | — | — |
| `harmonic_work_rates` | 0 | — | — | 30% [11,60] | — | — | — |
| `legendre_prime_valuation` | 0 | — | — | 20% [6,51] | — | — | — |
| `modular_arithmetic` | 0 | — | — | 0% [0,28] | — | — | — |
| `physics_kinematics` | 0 | — | — | 100% [72,100] | — | — | — |
| `polynomial_invariants` | 0 | — | — | 0% [0,28] | — | — | — |
| `quadratic_algebra` | 0 | — | — | 30% [11,60] | — | — | — |
| `sequences_progressions` | 0 | — | — | 70% [40,89] | — | — | — |
| `triangle_geometry` | 0 | — | — | 0% [0,28] | — | — | — |

## 4. Contamination Breakdown

| Arm | Budget | All Acc% | Clean-Only Acc% | Gap | Alert |
|-----|--------|----------|-----------------|-----|-------|
| A | 256 | 8.6% | 8.6% | 0.0pp | OK |
| A | 512 | 8.6% | 8.6% | 0.0pp | OK |
| A | 1024 | 8.6% | 8.6% | 0.0pp | OK |
| A | 2048 | 8.6% | 8.6% | 0.0pp | OK |
| B | 256 | 39.7% | 39.7% | 0.0pp | OK |
| B | 512 | 39.7% | 39.7% | 0.0pp | OK |
| B | 1024 | 39.7% | 39.7% | 0.0pp | OK |
| B | 2048 | 39.7% | 39.7% | 0.0pp | OK |
| C | 256 | 37.0% | 37.0% | 0.0pp | OK |
| C | 512 | 37.0% | 37.0% | 0.0pp | OK |
| C | 1024 | 37.0% | 37.0% | 0.0pp | OK |
| C | 2048 | 37.0% | 37.0% | 0.0pp | OK |
| D | 256 | 8.6% | 8.6% | 0.0pp | OK |
| D | 512 | 8.6% | 8.6% | 0.0pp | OK |
| D | 1024 | 8.6% | 8.6% | 0.0pp | OK |
| D | 2048 | 8.6% | 8.6% | 0.0pp | OK |
| E | 256 | 17.2% | 17.2% | 0.0pp | OK |
| E | 512 | 17.2% | 17.2% | 0.0pp | OK |
| E | 1024 | 17.2% | 17.2% | 0.0pp | OK |
| E | 2048 | 17.5% | 17.5% | 0.0pp | OK |
| F | 256 | 0.0% | 0.0% | 0.0pp | OK |
| F | 512 | 12.3% | 12.3% | 0.0pp | OK |
| F | 1024 | 33.3% | 33.3% | 0.0pp | OK |
| F | 2048 | 49.1% | 49.1% | 0.0pp | OK |

## 5. Pre-Registered Decision Rules

**Rule (from PREREG_VALUATION.md):** Arm C adds format value over Arm D ONLY IF the paired bootstrap 95% CI on (J/correct_D − J/correct_C) strictly excludes zero *and* Acc(C) ≥ Acc(D) − 2.0 pp.

| Budget | Acc_C | Acc_D | Acc_F | CI excludes zero (C vs D) | Acc condition | Verdict |
|--------|-------|-------|-------|--------------------------|---------------|---------|
| 256 | 37.0% | 8.6% | 0.0% | ✓ | ✓ | ✓ FORMAT VALUE CONFIRMED |
| 512 | 37.0% | 8.6% | 12.3% | ✓ | ✓ | ✓ FORMAT VALUE CONFIRMED |
| 1024 | 37.0% | 8.6% | 33.3% | ✓ | ✓ | ✓ FORMAT VALUE CONFIRMED |
| 2048 | 37.0% | 8.6% | 49.1% | ✓ | ✓ | ✓ FORMAT VALUE CONFIRMED |

## 6. Reproducibility

Full manifest: `EVAL_RUN_MANIFEST.json`  
Raw JSONL: `results/robust_eval/robust_eval_results.jsonl`  
Dataset SHA-256: `575B247066B65E5B8514990280C534BAA9448014E100ADCFE67D385E43B9AA8E`  
Commit: `daaced2b4f09bbfffb778df7b3467ad87fd33b5b`  
Seed: `42`  
SymPy: `1.14.0`  
