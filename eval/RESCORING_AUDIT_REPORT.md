# SymboLM Honest Rescoring Audit Report (Task 1)

**Audit Protocol:** Claude Haiku 5.5 Empirical Auditing Protocol  
**Author & Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**Date:** October 17, 2026  
**Source Artifact:** `eval_v2_1000_results.json`  

---

## 1. Executive Summary: The Honest Truth

Under strict rejection of trailing-number fallbacks and unclosed truncated outputs:
- **Old Published Accuracy:** **50.5%** (505 / 1000)
- **Honest Rescored Accuracy:** **3.2%** (32 / 1000)
- **Net Accuracy Shift:** **-47.30 percentage points**
- **Lenient False Positives Eliminated:** **473 items**
- **Rigorous Genuine Solves Retained:** **32 items**

---

## 2. Category Breakdown: Old vs. Honest Rescored

| Category | Total | Old Accuracy | Honest Rescored | Delta | Retained Solves |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `arithmetic_word_problem` | 100 | 56.0% | **0.0%** | -56.0% | 0 / 100 |
| `atomic_scientific_fact` | 40 | 80.0% | **80.0%** | +0.0% | 32 / 40 |
| `chinese_remainder_theorem` | 60 | 30.0% | **0.0%** | -30.0% | 0 / 60 |
| `combinatorics_combinations` | 40 | 45.0% | **0.0%** | -45.0% | 0 / 40 |
| `combinatorics_derangements` | 50 | 100.0% | **0.0%** | -100.0% | 0 / 50 |
| `diophantine_factorization` | 50 | 6.0% | **0.0%** | -6.0% | 0 / 50 |
| `discrete_recurrences` | 60 | 100.0% | **0.0%** | -100.0% | 0 / 60 |
| `elementary_algebra` | 60 | 51.7% | **0.0%** | -51.7% | 0 / 60 |
| `euler_totient` | 50 | 60.0% | **0.0%** | -60.0% | 0 / 50 |
| `fermat_modular_orders` | 50 | 8.0% | **0.0%** | -8.0% | 0 / 50 |
| `geometric_inradius` | 40 | 47.5% | **0.0%** | -47.5% | 0 / 40 |
| `harmonic_work_rates` | 40 | 32.5% | **0.0%** | -32.5% | 0 / 40 |
| `legendre_prime_valuation` | 60 | 51.7% | **0.0%** | -51.7% | 0 / 60 |
| `modular_arithmetic` | 40 | 17.5% | **0.0%** | -17.5% | 0 / 40 |
| `physics_kinematics` | 40 | 62.5% | **0.0%** | -62.5% | 0 / 40 |
| `polynomial_invariants` | 60 | 41.7% | **0.0%** | -41.7% | 0 / 60 |
| `quadratic_algebra` | 60 | 63.3% | **0.0%** | -63.3% | 0 / 60 |
| `sequences_progressions` | 50 | 54.0% | **0.0%** | -54.0% | 0 / 50 |
| `triangle_geometry` | 50 | 36.0% | **0.0%** | -36.0% | 0 / 50 |

---

## 3. Extraction Method Distribution

| Extraction Mechanism | Count | Proportion | Scientific Validity |
| :--- | :---: | :---: | :--- |
| `truncated_incomplete_derivation` | 887 | 88.7% | Strict Rejection (Truncated / Malformed) |
| `rejected_no_valid_closure` | 73 | 7.3% | Strict Rejection (Truncated / Malformed) |
| `closed_bypass_tag` | 40 | 4.0% | Valid Grounded Closure |

---

## 4. Sample Forensics: Why False Positives Were Eliminated

Below are representative examples where the previous regex fallback credited a correct answer despite the model truncating before reaching a conclusion:

### Item `MY2021_004` (`arithmetic_word_problem`)
- **Ground Truth:** `90`
- **Old Scorer Grabbed:** `90`
- **Tokens Emitted:** `512` (Hit 512 cap)
- **Raw Truncated Tail:** `...
<reg:deduce>W_A=9·5=45 | W_B=9·5=45 | W_total=45+45=90 bands: 90 bands</reg:deduce>
<reg:verify>bands==90 and 90>0 band`

### Item `MY2021_005` (`arithmetic_word_problem`)
- **Ground Truth:** `50`
- **Old Scorer Grabbed:** `50`
- **Tokens Emitted:** `512` (Hit 512 cap)
- **Raw Truncated Tail:** `...father:son=3:1 | present: son=15 | future: 5 years later</reg:plan>
<reg:deduce>father=3·15=45 | 45+5=50 bands: 50 bands`

### Item `MY2021_010` (`arithmetic_word_problem`)
- **Ground Truth:** `85`
- **Old Scorer Grabbed:** `85`
- **Tokens Emitted:** `512` (Hit 512 cap)
- **Raw Truncated Tail:** `... father:son=5:1 | present: son=15 | future: 10 years later</reg:plan>
<reg:deduce>father=5·15=75 | 75+10=85 bands: 85 ba`

### Item `MY2021_014` (`arithmetic_word_problem`)
- **Ground Truth:** `69`
- **Old Scorer Grabbed:** `69`
- **Tokens Emitted:** `512` (Hit 512 cap)
- **Raw Truncated Tail:** `...
method: combined_work | rates: A=11 | B=12 | time: 3 | W_A=11·3 | W_B=12·3 | W_total=W_A+W_B=69 bands: 69
</think>
band`

### Item `MY2021_015` (`arithmetic_word_problem`)
- **Ground Truth:** `55`
- **Old Scorer Grabbed:** `55`
- **Tokens Emitted:** `512` (Hit 512 cap)
- **Raw Truncated Tail:** `...:1 | present: son=16 | future: 7 years later</reg:plan>
<reg:deduce>father=3·16=48 | 48+7=55 bands: 55 bands: 55</reg:de`

