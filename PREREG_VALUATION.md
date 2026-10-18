# Scientific Pre-Registration: Hardware Valuation Protocol

**Study Title:** Physical Energy and Latency Valuation of SymboLM Multi-Register Reasoning vs. Router-Only and Standard CoT Baselines  
**Principal Systems Architect & Research Director:** Sanish Gyawali (Rupandehi, Nepal)  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**Adversarial Peer Reviewer & Protocol Design:** Claude (Haiku 5.5, Anthropic)  
**Pre-Registration Date:** October 17, 2026  
**Repository Branch:** `audit/empirical-hardening`  
**Evaluation Split Path:** `data/held_out_eval_200_clean.jsonl`  
**Dataset SHA-256:** `8369BD8CAF919F672D8EC87FFF2295FF6380AE9E451F668F9EEEE908F83CF07B`  
**Status:** COMMITTED & LOCKED PRIOR TO SILICON MEASUREMENTS  

---

## 1. Research Question & Hypotheses

### Primary Question
Does the SymboLM symbolic multi-register reasoning format (`<reg:plan>`, `<reg:deduce>`, `<reg:verify>`, `<reg:bypass>`) provide physical computational value (reduced energy consumption and reduced latency per correct solution) on real edge and accelerator hardware beyond what is achieved solely by a token-efficient routing mechanism?

### Hypotheses
- **$H_1$ (Format Value Hypothesis):** Arm C (Full Registers) achieves lower dynamic energy per correct answer ($J/\text{correct}$) than Arm D (Router-Only) on identical problems, with accuracy $\ge \text{Acc}_D - 2.0\%$.
- **$H_0$ (Routing-Only Null Hypothesis):** The computational savings of SymboLM stem exclusively from skipping reasoning on atomic queries (the router). Arm C and Arm D are within noise on $J/\text{correct}$, or Arm C exhibits higher energy per correct answer.

---

## 2. Pre-Registered Decision Rules (Multi-Budget Pareto Frontier)

1. **Multi-Budget Token Grid:**
   To evaluate the accuracy-energy frontier without confounding accuracy with premature truncation, all arms will be evaluated and reported across a 4-point token budget grid:
   $$\text{Token Budgets: } \{256, 512, 1024, 2048\}$$
   Truncated outputs failing to reach formal answer closure are strictly scored as **FAIL (0)**.

2. **Format Value Criterion Across Budgets:**
   Arm C (Full Registers) counts as adding empirical value over Arm D (Router-Only Control) **if and only if**:
   - At every budget where Arm D's accuracy is within 2.0 percentage points of its own peak accuracy across the grid:
     $$\text{Mean}(J/\text{correct}_C) < \text{Mean}(J/\text{correct}_D)$$
   - The 95% confidence interval of the paired difference $(J/\text{correct}_D - J/\text{correct}_C)$, computed via $B=10,000$ bootstrap resamples on identical problems, **strictly excludes zero**.
   - All budgets will be published transparently in `VALUATION.md`, including any budgets where Arm C loses.

3. **Null Interpretation:**
   If the paired difference confidence interval includes zero, or if Arm C fails the accuracy non-inferiority bound at peak budgets:
   - The scientific conclusion will explicitly state: *"The empirical efficiency advantage of SymboLM originates from the modular router mechanism rather than the register token grammar itself."*
   - No post-hoc metric reweighting or selective cherry-picking is permitted.

---

## 3. Base Model Hierarchy (In Order of Priority)

1. **Primary Flagship:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. The core backbone of all headline repo results, ensuring continuity. Target: mobile/edge phone and host CPU.
2. **Scale Point:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (Q4_K_M on laptop CPU, plus 4-bit LoRA on Kaggle T4 if quota allows). Target: consumer laptop.
3. **Cross-Family Reasoning Control:** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (4-bit). Meta LLaMA architecture with identical DeepSeek-R1 reasoning training. *(SmolLM2 is excluded as a cross-family control as it is not a reasoning model).*
*Quota rule:* If compute quota is constrained, evaluate all 5 arms on 1.5B only, and arms A and C on 7B.

---

## 4. The Five Experimental Arms

| Arm | Name | Strategy & Prompting | Register Usage | Router Active |
| :--- | :--- | :--- | :--- | :--- |
| **Arm A** | Standard CoT | Standard natural language chain-of-thought prompt ("Solve step by step..."). | None | No |
| **Arm B** | Direct Answer | Direct answer only ("Provide only the final answer directly with no explanation"). | None | No |
| **Arm C** | Full SymboLM Registers | System prompt instructing `<reg:plan>`, `<reg:deduce>`, `<reg:verify>`, `<reg:bypass>`. Router weights active. | Full | Yes |
| **Arm D** | Causal Router Control | Identical router classifier as Arm C. Atomic facts routed to `<reg:bypass>`. Multi-step problems routed to standard English CoT (Arm A prompt). | Bypass only | Yes |
| **Arm E** | CoT + Python Calculator | Standard CoT with execution sandbox. Emitted Python expressions are executed via sandboxed SymPy/Python subprocess (<500ms timeout, isolated environment). | None | No |

---

## 4. Evaluation Split & Cryptographic Anchor

- **File:** `data/held_out_eval_200_clean.jsonl`
- **Item Count:** Exactly 200 items (100% clean, verified out-of-distribution math and factual problems).
- **Contamination Verification:** Filtered through Task 4 audit; **0.0% exact prompt overlap** and **0.0% 8-gram training overlap** against the 5,000-sample training curriculum.
- **Cryptographic SHA-256 Digest:**
  ```
  8369BD8CAF919F672D8EC87FFF2295FF6380AE9E451F668F9EEEE908F83CF07B
  ```

---

## 5. Generation & Sampling Hyperparameters

To guarantee deterministic, zero-variance experimental reproducibility across runs:
- **Decoding Strategy:** Pure Greedy (`do_sample = False`)
- **Temperature:** `0.0`
- **Top-p:** `1.0`
- **Maximum Generation Budget:** `max_tokens = 4096` (arbitrary caps eliminated)
- **Repetition Penalty:** `1.1` (prevents degenerate token cycles)
- **Early Stopping:** Stop criteria on valid closure tags (`</ans>`, `\boxed{...}`)

---

## 6. Scoring Protocol (Honest Scorer)

Scoring will use the strict scoring engine established in Task 1 (`eval/rescore.py`):
1. **No Regex Tail Fallbacks:** Truncated outputs, outputs missing explicit solution closures, or outputs ending mid-token are strictly scored as **FAIL (0)**.
2. **Deterministic Equivalence:** Answers checked via symbolic equivalence (`sympy.simplify(pred - gold) == 0`) or exact canonical float/integer match.
3. **Zero Token Fallback:** Extracting arbitrary integers from unclosed reasoning traces is strictly forbidden.

---

## 7. Target Hardware & Physical Power Profiling

### A. Accelerator Target: Kaggle Cloud NVIDIA Tesla T4 GPU
- **Sampling Mechanism:** High-frequency polling via `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits` at 100ms intervals.
- **Baseline Correction:** Measured idle power $P_{\text{idle}}$ recorded over 5.0 seconds prior to inference batch.
- **Dynamic Energy Formulation:**
  $$\Delta P(t) = \max(0, P_{\text{sample}}(t) - P_{\text{idle}})$$
  $$E_{\text{sample}} = \int_{t_{\text{start}}}^{t_{\text{end}}} \Delta P(t) \, dt \approx \sum_{i=1}^{N} \Delta P(t_i) \cdot \Delta t_i \quad (\text{Joules})$$
- **Metric Formulation:**
  $$\text{Energy per Correct Answer} = \frac{\sum_{i \in \text{All Samples}} E_i}{\sum_{i \in \text{All Samples}} \mathbb{I}(\text{Correct}_i)} \quad (J/\text{correct})$$

### B. Edge / CPU Target: Host Workstation CPU
- **Timing:** High-resolution monotonic microsecond timer (`time.perf_counter_ns()`).
- **Power Classification:** In the absence of a physical inline wall wattmeter, CPU power will be calculated using the processor TDP baseline and **explicitly labeled as `ESTIMATE`** in all generated tables and figures, in full accordance with Claude Haiku 5.5's governance mandate.

---

## 8. Statistical Rigor & Output Artifacts

- **Resampling:** $B = 10,000$ paired bootstrap iterations for differences in means.
- **Confidence Intervals:** 95% empirical percentile bootstrap intervals.
- **Artifacts Generated:**
  - Raw individual query traces: `results/valuation/raw_traces_arm_{A,B,C,D,E}.json`
  - Energy profile time-series: `results/valuation/power_samples_*.csv`
  - Synthesized report: `VALUATION.md`
  - Pareto frontier plots: Energy vs. Accuracy, Latency vs. Accuracy.
