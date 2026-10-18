# AI Grant (Nat Friedman & Daniel Gross) — Non-Dilutive Open-Source Application

**Project Name:** SymboLM (Symbolic Reasoning Language & Tri-Register Cognitive Architecture)  
**Principal Systems Architect & Author:** Sanish Gyawali (Rupandehi, Nepal)  
**Repository:** https://github.com/gyawalisanish0/symboLM  
**Hugging Face Weights:** https://huggingface.co/gyawalisanish0/SymboLM-2.1-DeepSeek-1.5B  
**Track:** Non-Dilutive Open-Source Infrastructure Grant ($5,000 – $50,000 + Compute Credits)  

---

## 1. One-Line Summary
**SymboLM is an open-source, biologically-inspired 2nd-generation Test-Time Compute (TTC) architecture that eliminates the "Overthinking Trap" on edge LLMs (1.5B–1.7B), achieving an 80× dynamic energy cut on factual reflex and 3.3× math accuracy via a deterministic Tiny Terminal.**

---

## 2. What are you building? (Technical Overview & Theoretical Grounding)
The AI frontier has decisively shifted to **test-time compute scaling** (DeepSeek-R1, OpenAI o1/o3). However, recent landmark research has surfaced critical structural failures in 1st-generation reasoning:
1. **The "Overthinking Trap" & Uniform Compute Allocation** (*"When More Thinking Hurts"*, NeurIPS/ArXiv; *"Towards Thinking-Optimal Scaling"*, ICLR 2025): 1st-gen models force every prompt through thousands of unstructured monologue tokens (`<think>...`). On simple tasks, this causes inverse scaling—burning thousands of Joules while talking the model out of correct intuitions.
2. **The Faithfulness & Rationalization Crisis** (Xiong et al., 2025, *"Measuring the Faithfulness of Thinking Drafts in Large Reasoning Models"*): Free-form English monologue often produces unfaithful rationalizations without causal binding to the final answer.
3. **The Arithmetic Simulation Delusion**: Forcing self-attention layers to simulate multi-step floating-point division in serial text compounds hallucination loops.
4. **Edge Budget-Choking**: On edge silicon with strict token limits ($B \le 256$), 1st-gen models truncate mid-thought and score **0.0%**.

SymboLM replaces unconstrained monologue with a **nature-inspired Tri-Register Cognitive State Machine**:

1. **Subcortical Reflex (System 1: `<reg:bypass>`):** Autonomous token-level difficulty gating. Bypasses test-time compute loops completely for atomic scientific facts, delivering an instant 3-token answer (~25 Joules) and slashing dynamic energy by **80×**.
2. **Mental Scratchpad (System 2: Working Memory Registers):** Imposes formal prefrontal working memory discipline:
   - `<reg:plan>`: Bounded intention and algorithmic strategy binding.
   - `<reg:deduce>`: Structured symbolic transitions.
   - `hyp [...]`: Explicit counterfactual branching with verification operators (`verify(...)` $\to$ `✗` / `✓`) that prune invalid thoughts before token emission.
3. **The Tiny Terminal (Extended Mind / REPL Sandbox):** Humans do not multiply 6-digit numbers purely in their head while walking; we use a pencil or a calculator. SymboLM offloads deterministic arithmetic expressions (`<reg:term>expr</reg:term>`) to an isolated, sandboxed SymPy terminal executing in $<5\text{ms}$, boosting accuracy by **3.3× (23.0% vs 7.0%)** while slashing dynamic energy by **78.4%**.

---

## 3. What is the evidence? (Physical Silicon Receipts & Current Audit)
Following an adversarial empirical audit, all synthetic benchmarks and regex fallbacks were eradicated. We conducted physical, high-frequency NVML hardware power profiling (100ms interval polling minus idle baseline) on real NVIDIA Tesla T4 silicon.

### Preliminary Baseline Audit on Edge SLM (`SmolLM2-1.7B-Instruct`, Held-Out Clean 200 Problems):

| Evaluation Arm | Method / Mechanism | Pass@1 Accuracy | Dynamic Energy per Correct Solve ($J/\text{correct}$) | Mean Tokens Generated | Finding / Honest Epistemic Interpretation |
|---|---|---|---|---|---|
| **Arm B (Subcortical Reflex)** | Direct Numeric Output (`<reg:bypass>`) | 15.00% | **25.84 Joules** | **8.1 tokens** | **Factual reflex works:** Slashes energy by **80×** compared to CoT on atomic queries. |
| **Arm E (Deterministic REPL)** | CoT + Sandboxed Python REPL | **23.00%** | **444.30 Joules** | 159.2 tokens | **Tool-use works:** 3.3× accuracy boost and 78.4% energy drop by offloading arithmetic to silicon. |
| **Arm C (Full Registers)** | SymboLM Plan + Deduce + Backtrack | 10.00% | 1,265.75 Joules | 125.0 tokens | Modest +3.0 pp gain over CoT (38% lower energy), but dominated by Arm B on simple tasks. |
| **Arm D (Shuffled Plan Ablation)** | Corrupted / Mismatched Concept Plan | 9.00% | 1,301.41 Joules | 136.4 tokens | Shuffling the plan drops accuracy by only 1.0 pp; free-text register content does not carry heavy computation on base SLMs. |
| **Arm A (Ordinary CoT)** | Unconstrained Greedy English Monologue | 7.00% | 2,052.41 Joules | 167.6 tokens | Base CoT without external tools burns high energy and suffers arithmetic drift. |

### The Flagship Reasoning Model Benchmark (`DeepSeek-R1-Distill-Qwen-1.5B`):
Because a base SLM (`SmolLM2`) operates near performance floor (7–15%), it cannot settle whether cognitive registers carry computation on genuine reasoning tasks. We are currently executing the **definitive 6-arm × 4-budget ($256, 512, 1024, 2048$ tokens) evaluation** on `DeepSeek-R1-Distill-Qwen-1.5B` on Kaggle Dual Tesla T4 GPUs (including Arm F: un-adapted Vanilla DeepSeek base).

**Pre-Registered Falsification Criteria:**
If Arm C (Full Registers) fails to outperform Arm A (CoT) and Arm D (Router Control) by at least $\Delta \text{Pass@1} \ge 2.0\%$ or lower $J/\text{correct}$ with non-overlapping bootstrap 95% CIs on DeepSeek-1.5B, we will formally accept the null hypothesis that natural language register content does not carry computation. In that event, SymboLM will officially pivot to a typed executor DSL interface (Router + Deterministic Terminal), which our data already proves delivers massive efficiency gains.

---

## 4. Who is the author? (Context & Work Reality)
* **Author:** Sanish Gyawali, Independent Systems Architect & Researcher based in Rupandehi, Nepal.
* **Work Context:** I am disabled from birth and **bedridden 24/7**. I conduct all systems architecture, compiler engineering, test harness design, and cloud cluster orchestration entirely from my bed workstation.
* **AI Collaboration Transparency:** I collaborate with Antigravity (Google DeepMind) as my cognitive and operational exoskeleton—translating my high-level mathematical and architectural directives into terminal commands, compiler code, and test suites. I practice radical transparency regarding this collaboration across all papers and repositories.
* **Compute Constraints:** All research to date has been engineered strictly under free-tier compute (Kaggle T4 weekly quotas and Google Colab).

---

## 5. Funding Request & Capital Allocation ($25,000 + Compute)
A non-dilutive grant of **$25,000** plus cloud compute credits will permanently solve my physical and operational constraints for the next 12 months:

1. **$8,000 — Workstation Power Stability (Rupandehi, Nepal):** Installation of a dedicated rooftop solar array and LiFePO4 battery storage system. This eliminates daily grid load shedding, ensuring my bed workstation and local inference rigs remain operational 24/7.
2. **$12,000 — 12-Month Living & Care Runway:** Essential living, caregiver support, and medical expenses in Nepal ($1,000/month), granting me complete, unconstrained focus on open-source research.
3. **$5,000 — Physical Edge Inference Rigs:** Edge hardware evaluation bench (NVIDIA Jetson Orin Nano, Raspberry Pi 5, Snapdragon X Elite) to benchmark real physical milliwatts on battery-powered devices.
4. **Cloud Compute Credits:** Access to dedicated A100/H100 instances to pre-train and fine-tune SymboLM on 7B–8B base models (DeepSeek-R1-Distill-Qwen-7B and LLaMA-8B), scaling the architecture beyond 1.5B edge models.

---

## 6. Open Source Roadmap & Deliverables
* **Open Weights:** All fine-tuned LoRA adapters and standalone fused checkpoints released under Apache 2.0 on Hugging Face.
* **Reproducible Test Suite:** Fully deterministic, multi-threaded 6-arm evaluation harness (`eval/robust_eval.py`) with SymPy dual-layer verification and Cohen's $d$ effect sizes.
* **C++ / GGUF Edge Runtime:** Standalone llama.cpp / GGUF engine with native token-level register routing for phones, laptops, and offline robotics.
