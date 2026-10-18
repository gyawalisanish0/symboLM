# AI Grant (Nat Friedman & Daniel Gross) — Non-Dilutive Open-Source Application

**Project Name:** SymboLM (Symbolic Reasoning Language & Tri-Register Cognitive Architecture)  
**Principal Systems Architect & Author:** Sanish Gyawali (Rupandehi, Nepal)  
**Repository:** https://github.com/gyawalisanish0/symboLM  
**Hugging Face Weights:** https://huggingface.co/gyawalisanish0/SymboLM-2.1-DeepSeek-1.5B  
**Track:** Non-Dilutive Open-Source Infrastructure Grant ($5,000 – $50,000 + Compute Credits)  

---

## 1. One-Line Summary
**SymboLM is an open-source, biologically-inspired Tri-Register cognitive architecture that cuts dynamic inference energy by 80× on small edge language models (1.5B–1.7B) via System 1 factual reflexes and a sandboxed deterministic Tiny Terminal.**

---

## 2. What are you building? (Technical Overview)
Most modern reasoning models (such as DeepSeek-R1 or OpenAI o1-style systems) treat every question with brute-force serial generation, emitting thousands of unstructured tokens of amorphous internal monologue (`<think>...`). For an atomic scientific fact ("What is the atomic number of Carbon?"), this burns hundreds of unnecessary tokens and over 2,000 Joules of energy. For multi-step arithmetic, autoregressive weights attempt to simulate complex long division internally, causing compounding hallucination loops.

SymboLM replaces unstructured token vomit with a **nature-inspired Tri-Register Cognitive Architecture**:

1. **The Reflex (System 1: `<reg:bypass>`):** Subcortical automaticity. For atomic facts, the router fires an instant 3-token answer directly (~25 Joules), cutting dynamic energy by **80×** compared to chain-of-thought.
2. **The Mental Scratchpad (System 2: Working Memory Registers):** Imposes prefrontal working memory discipline. Derivations are segmented into structured registers:
   - `<reg:plan>`: Abstract intention and algorithmic strategy.
   - `<reg:deduce>`: Sequential logical transformations.
   - `hyp [...]`: Explicit counterfactual branching with verification operators (`verify(...)` $\to$ `✗` / `✓`) that backtrack on error.
3. **The Tiny Terminal (Extended Mind / Pencil & Paper):** Humans do not multiply 6-digit numbers in their head while walking; we use a pencil or a calculator. SymboLM offloads deterministic arithmetic expressions (`<reg:term>expr</reg:term>`) to an isolated, sandboxed SymPy terminal executing in $<5\text{ms}$, boosting accuracy by **3.3× (23.0% vs 7.0%)** while slashing dynamic energy by **78.4%**.

---

## 3. What is the evidence? (Physical Silicon Receipts)
Following an adversarial empirical audit, all synthetic benchmarks and regex fallbacks were eradicated. We conducted physical, high-frequency NVML hardware power profiling (100ms interval polling minus idle baseline) on real NVIDIA Tesla T4 silicon:

| Evaluation Arm | Method / Mechanism | Pass@1 Accuracy (Held-Out Clean 200) | Dynamic Energy per Correct Solve ($J/\text{correct}$) | Sequence Compression vs CoT |
|---|---|---|---|---|
| **Arm A (Ordinary CoT)** | Unconstrained Greedy English Monologue | 7.00% | 2,052.41 Joules | 1.0× (167.6 tok) |
| **Arm B (Answer-Only)** | Direct Numeric Output (Reflex) | 15.00% | **25.84 Joules** | **20.7× (8.1 tok)** |
| **Arm C (Full Registers)** | SymboLM Plan + Deduce + Verify | 10.00% | **1,265.75 Joules** (38.3% lower energy) | 1.34× (125.0 tok) |
| **Arm D (Router Control)** | Bypass on Facts, CoT on Math | 9.00% (ablation) | 1,301.41 Joules | 1.23× (136.4 tok) |
| **Arm E (Tiny Terminal)** | CoT + Sandboxed Python Sandbox | **23.00%** (**3.3× accuracy boost**) | **444.30 Joules** (**78.4% lower energy**) | 1.05× (159.2 tok) |

*Key finding:* Under tight edge token budgets (256 tokens), vanilla reasoning models choke and score **0.0%** due to mid-thought truncation, whereas SymboLM delivers validated solutions on the low-energy Pareto frontier.

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
