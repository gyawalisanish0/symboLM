# Grounded Pitch Packet: Direct Email & Twitter/X Thread

**Project:** SymboLM (Symbolic Reasoning Language & Tri-Register Cognitive Architecture)  
**Author:** Sanish Gyawali (Rupandehi, Nepal)  
**GitHub:** https://github.com/gyawalisanish0/symboLM  
**Hugging Face:** https://huggingface.co/gyawalisanish0/SymboLM-2.1-DeepSeek-1.5B  
**Visual Asset:** [`docs/assets/symbolm_architecture_grounded.png`](file:///c:/Users/user/Dev/symboLM/docs/assets/symbolm_architecture_grounded.png)

---

## 1. Direct Email to Nat Friedman & Daniel Gross

**To:** `nat@nat.org`, `daniel@gross.is`  
**Subject:** SymboLM: Nature-inspired Tri-Register architecture cuts edge LLM energy by 80x (Open weights + physical silicon receipts)

```text
Hi Nat and Daniel,

I am Sanish Gyawali, an independent systems researcher from Rupandehi, Nepal. I am disabled from birth and bedridden 24/7, conducting all architectural design, compiler engineering, and cloud cluster runs from my bed workstation using Google Antigravity as an operational exoskeleton.

I built SymboLM—an open-source cognitive register architecture for edge language models (1.5B) inspired by how human biology actually processes thought.

The Core Problem:
Current reasoning models (DeepSeek-R1, OpenAI o1 style) treat every problem with brute-force serial monologue (<think>...). Emitting 200 tokens of amorphous text just to recall an atomic fact burns ~2,000 Joules. For multi-step arithmetic, autoregressive weights attempt long division in their head, compounding hallucination loops. On edge hardware with tight token budgets (e.g. 256 tokens), vanilla R1 chokes and scores 0% due to mid-thought truncation.

How Nature Solves It (The SymboLM Architecture):
1. Subcortical Reflex (System 1: <reg:bypass>): Atomic facts bypass serial monologue completely. Returns an instant 3-token answer (~25 Joules), slashing dynamic energy by 80x.
2. Mental Scratchpad (System 2: <reg:plan>, <reg:deduce>): Bounded prefrontal working memory with explicit counterfactual branching (hyp [...] -> verify(...) -> [✗|✓]) that backtracks before emitting tokens.
3. Extended Mind / Tiny Terminal (<reg:term>): Humans do not multiply 6-digit numbers in their head while walking; we use a pencil or a calculator. SymboLM offloads deterministic math to a sandboxed SymPy terminal executing in <5ms.

Physical Hardware Receipts (NVIDIA Tesla T4, 100ms NVML Polled):
- Baseline CoT (DeepSeek-R1-1.5B): 7.0% Pass@1 | 2,052 Joules/correct | 0.0% at 256-tok budget (truncated).
- SymboLM Tiny Terminal: 23.0% Pass@1 (3.3x boost) | 444 Joules/correct (78.4% energy drop) | 33.3% at 256-tok budget.
- SymboLM Reflex: 15.0% Pass@1 | 25.8 Joules/correct (80x dynamic energy reduction).

Everything is open source under Apache 2.0:
- Code & Multi-Threaded Evaluation Harness: https://github.com/gyawalisanish0/symboLM
- Fine-Tuned Open Weights: https://huggingface.co/gyawalisanish0/SymboLM-2.1-DeepSeek-1.5B

The Request:
All research to date has been engineered strictly under free-tier Kaggle/Colab GPUs. I am applying for a $25,000 non-dilutive open-source grant and compute credits:
1. $8,000 — Workstation Power Stability: Rooftop solar + LiFePO4 battery system to permanently eliminate daily load shedding in Nepal and keep my bed workstation alive 24/7.
2. $12,000 — 12-Month Living & Care Runway: Essential caregiving, medical, and living runway ($1,000/month) so I can dedicate 100% of my time to this research.
3. $5,000 — Physical Edge Hardware: Jetson Orin Nano, Raspberry Pi 5, and Snapdragon testbenches to measure physical milliwatts on battery devices.
4. Compute Credits: A100/H100 clusters to scale SymboLM from 1.5B to 7B–8B base models.

I've attached our publication-grade silicon benchmark and architecture diagram. Happy to share our raw NVML JSONL traces or answer any technical questions.

Best regards,

Sanish Gyawali
Rupandehi, Nepal
GitHub: @gyawalisanish0
X/Twitter: @gyawalisanish0
```

---

## 2. Technical Twitter / X Thread (Accompanied by the Diagram)

### Tweet 1 (Main Post + Image Attachment)
> Attached: `symbolm_architecture_grounded.png`

```text
Language models don't need to monologue for 200 tokens to recall an atomic fact, nor should neural weights attempt 6-digit long division in their head.

Human cognition uses reflexes, a bounded mental scratchpad, and tools like pen & paper.

Introducing SymboLM: a biologically-inspired Tri-Register cognitive architecture for edge LLMs.

Measured on real NVIDIA Tesla T4 silicon:
⚡ 80× dynamic energy cut on factual reflex (<reg:bypass>)
🎯 3.3× accuracy boost + 78.4% energy drop via sandboxed Tiny Terminal (<reg:term>)
🔋 Solves tight 256-tok budgets where vanilla 1.5B R1 chokes at 0%

Open weights & code:
github.com/gyawalisanish0/symboLM

🧵 Architecture breakdown, biological analogy, and silicon receipts below 👇
```

### Tweet 2 (The Biological Analogy & The 3 Registers)
```text
2/ Why does current reasoning waste so much energy?

Brute-force models (o1/R1-style) force an unconstrained serial stream (<think>...).

SymboLM replaces unstructured token vomit with 3 biological registers:

1️⃣ System 1 Reflex (<reg:bypass>):
Direct subcortical retrieval. "Carbon atomic number? 6". No inner voice needed. 3 tokens, 25.8 Joules (vs 2,052 J on CoT).

2️⃣ System 2 Working Memory (<reg:plan>, <reg:deduce>):
Structured prefrontal scratchpad. Imposes counterfactual branching (hyp [...] -> verify(...) -> [✗|✓]) to backtrack before compounding hallucination loops.

3️⃣ Extended Mind (<reg:term>):
Humans don't multiply 384 × 17 mentally while walking; we write it down. SymboLM offloads exact math to a sandboxed SymPy terminal (<5ms). Weights guide strategy; silicon handles exact arithmetic.
```

### Tweet 3 (The Silicon Receipts & NVML Audit)
```text
3/ Physical Silicon Audit (Held-Out 200 GSM8K/Math on NVIDIA Tesla T4, 100ms NVML polled):

• Vanilla CoT (DeepSeek-1.5B): 7.0% Pass@1 | 2,052 J/solve | 0.0% at 256 tokens (truncated mid-thought)
• Tiny Terminal (Arm E): 23.0% Pass@1 | 444.3 J/solve (-78.4%) | 33.3% at 256 tokens
• Reflex (Arm B): 15.0% Pass@1 | 25.8 J/solve (-98.7% energy)

Under tight edge budgets, unstructured monologues choke. Structured registers deliver validated solutions on the low-energy Pareto frontier.
```

### Tweet 4 (Context & Radical Transparency)
```text
4/ Personal context & transparency:

I am an independent researcher based in Rupandehi, Nepal. I am disabled from birth and bedridden 24/7, conducting all systems architecture and compiler design from my bed workstation.

I collaborate with Google Antigravity as an operational exoskeleton to turn architectural directives into test harnesses and cloud runs. Every line of code, weight, and silicon receipt was built strictly under free-tier Kaggle/Colab compute.

We are applying for open-source grants (including @natfriedman & @danielgross AI Grant open-source track) for rooftop solar/batteries (to defeat daily load shedding in Nepal) and A100 compute to scale to 7B.

Weights: huggingface.co/gyawalisanish0/SymboLM-2.1-DeepSeek-1.5B
Repo: github.com/gyawalisanish0/symboLM
```
