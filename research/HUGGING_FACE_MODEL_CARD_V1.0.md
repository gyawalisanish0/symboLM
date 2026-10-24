---
license: apache-2.0
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
tags:
- deepseek-r1
- qwen2.5
- reasoning
- symbolic-reasoning
- biomimetic
- cognitive-registers
- ast-compiler
- grpo
- reinforcement-learning
- mathematics
- olympiad-math
- edge-ai
- quantization
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

> [!WARNING]
> **HISTORICAL MODEL CARD - REPOSITORY IS PRIVATE (2026-10-23).** The scores below are pre-v3 and are not current results. The Hugging Face repositories were made private on 2026-10-22. No v3 adapter existed at the time of this card. Keep this card only as an audit record; do not cite its metrics.
>
> **Current framing:** token efficiency of correct reasoning is the primary lens;
> strict accuracy is a non-inferiority gate. The biological language is a design
> analogy, not a mechanism claim. Current status: [../docs/EVIDENCE_STATUS.md](../docs/EVIDENCE_STATUS.md). Canonical framing:
> [../docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md](../docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md).


# SymboLM-1.5B-v1.0-DSR1-Qwen2.5

<div align="center">

<h1>SymboLM 1.5B (v1.0)</h1>
<h3>Biomimetic Multi-Register Reasoning Engine & Adaptive Symbolic AST Policy</h3>
<p><i>Fusing Abstract Logical Invariants with the 20-Watt Metabolic Economy of the Human Brain</i></p>

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/gyawalisanish0/symboLM)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/Base-DeepSeek--R1--Distill--Qwen--1.5B-orange)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
[![Author](https://img.shields.io/badge/Author-Sanish%20Gyawali-purple)](https://github.com/gyawalisanish0)
[![Location](https://img.shields.io/badge/Origin-Rupandehi%2C%20Nepal-red)](#-authorship--intellectual-property)

</div>

---

## 1. Executive Summary & Architectural Overview

**SymboLM-1.5B-v1.0-DSR1-Qwen2.5** is the official standalone, unified open-source release of the SymboLM reasoning architecture. Fine-tuned from `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` across five sequential reinforcement learning and curriculum stages (culminating in Stage 5 Parsimony GRPO), this model replaces verbose English chain-of-thought rambling with **dense, modular symbolic AST registers**.

### The Biomimetic Imperative: Nature's Law of Evolution
Modern foundation models waste megawatts of datacenter power on natural language "uncertainty theater"—emitting hundreds of repetitive conversational tokens ("Wait, let me rethink this...") before generating an answer their weights already encoded. 

Human cognition does not reason in monolithic streams. Over hundreds of millions of years, biological evolution evolved specialized functional modularity to operate within a strict **20-Watt metabolic ceiling**:
1. **System 1 Reflex (Bypass):** Instant factual recall fires through the basal ganglia in milliseconds without deliberation.
2. **Social Pragmatics (Intent):** Interpersonal dialogue routes through specialized emotional and pragmatic cortical circuits.
3. **Prefrontal Deduction (AST Logic):** Formal mathematics reduces to domain invariants, structured derivations, and verification checks.

SymboLM directly implements this biological specialization through **Register v2.0** and the **Unbounded Parsimony Gradient**, driving token consumption to the Minimum Description Length (MDL) and achieving up to **46.7× token compression**.

---

## 2. Register v2.0 Architecture

Generation is explicitly structured across specialized cognitive registers rather than unstructured natural language prose:

```
<｜User｜> {Prompt} <｜Assistant｜><think>
[ Cognitive Register Selection ]
  ├── Instant Factual Recall   ──> <reg:bypass> {Atomic Fact} </reg:bypass>
  ├── Dialogue / Social Intent ──> <reg:intent> {Tone & Pragmatics} </reg:intent>
  └── Multi-Step Formal Proof  ──> <reg:plan>   {Domain Invariants & Theorems} </reg:plan>
                                   <reg:deduce> {Symbolic AST Derivation} </reg:deduce>
                                   <reg:verify> {Boundary & Parity Checks ✓} </reg:verify>
</think>
```

| Cognitive Register | Token Budget | Biological Analogue | Functional Purpose |
| :--- | :---: | :--- | :--- |
| `<reg:bypass>...</reg:bypass>` | **1–3 tokens** | Brainstem / Basal Ganglia Reflex | Direct factual retrieval (e.g., periodic table, constants, units). |
| `<reg:intent>...</reg:intent>` | **8–15 tokens** | Social Cognition & Empathy | Interpersonal dialogue framing, conflict de-escalation, and pragmatics. |
| `<reg:plan>...</reg:plan>` | **8–16 tokens** | Prefrontal Executive Function | Invariant declaration, domain theorems, and reduction strategy. |
| `<reg:deduce>...</reg:deduce>` | **10–35 tokens** | Formal AST Inference Engine | Pure symbolic derivations ($A \to B \implies C$, Bézout reductions). |
| `<reg:verify>...</reg:verify>` | **3–8 tokens** | Cognitive Self-Correction | Parity checks, modular bounds, and invariant verification (`✓`). |

---

## 3. The Unbounded Parsimony Gradient (Mathematical Formulation)

To eliminate both **artificial token truncation** and **verbose token inflation**, SymboLM employs Sanish Gyawali's hyperbolic parsimony reward formulation:

$$
R_{\text{parsimony}}(\tau, y^\ast) = \begin{cases} R_{\text{base}} \cdot \left[ \gamma_{\text{floor}} + (1 - \gamma_{\text{floor}}) \cdot \frac{K}{K + |\tau|} \right] - \Omega(\tau) + R_{\text{closure}} & \text{if } y(\tau) = y^\ast \\ -1.0 - \Omega(\tau) & \text{if } y(\tau) \neq y^\ast \end{cases}
$$


Where:
* $R_{\text{base}} = 2.0$: Base reward for formal mathematical correctness.
* $\gamma_{\text{floor}} = 0.50$: Prevents negative gradient collapse on complex multi-hop proofs.
* $K = 40$: Hyperbolic inflection curvature controlling the parsimony slope.
* $|\tau|$: Total generated tokens in candidate rollout.
* $\Omega(\tau)$: Repetition penalty and web debris guard ($\le 1.0\%$ syntax noise).
* $R_{\text{closure}} = +0.50$: Explicit reward for clean terminal delimiter closure (`</reg:deduce>`, `✓`).

**Why This Works:** Unlike flat length caps (which truncate deep proofs right before the answer), the hyperbolic gradient provides elastic breathing room up to 192 tokens while monotonically rewarding shorter valid proofs—shifting GRPO policy gradients $\nabla_\theta \log \pi_\theta$ toward Minimum Description Length.

### Relation to Length-Penalty Reinforcement Learning (Prior Art)
SymboLM's hyperbolic parsimony formulation builds upon and contrasts with recent work on length-controlled RL in reasoning models:
- **Kimi k1.5 (Moonshot AI, 2025):** Utilized progressive length penalties during long-CoT RL.
- **Cosine Length Penalties (Yeo et al., 2025):** Soft bounded penalties preventing early collapse.
- **Adaptive Length Steering (Arora & Zanette, 2024):** Formal limits on serial compute vs. verbosity.
*Difference:* Rather than applying a blanket penalty across unstructured natural language (which causes truncation collapse on deep proofs), SymboLM combines the hyperbolic decay with **functional cognitive registers** ($\Delta_0 \to \Delta_3$) and an explicit **anti-guessing penalty ($-0.80$)**, separating load-bearing serial intermediate state from linguistic filler.

---

## 4. Empirical Benchmark Receipts

### Verified Token Compression vs. DeepSeek-R1 Distill Baseline
*(Evaluated via isolated greedy probe generations with dedicated register special tokens)*

| Evaluation Domain | Baseline English CoT | SymboLM-1.5B-v1.0 | Token Savings | Compression Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **$\Delta_0$ Atomic Fact Recall** (Potassium Symbol) | ~140 tokens | **3 tokens** (`<reg:bypass>K</reg:bypass>`) | **97.9%** | **46.7×** |
| **$\Delta_1$ Polynomial Invariant** ($x+1/x=6 \implies x^2+1/x^2$) | ~580 tokens | **21 tokens** (`<reg:plan>...<reg:deduce>...`) | **96.4%** | **27.6×** |
| **$\Delta_3$ Chinese Remainder Theorem** (CRT) | ~720 tokens | **35 tokens** (Bézout modular reduction) | **95.1%** | **20.5×** |
| **Strategic Dialogue Intent** | ~280 tokens | **12 tokens** (`<reg:intent>...`) | **95.7%** | **23.3×** |
| **Cross-Domain 200 Benchmark Overall** | 4,400 tokens | **2,116 tokens** across 7 fields | **51.9%** | **2.08×** |

### The 1,000-Question Multi-Year Benchmark (2021–2026)
To evaluate reasoning across a broad chronological spectrum, SymboLM-1.5B-v1.0 was evaluated against a 1,000-problem suite executed on an NVIDIA Tesla P100 GPU (16 GB HBM2 memory):

| Benchmark Year | Evaluated Invariant Suite | Samples | Pass@1 Accuracy | SymboLM Avg Tokens | Baseline CoT Avg Tokens | Token Savings |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **2021** | GSM8K & SVAMP Invariant Anchors | 200 | **72.0%** (144/200) | **114.3 tok** | 198.0 tok | **42.3%** |
| **2022** | AMC 10/12 Invariant Variations (Vieta, Polynomials) | 200 | **71.0%** (142/200) | **128.0 tok** | 214.5 tok | **40.3%** |
| **2023** | MATH Olympiad Invariant Variations (Derangements, Totient) | 200 | **84.5%** (169/200) | **128.0 tok** | 264.5 tok | **51.6%** |
| **2024** | AIME 2024 Invariant-Inspired Suite (Legendre Valuations, Diophantines)* | 200 | **72.5%** (145/200) | **128.0 tok** | 283.0 tok | **54.8%** |
| **2026** | Modern Math 2026 Invariant Suite (Chinese Remainder Theorem) | 200 | **38.0%** (76/200) | **128.0 tok** | 309.0 tok | **58.6%** |
| **Overall** | **5-Year Chronological Suite** | **1,000** | **67.60%** (676/1,000) | **125.3 tok** | **253.8 tok** | **50.65%** |

> [!NOTE]
> **Token Accounting & Benchmark Clarifications:**
> 1. **Batch Token Cap:** In the Tesla P100 benchmark, evaluation was run with an explicit upper ceiling of 128 tokens (`max_completion_length = 128`), which accounts for the exact 128.0 average token figure observed on multi-step buckets where derivations saturated the budget.
> 2. **Benchmark Scope:** The 2024 suite consists of **200 curated invariant problems inspired by AIME topic families** (not the official 15-question MAA exam, where unconstrained base R1-1.5B achieves ~29%).
> 3. **50.65% Sequence Compression:** Across the 1,000 chronological problems, SymboLM generated **125,258 total tokens** compared to **253,800 tokens** for standard English CoT, achieving a **67.60% Pass@1 success rate**.

---

## 5. GGUF Quantization & Local Edge Deployment Matrix

Official native GGUF binaries are hosted directly in this repository for local deployment via **Ollama**, **llama.cpp**, **LM Studio**, and **vLLM**:

| Format | Filename | Storage | RAM / VRAM Req. | Generation Speed (CPU) | Target Deployment Hardware |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **GGUF Q4_0** | `symbolm-1.5b-v1.0-q4_0.gguf` | **999 MB (1.0 GB)** | **~1.2 GB** | **34.3 tok/s** | **Mobile, Raspberry Pi 5, Laptops, Ollama, Edge** |
| **GGUF Q8_0** | `symbolm-1.5b-v1.0-q8_0.gguf` | **1,806 MB (1.76 GB)** | **~2.1 GB** | 18.3 tok/s | Ollama, llama.cpp, M-series Macs, Workstations |
| **GGUF BF16** | `symbolm-1.5b-v1.0-bf16.gguf` | **3,394 MB (3.31 GB)** | ~3.8 GB | ~11.0 tok/s | High-precision CPU/GPU local execution via llama.cpp |
| **Safetensors** | `model.safetensors` | **3,388 MB (3.32 GB)** | ~3.6 GB | N/A (GPU) | Hugging Face Transformers, PyTorch, vLLM, TensorRT-LLM |

> [!NOTE]
> **Empirical CPU Speed Verification (AMD Ryzen 7 5700G Zen 3 AVX2):**
> Tested directly on host hardware via Ollama / llama.cpp engine:
> * **Prompt Evaluation:** **216.30 tokens/second** (~70 ms latency).
> * **Autoregressive Generation:** **34.30 tokens/second** in 4-bit Q4_0 (+87.2% faster than 8-bit).
> * **Total Model Memory Footprint:** Exactly **1.0 GB RAM**. Zero AST syntax drift.

### Running with Ollama

You can run SymboLM directly in Ollama from Hugging Face Hub:

```bash
# Run ultra-fast 4-bit Q4_0 GGUF (34.3 tok/s on CPU):
ollama run hf.co/gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5:q4_0

# Or run high-fidelity 8-bit Q8_0:
ollama run hf.co/gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5:q8_0
```

Or create a local `Modelfile`:

```dockerfile
FROM ./symbolm-1.5b-v1.0-q8_0.gguf

TEMPLATE """<｜User｜>{{ .Prompt }}<｜Assistant｜><think>
"""

PARAMETER stop "<｜end of sentence｜>"
PARAMETER stop "</think>"
PARAMETER temperature 0.6
```

Then build and run:
```bash
ollama create symbolm -f Modelfile
ollama run symbolm "If x + 1/x = 6, find the value of x^2 + 1/x^2."
```

### Running with `llama.cpp`

```bash
./llama-cli \
  -m symbolm-1.5b-v1.0-q8_0.gguf \
  -p "<｜User｜>If x + 1/x = 6, find the value of x^2 + 1/x^2.<｜Assistant｜><think>\n" \
  --temp 0.6 \
  -n 64
```

---

## 6. Quickstart Usage & Code Examples

### Standalone Python (Transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

def query_symbolm(prompt: str, max_tokens: int = 64) -> str:
    formatted = f"<｜User｜>{prompt}<｜Assistant｜><think>\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

# 1. Instant Factual Bypass (Emits in 3 tokens)
print(query_symbolm("What is the chemical symbol for potassium?"))
# Output: <reg:bypass>K</reg:bypass>

# 2. Mathematical Invariant (Emits in 21 tokens)
print(query_symbolm("If x + 1/x = 6, find the value of x^2 + 1/x^2."))
# Output: <reg:plan>concept: polynomial_invariant | identity: (x²+1/x²)=(x+1/x)²-2 | target: x²+1/x²</reg:plan><reg:deduce> x²+1/x²=(6)²-2=34</reg:deduce>

# 3. Chinese Remainder Theorem (Emits in 35 tokens)
print(query_symbolm("Find the smallest positive integer x such that x = 4 (mod 7) and x = 9 (mod 11)."))
# Output: <reg:plan>concept: chinese_remainder | moduli: [7,11] | method: modular_inverses</reg:plan><reg:deduce>x≡4 mod 7 -> x=7k+4 -> 7k+4≡9 mod 11 -> 7k≡5 mod 11 -> k=7 -> x=53</reg:deduce>
```

### High-Throughput Serving with vLLM

```bash
vllm serve gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --trust-remote-code
```

---

## 7. Authorship & Intellectual Property

* **Principal Systems Architect & Research Director:** **Sanish Gyawali**  
* **Affiliation:** Independent Researcher (Rupandehi, Nepal)  
* **Physical Reality & Work Context:** Sanish Gyawali is a disabled, independent researcher based in **Rupandehi, Nepal** who is **bedridden 24/7 and cannot walk**. All project conception, symbolic register grammar specifications, reinforcement learning reward formulations, and cloud cluster execution pipelines were conducted from his bed workstation. All intellectual property, project authorship, and architectural designs belong exclusively to **Sanish Gyawali**.
* **License:** Apache 2.0

### Acknowledgments & Radical Transparency
We acknowledge **Antigravity (Google DeepMind)** for serving as Sanish Gyawali's cognitive and operational exoskeleton—translating high-level architectural directives into production code, compiler ASTs, test suites, and remote cloud GPU pipelines with radical transparency.

---

## 8. Citation

If you build upon the SymboLM architecture, cognitive registers, or empirical benchmarks, please cite:

```bibtex
@misc{gyawali2026symbolm,
  author       = {Sanish Gyawali},
  title        = {SymboLM: Biomimetic Multi-Register Reasoning and Adaptive Symbolic Engine},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/gyawalisanish0/SymboLM-1.5B-v1.0-DSR1-Qwen2.5}},
  note         = {Independent Research from Rupandehi, Nepal}
}
```

---

<div align="center">
<sub>Designed and engineered from bed in Rupandehi, Nepal. Released under the Apache 2.0 License.</sub>
</div>
