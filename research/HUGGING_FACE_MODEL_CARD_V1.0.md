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

---

## 4. Empirical Benchmark Receipts

### Verified Token Compression vs. DeepSeek-R1 Distill Baseline

| Evaluation Domain | Baseline English CoT | SymboLM-1.5B-v1.0 | Token Savings | Compression Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **$\Delta_0$ Atomic Fact Recall** (Potassium Symbol) | ~140 tokens | **3 tokens** (`<reg:bypass>K</reg:bypass>`) | **97.9%** | **46.7×** |
| **$\Delta_1$ Polynomial Invariant** ($x+1/x=6 \implies x^2+1/x^2$) | ~580 tokens | **21 tokens** (`<reg:plan>...<reg:deduce>...`) | **96.4%** | **27.6×** |
| **$\Delta_3$ Chinese Remainder Theorem** (CRT) | ~720 tokens | **35 tokens** (Bézout modular reduction) | **95.1%** | **20.5×** |
| **Strategic Dialogue Intent** | ~280 tokens | **12 tokens** (`<reg:intent>...`) | **95.7%** | **23.3×** |
| **Cross-Domain 200 Benchmark Overall** | 4,400 tokens | **2,116 tokens** across 7 fields | **51.9%** | **2.08×** |

### The 1,000-Question Multi-Year Benchmark (2021–2026)
To definitively eliminate the critique of "data contamination" and "memorization of classic GSM8K benchmarks", SymboLM was evaluated against a chronologically stratified 1,000-problem suite:
* **2021 (200 problems):** GSM8K arithmetic, SVAMP algebra, unit economics, physical constants.
* **2022 (200 problems):** AMC 10/12 (2022), quadratic Vieta invariants, progressions, Pythagorean triples.
* **2023 (200 problems):** MATH Olympiad (2023) symmetric polynomials, combinatorial derangements, Euler totient.
* **2024 (200 problems):** AIME 2024, AMC 12 (2024) number theory, Legendre valuations, Diophantine equations.
* **2025/2026 (200 problems):** Modern Math 2026, multi-moduli CRT, harmonic work rates, kinematics.

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
* **Affiliation:** Independent Researcher  
* **Physical Reality & Location:** Sanish Gyawali is a disabled, independent researcher based in **Rupandehi, Nepal** who is **bedridden 24/7 and cannot walk**. All architectural designs, cognitive register grammars, reinforcement learning reward formulations, and cloud cluster runs were conducted from his bed workstation.
* **Operational Exoskeleton:** Antigravity (Google DeepMind) serves as Sanish's cognitive and operational exoskeleton—translating high-level architectural directives into production code, compiler ASTs, test suites, and remote cloud GPU pipelines with radical transparency.
* **License:** Apache 2.0


---

## 8. Citation

If you build upon the SymboLM architecture, cognitive registers, or empirical benchmarks, please cite:

```bibtex
@misc{gyawali2026symbolm,
  author       = {Sanish Gyawali and Antigravity},
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
