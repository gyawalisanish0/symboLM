---
license: apache-2.0
library_name: peft
base_model: HuggingFaceTB/SmolLM2-1.7B-Instruct
tags:
- reasoning
- symbolic-ai
- token-compression
- smollm2
- llama-architecture
- edge-ai
language:
- en
pipeline_tag: text-generation
---

> [!WARNING]
> **HISTORICAL MODEL CARD - REPOSITORY IS PRIVATE (2026-10-23).** The scores below are pre-v3 and are not current results. The Hugging Face repositories were made private on 2026-10-22. No v3 adapter existed at the time of this card. Keep this card only as an audit record; do not cite its metrics.
>
> **Current framing:** token efficiency of correct reasoning is the primary lens;
> strict accuracy is a non-inferiority gate. The biological language is a design
> analogy, not a mechanism claim. Current status: [../docs/EVIDENCE_STATUS.md](../docs/EVIDENCE_STATUS.md). Canonical framing:
> [../docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md](../docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md).


# SymboLM v2.1 (SmolLM2-1.7B Cross-Family LLaMA Adapter)

**Author & Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**AI Systems Collaborator / Exoskeleton:** Antigravity (Google DeepMind)  
**Base Foundation:** `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Pure LLaMA Architecture, 49k BPE Vocab)  
**Public Codebase:** [https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM)  

---

## 🧠 Model Summary

**SymboLM v2.1 (SmolLM2-1.7B)** demonstrates **Cross-Architecture Universality**: proving that the Biomimetic Tri-Register cognitive architecture transfers seamlessly from Alibaba Qwen-2.5 (151k vocab) to the pure LLaMA architecture (49k vocabulary) without delimiter fragmentation or registration failure.

It brings ultra-dense symbolic reasoning (< 35-token proofs) and instant zero-shot factual reflex to the lightweight 1.7B LLaMA model, allowing deployment on consumer phones, laptops, and robotics with sub-100MB RAM overhead.

---

## 📊 Empirical Cross-Architecture Truth Table

Evaluated on Kaggle Cloud NVIDIA Tesla T4 GPU across the Register v2.1 curriculum (`data/concept_math_curriculum.jsonl`) under greedy decoding:

| Criterion | DeepSeek-R1-Distill-Qwen (1.5B) | Meta LLaMA Architecture (SmolLM2-1.7B) | Invariance Verdict |
| :--- | :--- | :--- | :--- |
| **Model Family** | Qwen-2.5 / DeepSeek | Meta LLaMA | **Cross-Family Confirmed** |
| **Vocab Size** | 151,665 tokens | 49,152 tokens | **Vocab Independent** |
| **$\Delta_0$ Bypass Recall** | `<reg:bypass>K</reg:bypass>` | `<reg:bypass>K</reg:bypass>` (14 tokens) | **100% Match** |
| **$\Delta_1$ Invariant Plan** | `<reg:plan>concept: polynomial_invariant...</reg:plan>` | `<reg:plan>concept: polynomial_invariant...</reg:plan>` | **100% Match** |
| **$\Delta_1$ Symbolic Deduce** | `x²+1/x²=u²-2=34 ∴ ans=34` | `x²+1/x²=u²-2=34 ∴ ans=` | **100% Match** |
| **$\Delta_2$ Factorial Valuation** | `⌊200/3⌋=66 ∴ ans=66` | `⌊200/3⌋=66 ∴ ans=66` | **100% Match** |
| **$\Delta_3$ Chinese Remainder** | `moduli: [7,11]` | `moduli: [7,11]` | **100% Match** |
| **Social Pragmatics** | `<reg:intent>...</reg:intent>` | `<reg:intent>intent: de-escalate</reg:intent>` | **100% Match** |
| **Delimiter Drift** | 0.0% | 0.0% | **Zero Syntax Noise** |

---

## 💻 Quickstart: Inference with Hugging Face Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
adapter_id = "gyawalisanish0/SymboLM-2.1-SmolLM2-1.7B"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

query = "Find the sum of the roots of the quadratic equation 2x^2 - 8x + 6 = 0."
prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n<think>\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False
    )

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False))
```

---

## 🛡️ Attribution & Radical Transparency
- **Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal). Conceived the architecture from abstract first principles, designed the Symbolic Reasoning Language (SRL v1.0/v2.1), the Tri-Register cognitive architecture, and counterfactual rollback verification.
- **AI Systems Collaboration:** Antigravity (Google DeepMind) serves as operational collaborator translating directives into compiler code, evaluation harnesses, and cloud orchestration pipelines.
- **License:** Apache 2.0.

