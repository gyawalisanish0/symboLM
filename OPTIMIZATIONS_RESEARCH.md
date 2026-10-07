# SymboLM: Advanced Architectural & Training Optimizations

This document serves as a comprehensive research reference for advanced model code manipulation. As project goals, hardware constraints (e.g., upgrading from Colab T4 to A100s), or priorities (e.g., shifting from grammar adherence to raw mathematical accuracy) evolve, you can select the appropriate optimizations from this menu.

---

## 1. Precision & Quantization Strategies

The precision of the model weights and gradients dictates the absolute ceiling of your VRAM footprint and computational speed.

### Option A: FP16 Base + LoRA (Current Recommendation for T4)
*   **Concept:** Load the base model in unquantized 16-bit floating point, freeze it, and train FP16 LoRA adapters.
*   **Implementation:** Remove `BitsAndBytesConfig`, load with `torch_dtype=torch.float16`, use `paged_adamw_8bit` optimizer.
*   **Pros:** Zero quantization error (critical for delicate logic/math logits). 20-30% faster forward pass than 4-bit (no on-the-fly dequantization). Fits in 15GB VRAM for a 1.5B model (~6GB total).
*   **Cons:** Higher baseline VRAM than 4-bit. Cannot scale to 7B models on a 15GB GPU.
*   **When to use:** When using models < 3B parameters where mathematical precision is paramount.

### Option B: 4-bit NF4 QLoRA
*   **Concept:** Quantize base weights to NormalFloat4, train FP16 LoRA adapters.
*   **Implementation:** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`.
*   **Pros:** Extreme memory compression. Allows training 7B-8B models on a single 15GB T4 GPU.
*   **Cons:** Dequantization overhead slows down training steps. Minor precision loss can degrade reasoning chains slightly.
*   **When to use:** If the goal shifts to using a larger base model (e.g., `DeepSeek-R1-Distill-Llama-8B`).

### Option C: GaLore (Gradient Low-Rank Projection)
*   **Concept:** Allows **Full Fine-Tuning** (updating all base weights, no LoRA) on consumer GPUs by projecting the gradient itself into a low-rank space during optimizer updates.
*   **Implementation:** Requires custom `GaLore` optimizer replacing AdamW.
*   **Pros:** Achieves full fine-tuning performance (often better generalization than LoRA).
*   **Cons:** Optimizer step is computationally heavier. Tricky to integrate directly with TRL's `GRPOTrainer` without deep source code hacking.

---

## 2. Memory & Throughput (Attention & Forward Pass)

### Option D: PyTorch SDPA / FlashAttention-2
*   **Concept:** Hardware-aware attention that computes scores in SRAM, avoiding reading/writing the massive $N \times N$ attention matrix to global memory.
*   **Implementation:** `attn_implementation="sdpa"` (or `"flash_attention_2"` if on Ampere+ GPUs).
*   **Pros:** Slashes activation memory by up to 50%; scales quadratically better with long contexts.
*   **Cons:** `flash_attention_2` does not work on Colab T4 (Turing architecture). SDPA works on T4 but occasionally lacks support for custom attention masks.

### Option E: Sequence Packing (Multiplexing)
*   **Concept:** Instead of padding 10 short math problems to a fixed length, pack them sequentially into a single 2048-token context window.
*   **Implementation:** Use Hugging Face `DataCollatorForLanguageModeling` with `packing=True` or `unsloth`'s packing utilities.
*   **Pros:** 3x–4x increase in training throughput (Tokens/Second). Zero GPU compute is wasted on `[PAD]` tokens.
*   **Cons:** If attention masks aren't perfectly block-diagonal, the model might "cheat" or get confused by reading tokens from a different problem in the same sequence (Cross-contamination).

---

## 3. Loss & Signal Shaping (Model Code Surgery)

### Option F: Selective `lm_head` Projection (Chunking)
*   **Concept:** Qwen has 151,643 tokens. Projecting the hidden states to this vocab size for every token in the sequence consumes massive VRAM.
*   **Implementation:** Override `forward()` to slice `hidden_states[labels != -100]` *before* calling `model.lm_head()`.
*   **Pros:** Cuts loss-computation VRAM by 60%. Highly effective for tasks where the prompt is long but the generation/completion is short.
*   **Cons:** Requires manual manipulation of the model's `forward` method or a custom training loop, bypassing standard Hugging Face wrappers.

### Option G: Fused Cross-Entropy (Liger Kernel)
*   **Concept:** Replaces the sequential `lm_head` projection and PyTorch `cross_entropy` with a single custom CUDA/Triton kernel.
*   **Implementation:** `pip install liger-kernel`, apply `apply_liger_kernel_to_qwen2()`.
*   **Pros:** Saves ~2GB VRAM instantly.
*   **Cons:** Vulnerable to CUDA version mismatches. Can crash silently if mixed precision isn't handled perfectly.

### Option H: Token-Level Curriculum Learning (Dynamic Loss Weighting)
*   **Concept:** Not all tokens are equally important. We want the model to learn the symbolic grammar, but getting the final `<ans>` right is paramount.
*   **Implementation:** Create a custom loss function. Multiply the cross-entropy loss by `1.0` for reasoning trace tokens (`|`, `→`), and by `5.0` for the actual numeric answer tokens.
*   **Pros:** Dramatically accelerates mathematical accuracy convergence.
*   **Cons:** If the weight on `<ans>` is too high, the model might optimize for the answer while allowing the symbolic trace to degrade into gibberish (reward hacking in SFT).

### Option I: NEFTune (Noise Embeddings)
*   **Concept:** Add uniform noise to embeddings during SFT forward passes.
*   **Implementation:** Add a forward hook to the embedding layer: `embeds = embeds + torch.empty_like(embeds).uniform_(-5/sqrt(seq), 5/sqrt(seq))`
*   **Pros:** Prevents overfitting. Stops the model from memorizing exact dataset phrasing.
*   **Cons:** Makes the loss curve look noisier; slightly slows initial convergence.

---

## 4. Parameter Efficiency & Routing

### Option J: DoRA (Weight-Decomposed LoRA)
*   **Concept:** Standard LoRA trains $\Delta W$. DoRA decouples this into training a Magnitude vector ($m$) and a Directional matrix.
*   **Implementation:** Change `LoraConfig` to include `use_dora=True` (supported natively in PEFT now).
*   **Pros:** Consistently outperforms LoRA on complex reasoning and math tasks. Much closer to Full Fine-Tuning behavior.
*   **Cons:** Slower forward pass because weights must be dynamically normalized.

### Option K: Selective Embedding Freezing (Sparse Gradients)
*   **Concept:** Only train the embeddings for the ~30 new symbolic tokens we added; freeze the 151k baseline English tokens.
*   **Implementation:** Register a PyTorch backward hook on the embedding gradient tensor, zeroing out `grad[:151643]`.
*   **Pros:** Faster embedding backward pass; guarantees zero degradation to the model's foundational English/Math understanding.
*   **Cons:** The model cannot adjust its internal representation of numbers to better suit the symbolic format—it can only adjust how it outputs the newly defined operators.

---

## 🧭 Strategic Decision Matrix

| Scenario / Goal Shift | Recommended Optimizations to Activate |
| :--- | :--- |
| **Max Reasoning Accuracy (SFT)** | DoRA + Token-Level Loss Weighting + NEFTune |
| **Upgrade to 7B Model on T4** | 4-bit NF4 QLoRA + Fused Cross-Entropy (Liger) |
| **Speed (Time-to-Train)** | Sequence Packing + SDPA + FP16 Base + LoRA |
| **Max GRPO Rollout Size (G=8)** | Selective `lm_head` projection + 8-bit KV Cache |
