# Token Efficiency and Biologically Inspired Control

**Status:** current thesis and framing document (2026-10-23)  
**Scope:** explains what SymboLM is trying to measure, what “biologically inspired” means here, and which claims are allowed.  
**Protocol:** this document does **not** change the frozen hypotheses or endpoints in [`PREREG_VALUATION_V2.md`](../PREREG_VALUATION_V2.md). It records the framing correction that the earlier, accuracy-first documents obscured.  
**Current result:** **none for v3.** No admissible SFT adapter exists yet; the 4,800-cell evaluation has not been run.

---

## 0. The correction in one paragraph

The primary dependent variable is **token efficiency of correct reasoning**, not raw accuracy. Accuracy is a **non-inferiority constraint**: an arm that is cheaper because it is less correct has not won. The central hypothesis is that a small model can spend fewer generated tokens — and fewer joules — on a problem by allocating computation through a **fast reflex route** for atomic facts and a **bounded deliberative route** for everything else. That is a biologically inspired **control analogy**, not a claim that the model reproduces a brain, a neuron, or a biological energy budget. The old v2.1 documents reversed the emphasis: they led with accuracy and treated token compression as a secondary benefit. That framing is corrected here.

---

## 1. What is being tested

### 1.1 Primary question

> At a fixed strict-answer correctness level, does adaptive allocation — an atomic bypass plus a typed deliberative register contract — reduce **generated tokens and joules per correct answer** relative to ordinary chain-of-thought, answer-only, or a router-only control?

The preregistered primary test is narrower: **Arm C versus Arm D at 256 tokens**, with a paired bootstrap on `J/correct` and a strict-accuracy non-inferiority margin of −2 percentage points. See [`PREREG_VALUATION_V2.md`](../PREREG_VALUATION_V2.md), §4.1.

### 1.2 Primary dependent variable

The primary dependent variable is **cost per correct answer**, measured in:

- generated tokens per correct answer, and
- dynamic joules per correct answer when NVML is available.

Accuracy is measured at the same time, but it is used first as a **gate**. A token-efficiency gain that comes from answering fewer problems correctly is not a gain.

**Endpoint precision:** the frozen preregistration's formal win condition is the
`J/correct` test at 256 tokens. This framing treats token efficiency as the primary lens and
reports `tokens_per_correct` as the token-level endpoint, with its own paired interval. A
token-only win that does not satisfy the `J/correct` gate is a diagnostic, not a
preregistered win. This distinction prevents the repository from silently replacing the
frozen endpoint with a more convenient one.

### 1.3 What is not claimed

- That the register format makes a small model “smart.”
- That SymboLM is a brain-like or biologically faithful architecture.
- That the current 1.5B model uses sparse routing, a mixture of experts, or a CED/Engram design.
- That any 20×, 80×, or 224× multiplier is a measured result.
- That a partial, quota-stopped run is a negative result, a positive result, or a completed experiment.
- That raw accuracy alone is the headline.

---

## 2. Metric definitions

These are the definitions used in this repository. When a document uses a different definition, it must say so.

| Metric | Definition | Role | Current status |
|---|---|---|---|
| `strict_correct` | The declared answer matches ground truth under `eval/strict_scorer.py`; truncation without EOS is a failure. | Correctness gate | Implemented |
| `tokens_emitted` | Number of generated tokens in the cell, including the terminal EOS when emitted; prompt tokens are excluded. | Total generation cost | Implemented in `eval/robust_eval.py` |
| `thinking_tokens` | Generated tokens before the terminal answer channel (`<ans>`, `<reg:bypass>`, or `\boxed{}`). | Mechanism-specific cost | **Not yet logged separately**; must be instrumented before a thinking-token-only claim |
| `answer_tokens` | Generated tokens in the terminal answer channel, including delimiters. | Answer overhead | **Not yet logged separately** |
| `tokens_per_correct` | `sum(tokens_emitted) / strict_correct_count` over the evaluated cells. Wrong answers consume tokens and remain in the numerator. | Token-efficiency endpoint | Implemented in the report summary |
| `correct_per_1k_tokens` | `1000 × strict_correct_count / sum(tokens_emitted)`. | Reciprocal view | Implemented in the report summary |
| `dynamic_joules` | NVML energy minus the measured idle baseline, 100 ms polling. | Physical cost | Implemented when NVML is available |
| `j_per_correct` | `sum(dynamic_joules) / strict_correct_count`. | Primary physical endpoint | Implemented |
| `OAA_t` | `(1/N) × Σ Correct_i × 1[tokens_i ≤ t]`. | Overthinking-adjusted accuracy | Implemented |
| `AUC_OAA` | Normalized area under the `OAA_t` curve. | Overthinking summary | Implemented |
| `TokenEff` | `min(1, target_budget / mean_tokens)`. | Token-efficiency factor | Implemented |
| `O` | Harmonic mean of accuracy and `TokenEff`. | Secondary overthinking score | Implemented |
| `format_valid` | The arm’s declared structural contract is satisfied. | Validity gate | Implemented |
| `eos_emitted` | The real tokenizer EOS is the final emitted token. | Termination validity | Implemented |
| `BCF` | Budget-choking factor across the 256/2048 budgets. | Budget-stress diagnostic | Implemented; not a primary endpoint |

### 2.1 Rules for using the metrics

1. **Never report a token-efficiency gain without the strict correctness gate.** A cheaper wrong answer is not efficient.
2. **Never drop failures from the token denominator.** Tokens spent on wrong answers are part of the cost of the system.
3. **Never compare `tokens_emitted` across different prompt lengths or different tokenizers without saying so.**
4. **Do not call `tokens_emitted` “thinking tokens.”** It is total generation. The split in this table is a required next instrumentation step, not a current measurement.
5. **Energy claims require measured NVML data.** If NVML was unavailable, report tokens and accuracy only.
6. **Use paired resampling over problems**, not independent arm-level averages, for C-versus-D comparisons.
7. **Report confidence intervals, not just point estimates.** A difference whose interval includes zero is not evidence of a win.

### 2.2 Why total tokens are primary, with thinking tokens as a mechanism measure

Deployment cost is driven by total generated tokens. The biological analogy is specifically about **how much deliberation is allocated**, so `thinking_tokens` is the mechanism-level measure. Both are needed:

- total tokens answer “what does serving cost?”;
- thinking tokens answer “did the control policy actually reduce deliberation, or did it merely move tokens into the answer channel?”

The current harness records total tokens only. The v3 report must say that. A later instrumentation change may add the split without changing scoring.

---

## 3. “Biologically inspired” means a control analogy

The design language borrows three ideas from biological cognition. None of them is a claim of biological fidelity.

| Biological analogy | Implementation in SymboLM | What the evidence supports | What it does not support |
|---|---|---|---|
| **System 1 / reflex** | `<reg:bypass>answer</reg:bypass>` for atomic facts; no deliberative trace. | A narrow atomic-fact subset can be answered by a short bypass: 32/40 atomic items in the old strict rescore, 32/1,000 overall. | General reasoning, or a claim that the model has a fast “subcortical” module. |
| **System 2 / deliberation** | `<reg:plan>`, `<reg:deduce>`, optional `<reg:verify>`, then `</think>` and `<ans>`. | A trainable text contract exists and can be validated structurally. | That typed registers cause better reasoning, or that the model uses a separate prefrontal module. |
| **Bounded working memory** | A short register trace with explicit state and a terminal answer. | The trace is bounded and parseable; long traces are not required by the contract. | A cognitive-science model of working memory, or a claim about hippocampal function. |
| **Metabolic parsimony** | Length/token-aware reward shaping in `training/reward.py`; compact SFT targets. | The repository contains parsimony reward code and historical GRPO stages. | That parsimony training produced the v3 adapter; v3 is LoRA SFT only, and no GRPO result is part of this protocol. |
| **Router** | Prompt-level atomic/non-atomic route selection in the evaluation harness. | Arm D is a router-only control; Arm C adds the register contract. | A learned sparse router, a mixture-of-experts model, or a hardware routing claim. |
| **Tool use / extended mind** | Arm E permits Python blocks; `symbolm/tiny_terminal.py` provides deterministic SymPy checks. | In the SmolLM2-1.7B pilot, Arm E was the strongest engineering arm (23.0%, 444.30 J/correct). | That tool use is the register format, or that the Tiny Terminal is active in the v3 SFT run. |

### 3.1 The current model is dense, not a brain-like MoE

The v3 base model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, a **dense 1.5B transformer**, served with a LoRA adapter. There is no sparse expert routing, no shared global KV cache, no CED encoder, and no Engram module in the current experiment. Those are research directions, not properties of the artifact.

### 3.2 The 20× / 80× / 224× numbers

These numbers appear in historical and grant documents. They are **projections or narrow pilot ratios**, not current v3 results:

- **80×** was a pilot ratio between answer-only and CoT energy on SmolLM2; it is not a general speedup and not a v3 result.
- **20×** was an informal multiplication of “4× token compression” by “5× sparse routing.” The multiplication is not a measured compound effect, and the current model has no sparse routing. Do not use it.
- **224×** is an analytical KV-cache projection combining a hypothetical 40-token trace with a hypothetical DeepSeek-V4.1-Flash cache density. It is a co-design sketch, not a benchmark. See [`research/DEEPSEEK_V4_FLASH_AND_SYMBOLM_CO_DESIGN.md`](../research/DEEPSEEK_V4_FLASH_AND_SYMBOLM_CO_DESIGN.md).

The phrase “20-watt human brain” is a motivating analogy about metabolic efficiency. It is not an energy measurement of this model and must not be presented as one.

---

## 4. What the training does — and does not do

### 4.1 Implemented in the v3 SFT run

- One LoRA SFT adapter on the pinned base model and base revision.
- **4,600 targets:** 4,200 reasoning traces + 400 atomic bypass traces.
- **4,140 train / 460 validation**, stratified by target kind.
- The manual Qwen chat builder is used; the stock chat template is not used for training because it can discard the reasoning trace.
- The prompt is an exact token prefix; prompt labels are masked with `−100`; the target ends in the real tokenizer EOS.
- Contrastive-negative rows use the final corrected branch, never the deliberately wrong branch.
- 400 dialogue/intent rows are excluded from the math/edge thesis.
- No GRPO, no Tiny Terminal execution, and no FSM-constrained decoding in the primary v3 SFT run.

The efficiency-relevant training choices are: compact canonical traces, explicit bypass examples, a terminal answer contract, and real EOS discipline. These are **design choices**, not yet a demonstrated efficiency effect.

### 4.2 Not implemented or not yet tested

- A matched no-format adapter trained on the same data. Without it, C-versus-D is a serving-contract comparison, not a causal training-data comparison.
- Multiple SFT seeds. One seed cannot separate a real effect from run variance.
- A logged thinking-token / answer-token split.
- Efficiency-aware GRPO with a strict correctness gate. The historical reward code exists, but it is not part of the v3 protocol.
- Public-benchmark evaluation with the same strict scorer.

### 4.3 What would make the training claim stronger

1. Train a matched no-register adapter and compare C with a true no-format model.
2. Run at least three SFT seeds and report variance.
3. Instrument thinking and answer tokens separately.
4. Use an external benchmark (for example MATH500) with the same strict scorer.
5. Measure energy on at least two machines, one GPU and one edge device.
6. Have an independent reviewer audit a random sample of traces.

Until then, the training contribution is a carefully specified efficiency-oriented SFT recipe, not a proven new training paradigm.

---

## 5. Evidence ledger

### 5.1 Current v3 status

| Item | Status | Meaning |
|---|---|---|
| v3 SFT adapter | **No admissible adapter** | The r4 preflight passed, but training stopped before LoRA injection on Kaggle's `torchao 0.10.0`. r5 trained 777 steps but failed the strict generation gate because its frozen `<ans>`, `</ans>`, and `answer` rows were byte-identical. r6 repairs that artifact and has not been run. |
| r3 provenance | **Superseded** | The r3 hashes were computed over Windows CRLF files. The committed blobs are LF, so a Linux/Kaggle checkout hashed differently. |
| r4 provenance | **Validated** | `.gitattributes` forces LF; the canonical manifests, preregistration, evaluator, and kernel constants record the LF byte hashes. The Kaggle preflight passed; the data rows and split are unchanged. |
| r5 package fix | **Validated for packaging; run failed generation** | The unused `torchao` was removed before LoRA injection and the package gate requires `"torchao": "absent"`. Training completed but the strict generation gate failed; no adapter was packaged. |
| r6 token-delta repair | **Prepared, not run** | Trainable deltas for the 47 genuinely added rows break the tied-marker failure; both gates reject exactly tied rows. |
| Full 4,800-cell run | **Not started** | Requires a clean adapter and separate explicit approval. |
| v3 token-efficiency result | **None** | No v3 result exists. |

The r3 hash mismatch was a packaging/provenance defect, not a scientific result. It was fixed
in r4 with a new tag and regenerated hashes from LF bytes; r4 then passed its Kaggle
preflight. The subsequent stop at PEFT LoRA injection was a separate runtime-packaging
failure (`torchao 0.10.0`), not a data or scientific result. The r5 fix removed the unused
package. The r5 run then completed training but failed the strict generation gate because
frozen `<ans>`, `</ans>`, and `answer` output rows were exactly tied; greedy decoding could
not emit the terminal markers. r6 repairs the artifact. Do not bypass either gate.

### 5.2 Completed or historical evidence

| Evidence | Tier | Token-efficiency reading | Caveat |
|---|---|---|---|
| SmolLM2-1.7B pilot, 200 problems | B | A 7.0%/167.6 tok/2,052.41 J; B 15.0%/8.1 tok/25.84 J; C 9.5%/141.3 tok/1,265.75 J; D 10.5%/159.8 tok/1,301.41 J; E 23.0%/120.2 tok/444.30 J per correct. | C versus D paired 95% CI `[-844.57, +861.30]` includes zero. This is not format value. B is cheap but low-accuracy; E is the strongest engineering signal. |
| Task 2 SmolLM2 ablation | B | C 10.0%/125.0 tok; D 9.0%/136.4 tok; A 7.0%/167.6 tok; B 15.0%/8.1 tok. | +1.0 pp C-versus-corrupted-plan, CI includes zero. Not the v3 strict scorer. |
| v2.1 1,000-question strict rescore | B/C | 3.2% strict (32/1,000), 91.7% truncated, 823 looped traces, mean 494.5 tokens. | A failure diagnosis, not a capability result. |
| Old flagship six-arm run | Incomplete | 2,119/4,800 cells reported; 1,955 rows saved. All 800 Arm C traces lacked `<ans>`; 296 old “correct” C rows used `deduce_equality_rhs`; 692 C traces contained `</reg:verify>`. | Stopped mid-grid by Kaggle quota/infrastructure. The missing cells are an execution fact; the partial cells are not an admissible endpoint result. 37.0%, 39.7%, and “Format Value Confirmed” are retracted. |
| Post-hoc common-57 slice | Hypothesis only | C 26.3%, D 1.8%, F 42.1%, A 0%, B 29.8%, E 1.8% at 2048. | n=57, post-hoc, not reproducible under the strict declared-answer scorer for C because C lacks `<ans>`. Not a result. |

### 5.3 How to describe the quota-stopped run

The correct description is:

> “The old six-arm run stopped at 2,119 of 4,800 cells because the free Kaggle quota / infrastructure limit ended the session. The missing cells are incomplete, and the saved cells were produced by a scoring path that is not the v3 strict scorer. The run is retained as an engineering diagnostic, not as a confirmatory or negative result.”

The incorrect descriptions are:

- “The run failed, so the thesis is false.”
- “The run was complete, so the numbers are valid.”
- “The missing cells can be ignored because the remaining cells look good.”

---

## 6. Decision language

### 6.1 The preregistered win condition

At 256 tokens, Arm C must satisfy all three:

1. The paired 95% CI for `J/correct_D − J/correct_C` strictly excludes zero in the positive direction.
2. Strict accuracy is non-inferior to D: the lower bound of the paired 95% CI for `Acc_C − Acc_D` is at least −2.0 percentage points.
3. Arm C emits a terminal answer and passes the format validator in at least 90% of cells.

Token-efficiency secondary results cannot rescue a failed primary test.

### 6.2 Allowed claim if it passes

> “Under one adapter and the preregistered strict scorer, the full register contract reduced joules per correct answer relative to a router-only contract at 256 tokens without losing more than 2 percentage points of strict accuracy.”

### 6.3 Required claim if it does not pass

> “The data do not establish format value. The result is consistent with routing-only, with an underpowered test, or with a token-efficiency effect that does not survive the correctness constraint.”

### 6.4 Forbidden language

Do not use “confirmed,” “SOTA,” “general reasoning,” “brain-like,” “100%,” “20×,” or “224×” for a current SymboLM result. Do not describe a partial run as complete. Do not present an analytical projection as a measurement.

---

## 7. Where the rest of the repository fits

- [`README.md`](../README.md) — public entry point and current status.
- [`PREREG_VALUATION_V2.md`](../PREREG_VALUATION_V2.md) — frozen hypotheses, endpoints, and decision rule.
- [`docs/EVIDENCE_STATUS.md`](EVIDENCE_STATUS.md) — short claim ledger.
- [`docs/RESEARCH_REDIRECT.md`](RESEARCH_REDIRECT.md) — what survives, what is retracted, and why.
- [`docs/MARKDOWN_AUDIT.md`](MARKDOWN_AUDIT.md) — file-by-file audit of every first-party Markdown document.
- [`EVIDENCE_TRUTH_TABLE.md`](../EVIDENCE_TRUTH_TABLE.md) — historical claim audit.
- [`RESULTS_SUMMARY.md`](../RESULTS_SUMMARY.md) — old audit narrative and strict rescore.
- [`VALUATION.md`](../VALUATION.md) — SmolLM2 pilot token/energy numbers.
- [`eval/strict_scorer.py`](../eval/strict_scorer.py) — declared-answer scorer.
- [`eval/robust_eval.py`](../eval/robust_eval.py) — six-arm harness and overthinking metrics.

---

## 8. House rules

1. Token efficiency is the primary lens; accuracy is the non-inferiority gate.
2. Strict scores decide; lenient scores are diagnostics only.
3. A confidence interval that includes zero is not evidence of value.
4. An incomplete run is not a result, in either direction.
5. A biological metaphor is not a biological mechanism.
6. A projection is not a measurement.
7. A number without a command, a hash, and a receipt is not a result.
8. A negative result is allowed to be the final answer.

---

**Author:** Sanish Gyawali (Rupandehi, Nepal)  
**Date:** 2026-10-23  
**Status:** current framing document; not a substitute for the preregistration.
