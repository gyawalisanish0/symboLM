# Research Redirect: Token Efficiency First, Biology as Analogy

**Author:** Sanish Gyawali (Rupandehi, Nepal)  
**Date:** 2026-10-23  
**Branch:** `research/rescue-v3`  
**Status:** active rescue. No v3 adapter and no v3 GPU result exist yet.

---

## 1. What was wrong with the old framing

The early SymboLM story claimed four things at once:

1. A symbolic register language makes a small model reason better.
2. The format sharply compresses reasoning and saves energy.
3. The model reaches strong multi-year benchmark accuracy.
4. The result transfers across model families.

The audit did not support that bundle. It found a broken training format, a lenient scorer,
an early-stopped evaluator, a partial GPU run, and a dataset hash mismatch. Some components
are real; the headline is not.

The deeper error was **which variable led the story**. The old documents led with accuracy
and treated token compression as a secondary benefit. That is backwards for this project.
The primary question is how many generated tokens — and how many joules — a system spends
per correct answer. Accuracy is the constraint that prevents a cheaper-but-worse system from
being called efficient.

The project is therefore redirected from **“prove SymboLM is a new reasoning paradigm”** to:

> **“Measure whether adaptive allocation — a fast atomic reflex plus a bounded deliberative
> register contract — reduces tokens and energy per correct answer without sacrificing
> correctness, and report a negative result if it does not.”**

This is a narrower and more useful claim. It is also falsifiable.

---

## 2. The corrected thesis

### 2.1 Primary dependent variable

**Token efficiency of correct reasoning.** The primary comparison is cost per correct
answer, not raw accuracy:

- generated tokens per correct answer;
- dynamic joules per correct answer when NVML is available.

Accuracy is a **non-inferiority gate**. The preregistered C-versus-D test at 256 tokens
requires the lower bound of the paired 95% CI for `Acc_C − Acc_D` to be at least −2.0
percentage points. A cheaper arm that loses more accuracy than that has not won.

### 2.2 The biological analogy

The design borrows three ideas from biological cognition:

- **System 1 / reflex:** atomic facts can terminate through `<reg:bypass>` without a long
  deliberative trace.
- **System 2 / deliberation:** harder problems use a bounded register trace
  (`<reg:plan>`, `<reg:deduce>`, optional `<reg:verify>`, then `<ans>`).
- **Metabolic parsimony:** the training and reward design prefers compact correct traces
  over verbose ones.

These are **design analogies**, not claims of biological fidelity. The current artifact is a
dense 1.5B transformer with a LoRA adapter. It does not contain a separate reflex module, a
sparse router, a mixture of experts, a CED encoder, or an Engram module. “Biomimetic” in the
old documents should be read as a metaphor.

### 2.3 What the format is for

The register language is not assumed to be a reasoning mechanism. It is a **serving
contract** with three possible benefits:

1. A short terminal-answer channel that is easy to parse.
2. An explicit place for a router to bypass deliberation on atomic facts.
3. A structure that can be validated and, later, repaired externally.

The question is whether that contract produces more correct answers per token and per joule
than the alternatives at the same budget.

---

## 3. What survives

These are mechanisms or measurements with a clear receipt:

| Finding | Why it survives | Caveat |
|---|---|---|
| Atomic facts can be answered by a short bypass reflex | The strict 1,000-question rescore found 32 correct answers, all atomic bypasses; the atomic subset was 32/40 | A narrow factual subset, not general reasoning |
| The old 1,000-question run was mostly truncated | 91.7% of outputs hit the 512-token limit; 823 looped | A failure diagnosis, not a capability claim |
| Answer-only is much cheaper than CoT on the SmolLM2 pilot | 8.1 vs 167.6 mean tokens; 20.67× compression | SmolLM2-1.7B, 200 problems; only 15% strict accuracy |
| The sandbox arm is a positive engineering signal | E 23.0% vs A 7.0%; 444.30 vs 2,052.41 J/correct | SmolLM2 pilot; external arithmetic/tool use, not format value |
| Strict scoring changes the conclusion | The old lenient score was 3.2% strict, not the old headline | The new scorer is the only admissible one for v3 |
| Routing can reduce cost on the small pilot | C 1,265.75 J/correct vs A 2,052.41 J/correct | C is not better than the router-only D; the C-vs-D interval includes zero |

## 4. What must die

These claims must not appear in a current abstract, README, grant application, or talk:

- **37.0% Pass@1 overall.** The run stopped at 2,119 / 4,800 cells. The number is a commit
  message, not a completed result.
- **39.7% reflex accuracy / 822.6 J.** Part of the same incomplete run and the same invalid
  scoring path.
- **“Format Value Confirmed.”** The Task 2 causal ablation (C 10% vs corrupted-plan 9%,
  +1 pp) had a confidence interval that included zero. The valuation (C 9.5% vs router-only
  10.5%) also failed to establish a format advantage.
- **50.5% on the 1,000-question benchmark.** Legacy lenient extraction; strict rescore is
  3.2%.
- **100% MATH500.** n=5 anecdote, not a benchmark.
- **Cross-family LLaMA universality.** n=5 transfer probe; the separate 200-problem SmolLM2
  evaluation is not a LLaMA result.
- **Sub-50-token reasoning.** The v2.1 run averaged 494.5 tokens.
- **A “1,955-record complete evaluation.”** The receipts are partial and were scored before
  the strict scorer existed.
- **20× compound compute, 80× energy, or 224× serving compression.** These are a projection,
  a narrow pilot ratio, or an invalid multiplication. None is a current result.
- **“Brain-like,” “biomimetic,” or “biologically plausible” as a mechanism claim.** These are
  design metaphors only.

The old README and whitepaper are archived under `docs/` for provenance. They are not current
claims.

### How to describe the quota-stopped run

> “The old six-arm run stopped at 2,119 of 4,800 cells because the free Kaggle quota and
> infrastructure limit ended the session. The missing cells are incomplete; the saved cells
> used a scoring path that is not the v3 strict scorer. The run is retained as an engineering
> diagnostic, not as a confirmatory or negative result.”

The quota explains **why the grid is incomplete**. It does not make the partial cells
admissible, and it does not prove or disprove the thesis.

---

## 5. What would count as success

At **256 tokens**, the full register arm (C) must:

- have a paired 95% CI for `J/correct_D − J/correct_C` that strictly excludes zero in the
  positive direction;
- be non-inferior to the router-only arm (D) on strict accuracy, with a lower 95% CI bound
  no worse than −2 percentage points;
- emit a terminal answer in at least 90% of cells and pass the format validator in at least
  90%.

If those conditions hold, the claim is a **deployment-contract** claim, not a claim that the
format makes the model smarter. The token-efficiency report must also show tokens per correct
answer and the overthinking metrics; those are secondary and cannot rescue a failed primary
test.

### What counts as a negative result

If the CI includes zero, or C is less accurate than D, or the adapter cannot emit terminal
answers, the correct conclusion is:

> “The data do not establish format value. Routing-only, tool use, or external verification
> is the better explanation.”

That is a legitimate result. It is better than another retracted headline.

---

## 6. What the SFT training does — and does not do

The v3 run is **LoRA SFT only**:

- 4,600 targets: 4,200 reasoning + 400 atomic bypass.
- 4,140 train / 460 validation, stratified, seed 42.
- Real BOS/EOS, manual Qwen chat formatting, exact prompt-prefix assertion, `−100` prompt
  labels, and a terminal declared answer.
- Contrastive negatives use the final corrected branch; 400 dialogue rows are excluded.
- No GRPO, no Tiny Terminal execution, no FSM-constrained decoding, and no MoE in this run.

The efficiency-relevant training choices are compact canonical traces, explicit bypass
examples, terminal-answer discipline, and real EOS. These are design choices, not yet a
demonstrated efficiency effect.

The existing `training/reward.py` contains length-aware and parsimony reward functions, and
older stages used GRPO. Those are **historical or future work**, not part of the v3 primary
run. Do not describe the v3 adapter as GRPO-trained or as a product of the parsimony reward.

---

## 7. Evidence hierarchy

Use this order when deciding what to cite:

### Tier A — admissible for v3 decisions

- `data/held_out_eval_200_v2.jsonl` with the LF-normalized hash after the r4 fix.
- `data/research_sft/manifest.json` and its LF-normalized train/validation hashes.
- `eval/strict_scorer.py` and the strict report produced by `eval/robust_eval.py`.
- `run_manifest.json` from the v3 SFT run.
- The preregistered 4,800-cell run, or the explicit statement that it is incomplete.

### Tier B — measured but not comparable

- The SmolLM2-1.7B 200-problem ablation and valuation (A 7%, B 15%, C 10%, D 9%; E 23%).
- The old v2.1 strict rescore (3.2%, 91.7% truncation, 823 loops, 494.5 mean tokens).
- The partial six-arm run’s strict common-57 at 2048 (C 26.3%, D 1.8%, F 42.1%, A 0%,
  B 29.8%, E 1.8%; n=57, post-hoc).

These can motivate a hypothesis. They cannot be pooled with the v3 run.

### Tier C — historical or retracted

- The v2.1 README/whitepaper headline numbers.
- The old 37.0/39.7/50.5/100% claims.
- Any score produced by `deduce_equality_rhs`, trailing-number regexes, or an Arm C stop at
  `</reg:verify>`.
- The 20×/80×/224× projections.

Tier C is preserved for audit, not for promotion.

---

## 8. The experiment that can settle it

The new protocol is [`PREREG_VALUATION_V2.md`](../PREREG_VALUATION_V2.md). In one sentence:

> One adapter, six inference contracts, four budgets, 200 held-out problems, strict
> declared-answer scoring, and a single primary C-vs-D test at 256 tokens.

The design is intentionally narrow. It does not try to prove general reasoning or SOTA. It
asks whether one serving contract is worth its token and energy overhead.

### Why six arms

- **A:** ordinary CoT, the default.
- **B:** answer-only, the token-efficiency floor.
- **C:** full registers, the proposed contract.
- **D:** router-only, the format-free control.
- **E:** CoT plus a Python sandbox, the external-verification alternative.
- **F:** untouched base model, the adapter sanity check.

### Why one adapter

Training separate adapters for every arm would be a stronger causal design, but it is a
larger experiment. This run tests the deployment question: **given the adapter we can train
once, how should it be served?** A matched no-format adapter is the next experiment if C
shows a signal.

### Why strict scoring

The old harness could call an answer correct without `<ans>`, and it could stop Arm C before
the answer channel. That is not a measurement of the format; it is a measurement of the
parser’s optimism. V3 uses declared answers and EOS-aware truncation.

### Why the token split matters

`tokens_emitted` is total generation. A register trace could move tokens from the reasoning
channel into the answer channel and look “efficient” only because the split changed. The
harness does not yet log `thinking_tokens` and `answer_tokens` separately. That instrumentation
is required before any claim about “thinking-token compression” specifically.

---

## 9. Kill criteria and decision rule

The rescue stops or narrows if:

- The fresh adapter cannot produce terminal answers in the generation check.
- More than 10% of Arm C cells are format-invalid.
- C does not beat D on the primary energy endpoint at 256.
- C loses more than 2 percentage points of strict accuracy versus D.
- The unique-prompt or duplicate-cluster sensitivity analysis reverses the result.
- E dominates C on accuracy and energy, making the register layer unnecessary.
- The adapter is worse than the untouched base model F at the same budget.

If any of these happen, the public claim becomes a **negative systems result**. The repository
stays, the receipts stay, and the retraction stays.

---

## 10. What would make this a real paper

Not another bigger claim. Six things:

1. **Matched training.** At least two adapters trained on matched data: register and
   no-register. Then C vs D is a training-data comparison, not only a prompt comparison.
2. **Multiple seeds.** At least three SFT seeds; report variance, not one lucky run.
3. **Token-split instrumentation.** Log thinking tokens and answer tokens separately.
4. **External benchmark.** MATH500 or a comparable public benchmark, evaluated with the same
   strict scorer, not a custom template-only set.
5. **Energy on two machines.** At least one NVIDIA GPU with NVML and one CPU/edge device,
   with the same model and prompts.
6. **Human or independent audit.** A reviewer checks a random sample of traces against
   ground truth, especially the 173 unique prompts.

Until then, the project is an audit plus a small preregistered systems experiment. That is a
legitimate scope.

---

## 11. Immediate checklist

- [x] Replace fake BOS/EOS with real special tokens.
- [x] Build SFT targets manually; stop using the Qwen chat template for training.
- [x] Remove dialogue rows and wrong contrastive branches.
- [x] Add prompt-prefix and label-mask assertions.
- [x] Build a canonical 4,600-target split with a hash manifest.
- [x] Correct `MY2021_001` to 169.5.
- [x] Add duplicate-prompt flags to the benchmark.
- [x] Add strict declared-answer scoring and EOS-aware truncation.
- [x] Let strict generation reach real EOS; do not stop Arm C early at an answer.
- [x] Add run manifests and a mixed-kind post-training generation check.
- [x] Pass the local test suite (65 tests).
- [x] Freeze `PREREG_VALUATION_V2.md`.
- [x] Commit the rescue branch and tag it `sft-v3-20261022-r3` (r2 is superseded).
- [x] Prepare the Kaggle SFT kernel at `experiments/kaggle/sft_v3/`.
- [x] r3: require exact tag/commit with no branch fallback.
- [x] r3: pin the base revision and exact runtime packages; record torch.
- [x] r3: initialize input and output rows for genuinely added tokens.
- [x] r3: separate Trainer checkpoints from the final adapter and verify actual tensors.
- [x] Correct the repository-wide framing around token efficiency and the biology analogy.
- [ ] Fix the CRLF/LF provenance mismatch and create the `r4` tag.
- [ ] Run fresh SFT on Kaggle (kernel prepared, currently `ERROR`).
- [ ] Pass the mixed-kind generation check and package gate.
- [ ] Run the 100 × 4 diagnostic.
- [ ] Run the full 200 × 4 × 6 experiment (requires separate explicit approval).
- [ ] Update README/results with receipts, or publish the negative result.

---

## 12. House rules for claims

1. Token efficiency is the primary lens; strict accuracy is the non-inferiority gate.
2. A number without a command, a hash, and a receipt is not a result.
3. Strict and lenient scores may both be shown, but only strict scores decide.
4. Never call a run complete when it stopped early.
5. Never use “confirmed” for a confidence interval that includes zero.
6. Never compare an adapter arm to a separately trained baseline and call it a controlled
   ablation.
7. Never present a projection as a measurement, or a biological metaphor as a mechanism.
8. Preserve old artifacts; label them historical rather than deleting them.
9. A negative result is allowed to be the final answer.
