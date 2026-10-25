# SymboLM Long-Term Roadmap

**Status:** planning document, not a result.
**Date:** 2026-10-25
**Current gate:** the 100 × 4 × 6 v3 diagnostic. The r6 Kaggle kernel is `COMPLETE`; its
adapter passed the generation and package gates and is gate-eligible. No v3 benchmark result
exists yet. A 38-prompt CPU sweep is a diagnostic only: 24/38 strict, 38/38 format,
38/38 EOS, no truncation; its failures are arithmetic/semantic, not format. See
`results/r6_closeout/CPU_SWEEP.json`.
**Companion documents:** [`EVIDENCE_STATUS.md`](EVIDENCE_STATUS.md),
[`RESEARCH_REDIRECT.md`](RESEARCH_REDIRECT.md),
[`TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`](TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md),
[`../PREREG_VALUATION_V2.md`](../PREREG_VALUATION_V2.md),
[`../task.md`](../task.md).

> This file is a plan. It does not add a measured claim. A gate is not a result until
> its receipt exists. The claim ledger remains [`EVIDENCE_STATUS.md`](EVIDENCE_STATUS.md).

---

## 0. Executive summary

SymboLM has one engineering goal and two long-term research tracks.

**Goal:** reduce the true cost of a correct answer on edge hardware — generated tokens,
latency, memory, and joules — without losing strict accuracy.

**Track A — symbolic interface scaling:** test whether a compact register contract is a
useful serving and training interface at 1.5B, then at 7B, then at 14B/32B if the effect
survives. Use a large model as a teacher only after the interface has a receipt.

**Track B — architecture-level edge efficiency:** change the compute graph, not only the
notation. Test adaptive depth, KV/state compression, quantization/pruning, and
speculative decoding against a strong static baseline, on the target device, at matched
accuracy.

**First deployment target (2026-10-24): Android phone, `llama.cpp`, 0.5B Q4 student first.**
The 1.5B model remains the GPU/teacher artifact until a phone measurement says otherwise.

The tracks meet in two places:

1. **Teacher → student distillation:** a larger model generates and verifies symbolic
   traces; a small edge model is trained on them.
2. **Integration:** a compact interface plus an adaptive architecture may reduce both
   sequence cost and per-token cost. The interaction must be measured, not assumed.

The sequence is deliberately conservative:

1. Close r6 (done: artifact gates passed).
2. Confirm the 1.5B result under the frozen protocol.
3. Replicate at 7B.
4. Distill a teacher into an edge-sized student.
5. Change one architecture variable at a time.
6. Integrate and benchmark on the target device.
7. Scale out only if the evidence supports it; otherwise publish the boundary or negative
   result.

**The word “proven” is reserved for Tier 3–4 evidence in §3, not for a single run.**

---

## 1. Research questions and hypotheses

### H1 — Interface hypothesis (current v3)

> At a fixed strict-accuracy level, a compact register contract reduces generated tokens
> and joules per correct answer relative to a router-only contract, at the same maximum
> generation budget.

The frozen v3 test is Arm C versus Arm D at 256 tokens, with a strict-accuracy
non-inferiority margin of −2 percentage points. See
[`PREREG_VALUATION_V2.md`](../PREREG_VALUATION_V2.md) §4.1.

**Status:** untested. The r6 adapter is gate-eligible; the 100 × 4 × 6 diagnostic is the
next gate.

### H2 — Scaling hypothesis

> The interface effect, if it exists at 1.5B, survives at 7B and possibly at 14B/32B.

This is not assumed. Larger models may need the format less, may ignore it, or may adopt it
with less training. The plausible outcomes are growth, flatness, or disappearance. All are
reportable.

### H3 — Teacher–student hypothesis

> A large model can generate and verify symbolic traces that train a smaller edge model to
> retain most of the teacher’s accuracy at a fraction of the token, memory, and energy cost.

H3 is the most practical bridge to the edge goal. It must be compared against
natural-language distillation and answer-only distillation, not against an untrained
student.

### H4 — Architecture hypothesis

> An architecture-level change — adaptive depth, KV/state compression, quantization/pruning,
> or speculative decoding — reduces on-device latency or energy at matched accuracy
> relative to the strongest static baseline.

H4 is not a claim that symbols cause efficiency. It is a separate compute-graph claim.

### H5 — Integration hypothesis

> The symbolic interface and the architecture change compose, producing a larger measured
> gain than either alone.

This is an interaction test. It is the last hypothesis, not the first.

### What is not being claimed

- That the model thinks like a human, or without language.
- That the current model is a sparse MoE, a CED system, or an Engram system.
- That token compression automatically becomes energy savings.
- That a projection, a pilot ratio, or a biological metaphor is a measurement.
- That the old 20×/80×/224× figures are results.
- That a single 1.5B run proves a general law.

The cognitive-science question — whether human thought is language-independent — is a
source of hypotheses, not a result this engineering program can establish. The AI program
tests a narrow interface and architecture question. It should not be sold as a theory of
human cognition.

---

## 2. Starting point

### 2.1 What exists

| Asset | State | What it can support |
|---|---|---|
| Canonical SFT data | 4,600 targets: 4,200 reasoning + 400 bypass; 4,140/460 split; LF hashes | A reproducible training corpus |
| Benchmark | 200 held-out rows, 173 unique prompts, 20 duplicate groups | A paired evaluation set with sensitivity analyses |
| v3 protocol | Frozen six arms × four budgets; strict scorer; C-vs-D primary at 256 | One narrow confirmatory test |
| Base model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, dense transformer, pinned revision | 1.5B SFT and edge proxy |
| r6 repair | Token deltas for the 47 added rows; tied-row gates; generation and package gates passed | A gate-eligible 1.5B SFT adapter; not a benchmark result |
| Eval harness | Strict declared-answer scoring, EOS-aware truncation, energy path; `robust_eval.py` is the v2.1 path | A v3 diagnostic harness still has to be built and parity-checked |
| Old evidence | Retracted headlines, Tier B pilots, incomplete six-arm run | Hypothesis generation only |

### 2.2 What is missing

- A v3-compatible diagnostic harness; `eval/robust_eval.py` is the v2.1 path.
- The 100 × 4 × 6 diagnostic.
- The full 4,800-cell confirmatory run (separate explicit approval required).
- A matched no-register adapter for a training-data comparison.
- Multiple SFT seeds.
- Thinking-token / answer-token split instrumentation.
- Any 7B, 14B, or 32B run.
- Any teacher–student distillation run.
- Any architecture-level modification or on-device benchmark.
- A measured latency/energy budget for the recorded Android target (Redmi 15 4G,
  Snapdragon 685, HyperOS 3, 8 GB). The runtime decision is `llama.cpp`, CPU-only first;
  no phone measurement exists yet.

### 2.3 Current evidence boundary

No v3 token-efficiency, accuracy, format-value, scaling, or edge claim is allowed yet.
The r6 artifact passed its gates, but it is not a result; the 100 × 4 × 6 diagnostic is the
current gate. A quota or infrastructure stop is **incomplete**, never a negative result.

---

## 3. Definitions and the proof ladder

### 3.1 Metrics

| Metric | Definition | Role |
|---|---|---|
| `strict_correct` | Declared answer matches ground truth under `eval/strict_scorer.py`; truncation without EOS is failure | Accuracy gate |
| `tokens_emitted` | Generated tokens in the cell, including terminal EOS; prompt tokens excluded | Token cost |
| `tokens_per_correct` | `sum(tokens_emitted) / strict_correct_count`; wrong answers remain in the numerator | Token-efficiency endpoint |
| `dynamic_joules` | Device energy minus measured idle baseline | Physical cost |
| `j_per_correct` | `sum(dynamic_joules) / strict_correct_count` | Primary physical endpoint |
| `latency_p50/p95` | Wall-clock time per answer, batch 1, on the target device | Edge latency |
| `peak_memory` | Peak RSS/VRAM during generation | Edge memory |
| `format_valid` | The arm’s declared structural contract is satisfied | Validity gate |
| `terminal_answer_rate` | Fraction of cells with a parseable terminal answer | Validity gate |
| `eos_rate` | Fraction ending in the real tokenizer EOS | Termination gate |
| `OAA_t`, `AUC_OAA`, `TokenEff`, `O` | Overthinking and budget-stress diagnostics | Secondary |

Rules:

1. Never report a cost gain without the strict-accuracy gate.
2. Never drop wrong answers from the denominator.
3. Never compare token counts across tokenizers without saying so.
4. Never call `tokens_emitted` “thinking tokens.”
5. Never substitute estimated joules for measured joules.
6. Use paired resampling over problems, not independent arm averages.
7. Report confidence intervals, not just point estimates.

### 3.2 Evidence ladder

| Tier | Evidence | What it can support |
|---|---|---|
| 0 | One run, one seed, no controls | Anecdote only |
| 1 | Preregistered, matched controls, multiple seeds, paired CI | A local claim at one model/data condition |
| 2 | Tier 1 plus a held-out external benchmark and sensitivity analyses | A stronger local claim |
| 3 | Tier 2 plus at least two model sizes or two hardware targets | Scaling or edge evidence |
| 4 | Independent replication, open artifacts, audit | “Proven” in the practical sense |

The current v3 run aims at Tier 1. A 7B replication would aim at Tier 3. An on-device
architecture result would need Tier 3 hardware evidence. “Proven” is not used for Tier 0
or Tier 1.

### 3.3 Claim templates

Only these forms are allowed, with the blanks filled by receipts:

- **1.5B pass:** “At 1.5B, under the frozen protocol, the register contract reduced
  J/correct and/or tokens/correct relative to the router-only contract at 256 tokens, with
  strict accuracy non-inferior by no more than 2 percentage points.”
- **1.5B null:** “The data do not establish format value. The result is consistent with
  routing-only or with an underpowered test.”
- **7B replication:** “The effect [persisted / did not persist] at 7B under the same
  preregistered protocol; the estimated effect was [X] with CI [Y].”
- **Teacher–student:** “The [0.5B/1.5B] student retained [X]% of the teacher’s strict
  accuracy on the held-out set, at [Y] tokens and [Z] J per correct answer on device D.”
- **Architecture:** “On device D, [early exit/quantization/state compression] reduced
  p50 latency by [X]% and energy per correct answer by [Y]% at non-inferior accuracy
  relative to the strongest static baseline.”
- **Integration:** “The interface × architecture interaction was [positive/negative/null]
  under the preregistered factorial test.”

No wording such as “confirmed,” “SOTA,” “general reasoning,” “human-like,” or “100%” is
allowed without a separate preregistered test.

---

## 4. Workstreams

### Track A — Symbolic interface scaling

#### A0. Close the 1.5B experiment

- ✅ Finish r6; the generation and package gates passed.
- Run the 100 × 4 × 6 diagnostic with a v3-compatible harness.
- Request separate approval for the full 4,800-cell run.
- Report strict accuracy, tokens/correct, J/correct, format validity, terminal-answer rate,
  EOS rate, and the C-vs-D decision.

#### A1. Strengthen the causal claim at 1.5B

The frozen v3 run uses one adapter and six serving contracts. That tests a deployment
question, not a training-data question. Add, as a separate preregistered study:

- a matched no-register adapter trained on natural-language versions of the same problems;
- a format-only control with placeholder registers;
- at least three SFT seeds;
- thinking-token / answer-token split instrumentation;
- a corrupted-register intervention to test causal use;
- one external benchmark with the same strict scorer.

This is the difference between “the serving contract is cheaper” and “the symbolic
training data causes the gain.”

#### A2. Replicate at 7B

- Candidate base: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` or another pinned
  Qwen-family 7B model. Prefer tokenizer compatibility so the same added rows can be
  reused; verify the tokenizer hash and added-token IDs before training. Do not assume
  token IDs transfer across tokenizers; if they do not, re-extend and re-run preflight.
- First run a cheap zero-shot/few-shot format-adoption probe. If the 7B model cannot follow
  the contract without SFT, that is a finding, not a reason to skip the control.
- Train register and matched no-register adapters with QLoRA on the same data.
- Start with one seed as a screening run; if the signal is plausible, run three seeds for
  confirmation.
- Evaluate on the same held-out 200 rows for comparability, plus one external benchmark.
- Report the effect size at 1.5B vs 7B, not just “it worked.”

**Decision:** if the effect disappears at 7B, report the boundary and shift effort to the
edge/teacher track. Do not keep scaling to hide a null result.

#### A3. Scale to 14B/32B only conditionally

Only if the 7B result is positive and the compute budget exists. Use 4-bit QLoRA on
A100/H100-class hardware or a cloud grant. Do not run 14B/32B on the free T4 quota unless a
specific feasibility test says it fits.

#### A4. Data scaling

The 4,600-target corpus may be enough to teach the format to 1.5B but not to 7B+. If format
adherence fails, add verified traces rather than loosening the gate. Teacher-generated data
must be filtered by a symbolic solver, unit tests, or strict answer checks; do not train on
unverified generations.

### Track B — Teacher–student distillation for the edge

#### B1. Teacher selection

Use the best verified model available: a 7B/14B/32B register-trained model, or a strong
API model if its version is pinned and its outputs are cached and reproducible. The teacher
must be better than the student on the held-out set; otherwise distillation is pointless.

#### B2. Trace generation and verification

- Generate symbolic traces on a broad corpus, not only the 200 held-out rows.
- Verify each terminal answer with the strict scorer or a symbolic solver.
- Reject malformed, looping, truncated, or unverifiable traces.
- Deduplicate and hold out a test set before training.
- Record teacher model, revision, prompt, decoding parameters, and filter rate.

#### B3. Student training

- Students: Qwen2.5-0.5B and/or the 1.5B model, depending on the target device.
- Objectives: token-level distillation on verified traces; optionally KL to the teacher’s
  distribution if logits are available.
- Controls: natural-language CoT distillation, answer-only distillation, and no
  distillation.
- Evaluate on held-out and external benchmarks, then on the edge device.

#### B4. Edge evaluation

- Batch size 1; p50/p95 latency; time-to-first-token; tokens/s; peak memory; J/correct.
- Compare against the strongest static baseline at the same accuracy, not against the
  teacher’s unconstrained cost.
- Report thermal state and sustained performance if the device throttles.

### Track C — Architecture-level edge efficiency

All architecture work happens on a new branch and tag with its own preregistration. The
frozen v3 adapter, protocol, and claim ledger are not modified by these experiments.

Do not start with a mechanism. Start with a bottleneck measurement.

#### C0. Roofline and bottleneck analysis

On the target device, measure:

- weight-loading time and memory bandwidth;
- prefill vs decode time;
- KV-cache size and growth;
- attention vs MLP time;
- tokenization/detokenization overhead;
- energy at idle, prefill, and decode.

Then choose **one** change that addresses the dominant bottleneck. If weights and memory
bandwidth dominate, early exit and quantization matter more than KV compression. If long
traces dominate, KV/state compression matters. If serial latency dominates, speculative or
parallel decoding matters.

#### C1. Quantization, pruning, and distillation baseline

This is the strongest and least exotic edge baseline:

- 8-bit and 4-bit weight-only quantization;
- quantization-aware training if accuracy drops;
- structured pruning or width reduction;
- distillation to a smaller student.

A new architecture must beat this baseline, not just the unoptimized 1.5B model.

#### C2. Adaptive depth / early exit

- Add a confidence or router head that can exit at intermediate layers.
- Train with an accuracy loss plus a compute penalty.
- Measure average layers used, latency, energy, and accuracy at matched budgets.
- Test whether easy atomic items exit early and hard items use more depth.

This is the closest architecture analogue to the original “spend computation where it
changes the answer” thesis. It must be measured, not asserted.

#### C3. KV-cache and state compression

- KV quantization, eviction, or sliding-window attention;
- recurrent/state-space hybrid layers;
- compressed global state for long traces.

Only pursue if C0 shows KV memory or attention is a real bottleneck. Symbolic traces are
short; KV compression may be irrelevant for this workload.

#### C3.5. Speculative and parallel decoding

- A small draft model proposes tokens; the target verifies.
- Grammar-constrained decoding can make symbolic continuations more predictable.
- Measure latency and energy at batch 1, including draft-model memory.

#### C4. Latent / continuous reasoning (long horizon)

This is the architecture question behind “non-linguistic thought,” but it is not the first
edge prototype. It requires a different training objective and a new preregistration.
Candidates include continuous latent reasoning, adaptive recurrent depth, and modular
neuro-symbolic controllers. Do not mix this into the v3 claim.

### Track D — Measurement and hardware

- GPU: NVML dynamic joules minus idle baseline; 100 ms polling; same protocol as v3.
- Edge: external power meter or on-device telemetry; whole-device energy; idle subtraction;
  multiple runs; report variance.
- Latency: batch 1, p50/p95, time-to-first-token, tokens/s, thermal steady state.
- Memory: peak RSS/VRAM, model file size, KV-cache size at the tested budgets.
- Every result: commit/tag, base revision, dataset hash, runtime versions, hardware,
  command, raw generations, scores, and energy method.

### Track E — Governance and reproducibility

- Every confirmatory phase gets a frozen preregistration before its first scored cell.
- Expensive runs need separate explicit approval.
- Quota/infrastructure stops are labeled incomplete.
- No optional stopping, no post-hoc endpoint swaps, no lenient scorer.
- Deviations go in a dated `DEVIATIONS.md`; a deviation that changes the primary endpoint
  invalidates the confirmatory claim.
- Retractions stay in the repository.
- Every first-party Markdown file is audited; new docs are added to the audit.
- Private HF repositories and pinned revisions remain the default until an artifact passes
  its gates.

---

## 5. Phased plan and gates

Timelines are planning estimates, not commitments. Compute estimates are rough and must be
replaced by measured receipts.

### Phase 0 — r6 closeout (done) and diagnostic (now)

**Question:** Is there a clean 1.5B adapter at all?

**Actions:**

1. ✅ Monitor the r6 kernel; verify `run_manifest.json`, generation previews, merged token
   rows, `terminal_marker_rows`, and `token_delta` evidence.
2. ✅ Generation passed: `concept_poly_30` returned `62`, `concept_bypass_183` returned
   `100`; both were terminal/EOS-valid with no loop marker.
3. ✅ Package gate passed, including `"torchao": "absent"`, merged deltas, and distinct
   terminal-marker rows. Adapter and bundle hashes match the receipt.
4. ⛔ Run the 100 × 4 × 6 diagnostic with a v3-compatible harness. The existing
   `eval/robust_eval.py` is the v2.1 path and is not v3-ready.
5. ⛔ Only then request approval for the full 4,800-cell run.

**Gate G0:** met for r6: clean adapter + package manifest
(`results/r6_closeout/R6_CLOSEOUT.json`).
**Gate G1:** diagnostic passes format/EOS/terminal-answer checks; no more than 10% Arm C
format-invalid cells. Pending.

**Deliverables:** adapter hash, run manifest, generation previews, package manifest
(available for r6); diagnostic report (pending).

**Stop:** any gate failure means no v3 result. A failure is an artifact/pipeline event, not
a scientific negative result.

### Phase 1 — 1.5B confirmatory result (weeks 1–3)

**Question:** Does C beat D on the frozen primary endpoint?

**Design:** the frozen 4,800-cell grid; no changes to prompts, budgets, arms, or scoring.
Separate explicit approval is required before launch.

**Primary:** C vs D at 256 tokens: paired 95% CI for `J/correct_D − J/correct_C` strictly
excludes zero; strict-accuracy lower bound ≥ −2 pp; C format-valid and terminal-answer rates
≥ 90%.

**Quota note:** the full 4,800-cell grid may exceed one weekly 30-hour Kaggle window
depending on generation lengths. Checkpointing and asynchronous persistence are required.
If quota ends first, the run is incomplete, not negative.

**Secondary:** tokens/correct, OAA/AUC_OAA, TokenEff, overthinking, unique-prompt and
duplicate-cluster sensitivity, E vs A, B vs A, C vs F.

**Gate G2:** the frozen decision rule passes or fails. Either outcome is published.

**Deliverable:** a v3 research note with strict receipts, positive or negative.

### Phase 1b — Causal strengthening (weeks 3–6)

**Question:** Is the effect caused by the symbolic training data, or only by the serving
contract?

**Design:** matched no-register adapter; format-only control; three seeds; token-split
instrumentation; corrupted-register intervention; one external benchmark.

**Gate G2b:** the effect survives the matched-training comparison with a CI that excludes
zero and non-inferior accuracy. If it does not, narrow the claim to a serving-contract
result and say so.

### Phase 2 — 7B replication (months 1–3)

**Question:** Does the interface effect survive at 7B?

**Design:** pinned Qwen-family 7B; QLoRA; same data and benchmark; register vs matched
no-register adapter; one screening seed, then three seeds if plausible; external benchmark.

**Gate G3:** a paired 95% CI for `J/correct_D − J/correct_C` (or the preregistered token
endpoint) strictly excludes zero, strict accuracy is non-inferior by no more than 2 pp, and
format-valid/terminal-answer rates are at least 90%. Otherwise, report the boundary. A null
at 7B stops the 14B/32B scaling branch.

**Compute:** rough planning range 30–120 GPU-hours for a screening plus confirmation,
depending on quantization and provider. Replace with a measured feasibility run.

### Phase 3 — Teacher–student edge distillation (months 3–5)

**Question:** Can a large symbolic teacher train a small edge student without losing the
accuracy that matters?

**Design:** verified teacher traces; 0.5B student first (Q4 GGUF, `llama.cpp` on the chosen
Android phone); 1.5B as a measured stretch; natural-language and answer-only distillation
controls; held-out and external evaluation; edge proxy.

**Gate G4:** student accuracy is non-inferior to the best static student baseline within a
preregistered margin (default −2 pp unless the deployment requires tighter), and
tokens/latency/energy per correct answer are lower on the Android device.

**Deliverable:** a distillation report with teacher/student revisions, filter rates, and
raw traces.

### Phase 4 — Architecture prototype (months 5–8)

**Question:** Does one architecture change beat the strongest quantized/distilled baseline
on the target device?

**Design:** C0 bottleneck analysis first on the Android phone; then one of early exit,
KV/state compression, or speculative decoding; matched accuracy; batch 1; multiple runs.
Stock `llama.cpp` cannot expose early exit cleanly, so prototype that change in PyTorch or
a pinned fork; do not mix a runtime rewrite into the first mobile benchmark.

**Gate G5:** on-device latency or energy improves with a paired CI that excludes zero, at
non-inferior accuracy. If not, publish the negative and do not stack mechanisms.

**Deliverable:** code, device benchmark script, power/latency receipt.

### Phase 5 — Integration and external validation (months 8–12)

**Question:** Do the interface and architecture compose?

**Design:** 2 × 2 factorial on the student: interface (register vs natural language) ×
architecture (static vs adaptive/quantized). Same prompts, same hardware, same accuracy
gate. Measure the interaction.

**Gate G6:** a compound claim is allowed only if the interaction is measured and positive.
Otherwise report separate main effects.

**Deliverable:** paper-ready report, open artifacts after audit, independent replication
attempt.

### Phase 6 — Scale-out and theory (year 2)

Only after G3–G6:

- 14B/32B scaling if the 7B trend is positive;
- scaling-law analysis of interface effect vs model size;
- latent/continuous reasoning and modular neuro-symbolic architectures;
- a cross-device edge efficiency benchmark;
- grants, collaborations, and independent replication.

---

## 6. Experimental controls

Every scaling or architecture claim needs the relevant controls:

| Control | What it isolates |
|---|---|
| Router-only arm D | Serving contract effect without the full register trace |
| Matched no-register adapter | Training-data effect vs serving-contract effect |
| Format-only / placeholder registers | Structural effect vs semantic content |
| Answer-only arm B | Token-efficiency floor; low-accuracy baseline |
| Untouched base model F | Adapter sanity check |
| Natural-language CoT distillation | Whether symbolic traces add value over ordinary traces |
| Strongest quantized static baseline | Whether the architecture change adds value over standard edge optimization |
| Corrupted-register intervention | Whether the model causally uses the register content |
| Multiple seeds | Run variance vs treatment effect |
| External benchmark | Overfitting to the custom 200-row set |

No control, no causal claim.

---

## 7. Statistics and decision rules

### 7.1 Confirmatory analysis

- Paired bootstrap over problems, B = 10,000, seed 42, percentile 95% CI.
- Primary endpoint fixed before the run; no optional stopping.
- Non-inferiority margin: −2.0 pp for the v3 C-vs-D test. New phases reuse −2.0 pp unless
  a deployment-specific margin is preregistered before data collection; the margin is never
  loosened after seeing results.
- Multiple seeds: report mean and variance; use a mixed-effects model or a paired
  seed-level analysis where appropriate.
- Secondary endpoints are labeled exploratory; they cannot rescue a failed primary test.

### 7.2 Power

The 200-problem benchmark may be underpowered for small accuracy differences. Before the
7B study, use the observed v3 variance to estimate power for the planned effect size. If
the study is underpowered, either increase the benchmark (with a new preregistration) or
state that the result is a screening estimate, not a confirmatory one.

### 7.3 Edge statistics

- Pair prompts across baselines and the candidate.
- Report p50 and p95 latency, not only the mean.
- Report energy per correct answer, including wrong answers in the denominator.
- Repeat runs; report variance and thermal state.
- Do not compare devices or power modes without saying so.

---

## 8. Resources and budget

### 8.1 Current resources

- Kaggle GPU quota: 30 h/week; TPU: 20 h/week; refresh weekly (as of 2026-10-24).
- Local environment: CPU-only; no GPU.
- Base model and SFT data are local/reproducible.
- No cloud budget is documented.

### 8.2 Planning estimates

| Phase | Compute | Hardware | Notes |
|---|---|---|---|
| Phase 0 | 5–15 GPU-hours | Kaggle T4 | r6 + diagnostic; generation-heavy |
| Phase 1 | 20–50 GPU-hours | Kaggle T4 | 4,800 cells; may span quota windows |
| Phase 1b | 10–30 GPU-hours | Kaggle/cloud | 3 seeds, matched adapters |
| Phase 2 | 30–180 GPU-hours | 7B QLoRA on T4/A100 | Screening + confirmation; T4 and A100 hours are not interchangeable |
| Phase 3 | 10–100 GPU-hours plus teacher inference | A100/API + T4 | Teacher cost dominates |
| Phase 4 | 20–100 GPU-hours | A100/edge device | One architecture change |
| Phase 5 | 30–100 GPU-hours | Edge device + GPU | Integration and audit |

These are planning ranges, not measured costs. Do not cite them as receipts. A T4 hour is
not comparable to an A100 hour; replace these ranges with a measured feasibility run before
committing to a phase.

### 8.3 Hardware decision

**Decision (2026-10-24): Android phone first, using `llama.cpp`.** This supersedes the
earlier “Jetson first” recommendation.

| Profile | Example | Model budget | What it favors |
|---|---|---|---|
| E1: Android phone (primary) | 8–12 GB RAM, recent Snapdragon/Exynos/Tensor, Android 12+ | 0.5B at Q4 first; 1.5B Q4 as a measured stretch | Distillation, quantization, CPU/GPU/NPU comparison |
| E2: Laptop CPU (control) | x86/ARM laptop on AC power | 0.5B–1.5B Q4 | Stable baseline; easy energy measurement |
| E3: Edge GPU (secondary) | Jetson Orin Nano 8 GB, 10–25 W | 1.5B Q4 | Early exit, KV compression |
| E4: MCU | <1 MB RAM, mW | Symbolic solver or tiny model, not 1.5B | A different architecture; not the current LLM path |

**Recommendation:** prove the edge study on Android with the **0.5B distilled student at
Q4** first. Treat the 1.5B model as a measured stretch only if the phone has enough RAM and
the decode speed is acceptable. Use the laptop CPU as the control and measurement anchor.
Do not claim 1.5B mobile support from a desktop result.

**Required input:** exact phone model, SoC, RAM, Android version, backend (CPU / OpenCL /
Hexagon / Vulkan), power mode, latency target, energy target, and batch size. Without these,
the architecture track can measure relative gains but not deployment fitness.

**Available low-end device (user-reported, 2026-10-24):** Redmi 15 4G, Snapdragon 685,
HyperOS 3, **8 GB RAM**, CPU-only path, no Arm dot-product extension. This is a low-end
case study, not a representative “mobile” claim. The 8 GB removes the memory blocker for
1.5B Q4, but not the compute blocker: use the 0.5B Q4 model first; treat 1.5B as a measured
stretch only, not the primary mobile claim.

### 8.4 Android-first runtime plan

The runtime ladder is:

1. **Desktop CPU reference:** `llama.cpp` GGUF, fixed flags, AC power. This is the
   reproducibility anchor.
2. **Android CPU:** `llama.cpp` `arm64-v8a` build; NEON/KleidiAI; fixed thread count on
   performance cores. This is the primary claim.
3. **Android GPU/NPU:** OpenCL (Adreno), Hexagon (Snapdragon), or Vulkan where the device
   and build support it. This is a comparison, not an automatic improvement.
4. **Custom runtime:** only for architecture changes stock `llama.cpp` does not expose
   (for example, early exit). A fork is a research artifact, not a deployment baseline.

Build paths:

- Android Studio binding: `examples/llama.android` in the pinned `llama.cpp` tree.
- CLI: Termux or Android NDK cross-compile, `arm64-v8a`, then `adb push` the binary and
  model.
- Pin the `llama.cpp` commit; record the backend, build flags, thread count, context size,
  and model hash.
- On the Redmi 15 / Snapdragon 685, build CPU-only with `GGML_NATIVE=OFF` and no
  `+dotprod`/KleidiAI path; verify the reported CPU features before trusting the binary.
  Pin four worker threads to the Cortex-A73 cluster; do not use all eight cores.

**Existing harness:** [`gyawalisanish0/DomainAI`](https://github.com/gyawalisanish0/DomainAI)
(Apache-2.0) is the deployment base. Do not build a second Android app. Reference snapshot:
`main` at `dd0eff1094cfdd3835a79775cc11e09975852eaf` (2026-06-26), vendored `llama.cpp`
`f3e1828` (cloned 2026-06-18). Pin a fork/commit before measuring.

It already provides Kotlin/Compose + C++/JNI, GGUF import, a model manager, CPU/GPU backend
diagnostics, thread pinning, prefill/decode timing, and an in-app CPU-vs-GPU benchmark. It
is **not yet a research instrument**. Required changes:

1. Deterministic sampler profile (greedy/temperature 0) or a fixed, recorded seed; log
   top-k, top-p, temperature, and seed. The current native sampler is top-k 40, top-p
   0.95, temperature 0.7, default seed.
2. Stop-string support for the terminal symbolic marker in addition to EOS; record the
   stop reason and include emitted stop tokens in the token accounting. The current loop
   stops only on EOG or context length.
3. Exact training-template mode: bypass the app/embedded chat prompt or override it with
   the frozen v3 prompt. Hash the fully rendered prompt and log it. The current app uses
   the GGUF's embedded chat template; v3 training used manual Qwen formatting.
4. Headless benchmark runner (debug activity, instrumentation, or ADB-triggered service)
   that writes one JSONL row per cell: model/prompt hashes, backend, threads, context,
   sampler, token counts, prefill/TTFT/decode/total ms, peak RSS, battery/thermal state,
   stop reason, and raw output.
5. GGUF tokenizer/format gate for the 47 added tokens before any phone run.
6. Real arm64 phone execution; the current CI emulator is x86_64 and cannot load the
   native library.

Model path:

- Start with the 0.5B student in GGUF `Q4_K_M`; test `Q5_K_M` or `Q8_0` if accuracy drops.
- A stock 0.5B is a **baseline**, not the distilled student. Label it accordingly and do
  not present it as a teacher–student result.
- Do not substitute the catalog's DeepSeek-R1-Distill-Qwen-1.5B for the r6 base; record the
  exact base revision and the merged adapter hash.
- Run a **GGUF tokenizer/format gate** before any mobile benchmark: the added symbolic
  markers must tokenize the same way, the chat template and stop tokens must match, and the
  strict accuracy/format-valid rates must pass on the desktop with the GGUF model.
- Do not assume the FP16 adapter result transfers to Q4. Quantization is a new artifact and
  needs its own accuracy receipt.
- Test the 1.5B GGUF only after the 0.5B path is clean.

Android benchmark protocol:

- One specified phone, airplane mode, fixed brightness, unplugged, batch size 1.
- On HyperOS, use a foreground service and exempt the benchmark from battery optimization;
  disable memory extension and adaptive battery; record the thermal status. Otherwise the
  OS can kill or throttle the run and the latency receipt is invalid.
- Record prefill tokens/s, decode tokens/s, time-to-first-token, p50/p95 end-to-end
  latency, peak RSS, and battery/energy delta where measurable.
- Warm up, then run at least five measured repetitions per prompt; report variance and
  thermal state.
- Compare symbolic vs natural-language CoT vs answer-only, and CPU vs GPU/NPU if available.
- Label a CPU-only result as CPU-only; do not generalize it to “mobile.”

Mobile feasibility gates (engineering, not scientific claims):

- **0.5B gate:** complete a five-prompt timing run, then the 50-prompt stratified subset
  within the session budget. Record free RAM and peak RSS. If it OOMs or throttles into
  unusability, stop and report the device boundary.
- **1.5B gate:** only after the 0.5B path is clean. Run five prompts first; if the
  50-prompt subset would exceed the session budget, or thermal throttling is severe, stop.
  “Memory-feasible” is not “usable.”
- Disable HyperOS memory extension; virtual RAM can mask OOM and distort peak-RSS receipts.
- Report burst and sustained numbers separately; a short fast run is not a sustained claim.

Architecture caveat:

- Stock `llama.cpp` already supports quantization, speculative decoding, and several
  backends. It does **not** give you a clean early-exit or custom-KV interface.
- Early-exit and custom-state experiments need a PyTorch prototype first, then a pinned
  `llama.cpp` fork if the result justifies the engineering.
- Do not promise architecture-level results from a stock runtime benchmark.

### 8.5 Funding and collaboration

- Use the honest v3 result, positive or negative, as preliminary data for cloud credits or
  edge-hardware grants.
- Ask for A100/H100 credits for 7B/14B confirmation; do not promise results that the free
  quota cannot support.
- Recruit an independent auditor for the trace sample and the energy protocol.
- If pursuing the human-cognition question, collaborate with cognitive scientists and
  preregister a separate study; do not fold it into the LLM efficiency claim.

---

## 9. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Diagnostic fails format/EOS | No v3 result | Diagnose pipeline; do not weaken gate or relabel the diagnostic |
| 1.5B C-vs-D null | Thesis weakened | Publish negative; pivot to tool use/architecture |
| 7B ignores the format | Scaling branch stops | Zero-shot probe first; matched SFT; report boundary |
| 7B effect vanishes | No scaling claim | Report boundary; focus on edge/teacher |
| Teacher traces are wrong | Student learns errors | Verify with solver/strict scorer; filter; hold out |
| Edge energy unmeasurable | No J/correct claim | Report tokens/latency only; no estimated joules |
| Architecture change confounds | Uninterpretable gain | One change at a time; matched accuracy; static baseline |
| Quantization breaks format | Invalid outputs | Format gate; QAT; fallback precision |
| App template ≠ training template | Invalid accuracy comparison | Raw-prompt mode; prompt hash; desktop GGUF gate |
| Stochastic app sampler | Noisy or invalid cell comparison | Greedy or fixed seed; record sampler profile |
| Android app measures UI, not model | Misleading latency | Headless runner; batch 1; fixed device state |
| Compute/quota stop | Incomplete run | Label incomplete; resume; never call negative |
| Benchmark overfitting | Inflated result | Held-out external benchmark; unique-prompt sensitivity |
| Token gain ≠ energy gain | Overclaim | Measure on device; report both; no projection |
| Artifact size at 7B | Packaging failure | Save added-row deltas + base revision; package gate |
| Overclaiming under pressure | Credibility loss | Claim ledger; preregistration; retractions preserved |

---

## 10. Publication and artifact plan

### Paper 1 — v3 systems audit (after Phase 1)

A preregistered audit of a symbolic register serving contract at 1.5B: strict accuracy,
tokens/correct, J/correct, and a positive or negative C-vs-D result. The contribution is
the protocol and the honest receipt, not a new SOTA claim.

### Paper 2 — scaling study (after Phase 2)

Does the interface effect survive at 7B? Report the effect at 1.5B and 7B, matched training
controls, seeds, and the boundary if it fails. A well-supported null is a paper.

### Paper 3 — teacher–student edge distillation (after Phase 3)

Verified symbolic traces from a larger teacher; a small student; natural-language and
answer-only controls; on-device tokens, latency, memory, and energy.

### Paper 4 — architecture-level edge efficiency (after Phase 5)

One architecture change against the strongest static baseline, measured on the target
device at matched accuracy. The contribution is the measurement, not a projection.

### Artifacts

- code and pinned runtime lockfiles;
- adapters with hashes and base revisions;
- raw generations and strict scores;
- manifests, commands, and hardware receipts;
- energy/latency scripts;
- private HF repositories first; public release after audit.

Venues are secondary. The claim ladder and receipts are primary.

---

## 11. Immediate next actions

1. ✅ Verify r6; the generation and package gates passed. Receipts:
   `results/r6_closeout/R6_CLOSEOUT.json`.
2. ✅ Run the 38-prompt CPU sweep and record `results/r6_closeout/CPU_SWEEP.json`.
   It is a diagnostic, not a benchmark. The failures are arithmetic/semantic and the
   `verify` blocks are often tautological or false.
3. ⛔ Harden the v3-compatible 100 × 4 × 6 diagnostic harness. The existing
   `eval/robust_eval.py` is the v2.1 path and must not be relabeled as v3. It must fail
   closed on generation errors, corrupted rows, adapter-hash mismatch, and nondeterministic
   subset selection.
4. ⛔ Run the 100 × 4 × 6 diagnostic; if it fails, fix the pipeline and do not run the full grid.
5. ⛔ Request separate explicit approval for the 4,800-cell run.
6. ⛔ Design and preflight r7 Verified-Trace Repair per
   `docs/NEXT_TRAINING_SESSION.md`; do not launch it before the v3 diagnostic path is closed
   and a Phase 1b preregistration exists.
7. Draft the Phase 1b preregistration for the matched no-register adapter and seeds.
8. Check 7B base-model availability, tokenizer compatibility, and QLoRA feasibility.
9. Write a one-page Phase 2 protocol before spending 7B compute.
10. Record the target decision: **Android phone + `llama.cpp`**. The recorded device profile
    is Redmi 15 4G / Snapdragon 685 / HyperOS 3 / 8 GB; build CPU-only without dotprod and pin
    four A73 threads. No phone claim until the desktop GGUF gate passes.
11. Add a GGUF tokenizer/accuracy gate before any mobile benchmark; the 0.5B student is the
    first model, not the 1.5B.
12. Add `docs/ROADMAP.md` to the Markdown audit and the README repository map.
13. Keep the claim ledger and retractions current; no new number without a receipt.

---

## 12. Decision tree

```
r6 clean?
├─ no  → fix artifact; no science; repeat
└─ yes → 100×4×6 diagnostic
         ├─ fail → fix pipeline; no full run
         └─ pass → 4,800-cell C-vs-D (approval)
                  ├─ null → publish negative; pivot to tool use/architecture
                  └─ positive → Phase 1b matched adapter + seeds
                                ├─ null → narrow to serving-contract claim
                                └─ positive → 7B replication
                                               ├─ null → boundary result; edge/teacher focus
                                               └─ positive → teacher–student + 14B/32B
                                                              └─ architecture prototype
                                                                 └─ integration/device
```

The program is allowed to stop. A clean negative or boundary result is a successful
research output if it is measured, preregistered, and reported without spin.

---

## 13. House rules

1. Token efficiency is the primary lens; strict accuracy is the non-inferiority gate.
2. A number without a command, a hash, and a receipt is not a result.
3. A partial, quota-stopped run is not a result in either direction.
4. A projection is not a measurement; a biological metaphor is not a mechanism.
5. Do not call the model “human-like,” “brain-like,” or “non-linguistic” on the basis of a
   token-format experiment.
6. Do not call token savings energy savings without on-device measurement.
7. Do not scale to a larger model to avoid a null result at a smaller one.
8. Do not add a new mechanism until the previous one has a receipt.
9. Expensive runs require separate explicit approval.
10. Preserve retractions; never rewrite a frozen result to make the story cleaner.
