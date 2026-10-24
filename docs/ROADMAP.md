# SymboLM Long-Term Roadmap

**Status:** planning document, not a result.
**Date:** 2026-10-24
**Current gate:** r6 adapter training/generation/package. The r6 Kaggle kernel is
`RUNNING`; no admissible v3 adapter exists yet.
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

The tracks meet in two places:

1. **Teacher → student distillation:** a larger model generates and verifies symbolic
   traces; a small edge model is trained on them.
2. **Integration:** a compact interface plus an adaptive architecture may reduce both
   sequence cost and per-token cost. The interaction must be measured, not assumed.

The sequence is deliberately conservative:

1. Close r6.
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

**Status:** untested. r6 is the current gate.

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
| r6 repair | Token deltas for the 47 added rows; tied-row gates | A candidate fix for the r5 artifact bug |
| Eval harness | Strict declared-answer scoring, EOS-aware truncation, energy path | Admissible v3 evaluation if the adapter is clean |
| Old evidence | Retracted headlines, Tier B pilots, incomplete six-arm run | Hypothesis generation only |

### 2.2 What is missing

- A clean r6 adapter that passes the generation and package gates.
- The 100 × 4 diagnostic.
- The full 4,800-cell confirmatory run (separate explicit approval required).
- A matched no-register adapter for a training-data comparison.
- Multiple SFT seeds.
- Thinking-token / answer-token split instrumentation.
- Any 7B, 14B, or 32B run.
- Any teacher–student distillation run.
- Any architecture-level modification or on-device benchmark.
- A target edge device and a latency/energy budget.

### 2.3 Current evidence boundary

No v3 token-efficiency, accuracy, format-value, scaling, or edge claim is allowed yet.
The r6 run is the current gate, not a result. A quota or infrastructure stop is
**incomplete**, never a negative result.

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

- Finish r6; pass the generation gate and the package gate.
- Run the 100 × 4 diagnostic.
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

### Phase 0 — r6 closeout (now, days)

**Question:** Is there a clean 1.5B adapter at all?

**Actions:**

1. Monitor the r6 kernel; verify `run_manifest.json`, generation previews, merged token
   rows, `terminal_marker_rows`, and `token_delta` evidence.
2. If generation fails, do not evaluate. Diagnose, fix in a new revision, and repeat.
3. If generation passes, run the package gate, including `"torchao": "absent"` and distinct
   terminal-marker rows.
4. If packaging passes, run the 100 × 4 diagnostic.
5. Only then request approval for the full 4,800-cell run.

**Gate G0:** clean adapter + package manifest.
**Gate G1:** diagnostic passes format/EOS/terminal-answer checks; no more than 10% Arm C
format-invalid cells.

**Deliverables:** adapter hash, run manifest, generation previews, package manifest,
diagnostic report.

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

**Design:** verified teacher traces; 0.5B and/or 1.5B students; natural-language and
answer-only distillation controls; held-out and external evaluation; edge proxy.

**Gate G4:** student accuracy is non-inferior to the best static student baseline within a
preregistered margin (default −2 pp unless the deployment requires tighter), and
tokens/latency/energy per correct answer are lower on the target device.

**Deliverable:** a distillation report with teacher/student revisions, filter rates, and
raw traces.

### Phase 4 — Architecture prototype (months 5–8)

**Question:** Does one architecture change beat the strongest quantized/distilled baseline
on the target device?

**Design:** C0 bottleneck analysis first; then one of early exit, KV/state compression, or
speculative decoding; matched accuracy; batch 1; multiple runs.

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

The roadmap requires a target device by Gate G2b/G3. Provisional profiles:

| Profile | Example | Model budget | What it favors |
|---|---|---|---|
| E1: Edge GPU | Jetson Orin Nano 8 GB, 10–25 W | 1.5B at 4-bit | Early exit, quantization, KV compression |
| E2: Phone | 8–12 GB RAM, 5–10 W | 0.5B–1.5B at 4-bit | Distillation, quantization, speculative decoding |
| E3: MCU | <1 MB RAM, mW | Symbolic solver or tiny model, not 1.5B | A different architecture; not the current LLM path |

**Recommendation:** start with E1 for the 1.5B proof because it can actually run the model
and measure energy; then test E2 with the distilled 0.5B student. Do not claim E3 support
without a separate feasibility study.

**Required input:** target device, power mode, latency target, energy target, and batch
size. Without these, the architecture track can measure relative gains but not deployment
fitness.

### 8.4 Funding and collaboration

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
| r6 fails generation again | No v3 result | Diagnose artifact; new revision; do not weaken gate |
| 1.5B C-vs-D null | Thesis weakened | Publish negative; pivot to tool use/architecture |
| 7B ignores the format | Scaling branch stops | Zero-shot probe first; matched SFT; report boundary |
| 7B effect vanishes | No scaling claim | Report boundary; focus on edge/teacher |
| Teacher traces are wrong | Student learns errors | Verify with solver/strict scorer; filter; hold out |
| Edge energy unmeasurable | No J/correct claim | Report tokens/latency only; no estimated joules |
| Architecture change confounds | Uninterpretable gain | One change at a time; matched accuracy; static baseline |
| Quantization breaks format | Invalid outputs | Format gate; QAT; fallback precision |
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

1. Monitor the r6 kernel. Do not evaluate before the generation and package gates pass.
2. If r6 passes, run the 100 × 4 diagnostic; if it fails, fix the artifact and repeat.
3. Request separate explicit approval for the 4,800-cell run.
4. Draft the Phase 1b preregistration for the matched no-register adapter and seeds.
5. Check 7B base-model availability, tokenizer compatibility, and QLoRA feasibility.
6. Write a one-page Phase 2 protocol before spending 7B compute.
7. Choose the target edge device and budget; without it, architecture work is relative, not
   deployable.
8. Add `docs/ROADMAP.md` to the Markdown audit and the README repository map.
9. Keep the claim ledger and retractions current; no new number without a receipt.

---

## 12. Decision tree

```
r6 clean?
├─ no  → fix artifact; no science; repeat
└─ yes → 100×4 diagnostic
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
