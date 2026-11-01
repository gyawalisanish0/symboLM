# Adaptive Compute, Human Analogy, and Compression Limits

**Status:** research notes and hypotheses, 2026-11-01. **Not a result.**<br>
**Scope:** conceptual framing for efficient LLM reasoning; no new benchmark claim.<br>
**Canonical protocol:** [`../PREREG_VALUATION_V2.md`](../PREREG_VALUATION_V2.md)<br>
**Claim ledger:** [`EVIDENCE_STATUS.md`](EVIDENCE_STATUS.md)<br>
**Related framing:** [`TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`](TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md)

> These notes record the conceptual discussion around the v3 diagnostic. They do
> not change the frozen hypotheses, endpoints, budgets, arms, or decision rule.
> They are allowed to guide future preregistrations; they are not evidence for
> the current one.

---

## 1. The short thesis

The best efficiency strategy for an LLM is not "always think in registers," "always
use chain-of-thought," or "make the model think like a human." It is:

> **Spend the least compute, in the cheapest representation, that still passes a
> correctness check, and stop when the check passes.**

This is an allocation problem over representations, not a single prompting trick.
The useful target is **adaptive verified minimal sufficient reasoning**: the
smallest intermediate state that is sufficient for the answer, selected by a
router and guarded by a verifier.

---

## 2. The compute ladder

A practical system chooses the lowest rung that is likely to work:

1. **Do not think.** Cache, retrieve, pattern-match, or answer directly.
2. **Think compactly.** Use a short register/state trace, a latent state, or an
   answer-only path.
3. **Think explicitly.** Use ordinary CoT, a program, or a symbolic derivation.
4. **Think deeply.** Use search, backtracking, multiple samples, or tool loops.
5. **Verify and stop.** Use arithmetic equality, unit tests, a solver, a checker,
   or a second model. Escalate only if verification fails.

The efficient system is a **router plus a verifier**, not a fixed arm. The router
estimates difficulty and chooses a rung; the verifier decides whether the result
is admissible. The cheapest rung that passes wins.

This is task-dependent:

| Task type | Cheapest likely representation |
|---|---|
| Atomic fact / cached answer | Direct answer or bypass |
| Arithmetic / exact computation | Program or symbolic tool, then verify |
| Structured multi-step problem | Compact register trace, then verify |
| Open-ended writing | Direct generation plus a short critique loop |
| Long-horizon agent task | Explicit plan, external memory, tools, checkpoints |
| Mobile / batch-1 deployment | Small model, retrieval, tools, quantization, early exit |

There is no universal "best." There is a Pareto frontier over accuracy, tokens,
energy, latency, and memory for each task and device.

---

## 3. The human-brain analogy

The brain is useful as a **functional inspiration**, not as a literal blueprint.

Useful analogies:

- **Sparsity:** activate only what is needed; do not run the whole network for
  every token.
- **Adaptive effort:** fast, cheap processing for easy cases; slower, deliberate
  processing for hard cases.
- **Working memory:** keep a small, reusable state rather than an unbounded
  transcript.
- **Consolidation:** cache reusable reasoning templates and retrieve them later.
- **Prediction and verification:** continuously check expectations against
  feedback; correct rather than confabulate.
- **Metabolic constraint:** computation has a cost; parsimony is a design
  pressure, not an aesthetic preference.

Important disanalogies:

- The brain is sparse and event-driven; current transformer inference is dense
  matrix multiplication per token.
- The brain is recurrent, multimodal, embodied, and continuously learning; an
  LLM call is a fixed-depth feed-forward pass over a context window with no
  lifelong consolidation.
- Human thought is not a sequence of discrete tokens. LLM "thinking" is
  serialized generation, not biological cognition.
- The "20-watt brain" is a motivating analogy about metabolic efficiency. It is
  not a measurement of this model, and it does not imply that a 1.5B model can
  achieve human-like general reasoning.

**Conclusion:** replicate functional principles if they help; do not claim
mechanistic replication. "Human-like" is a design metaphor, not a benchmark.

---

## 4. How deep can we go before something breaks?

There are four different depths. They fail differently.

### 4.1 Representation depth

How much intermediate state is kept: answer-only -> compact registers -> full CoT
-> code/tool trace.

This is an **information bottleneck**. If the state does not contain enough
information to determine the answer, accuracy must fall. Compressing below the
**minimal sufficient state** is where the concept breaks. The right target is not
the shortest trace; it is the shortest trace that is still sufficient and
verifiable.

### 4.2 Reasoning depth

How many steps the model takes.

More steps help up to a point, then hurt through error accumulation, drift,
looping, and overthinking. The optimal depth is task-specific. The right policy
stops when the answer is stable or verified, not at a fixed budget.

### 4.3 Model depth

Layers, recurrence, or latent computation steps.

More depth can help, with diminishing returns and rising training and deployment
cost. Recurrent or latent depth is promising but harder to interpret and verify.
On mobile, depth is constrained by memory, heat, and battery.

### 4.4 Evaluation depth

How strictly the answer is checked.

Too loose and the system produces false claims; too strict and verification costs
more than reasoning. The useful middle is a cheap, independent check:
arithmetic equality, unit tests, symbolic solvers, retrieval, or a held-out
verifier.

### 4.5 The breaking points

- **Information loss** from over-compression.
- **Error compounding** from over-thinking.
- **Reward hacking** when a length or parsimony reward is gamed.
- **Verification gaps** where the trace looks right but the answer is wrong.
- **Distribution shift** where a compact register format fails on new tasks.
- **Deployment limits** from quantization, memory, thermal throttling, and
  context length.

There is no universal "how many tokens" or "how many layers" answer. The useful
number is measured on the target task and device. On narrow, verifiable tasks,
meaningful token reduction is plausible. On open-ended reasoning, even a 2x
reduction at non-inferior accuracy is hard. Human-brain-level energy efficiency
for general reasoning is far away.

---

## 5. What this means for SymboLM

The strongest framing is not "registers are shorter." It is:

> **What is the minimal sufficient reasoning state, and when should the model use
> it instead of a direct answer, full CoT, or a tool?**

Arm C is one rung on the ladder. Arm D is the router-only control. Arm B is the
reflex rung. Arm E is the tool rung. Arm F is the unadapted base. The interesting
question is not whether C is always best; it is whether a router can choose the
cheapest rung that passes verification.

The [2,400-cell diagnostic](../results/v3_diagnostic_2400/DIAGNOSTIC_STATUS.md) is
a pipeline and format gate, not a benchmark. Its descriptive shape is still
informative for the next design:

- Arm C is format-clean, EOS-clean, short, and budget-invariant.
- It is not accuracy-non-inferior to the stronger arms at higher budgets in this
  subset.
- That is exactly what a too-tight representation bottleneck would look like.
- The response should be an **adaptive hybrid** (compact state for easy or
  structured cases; escalate to CoT or tools when verification fails), not a
  longer register trace or a post-hoc rescue with r7.

r6 remains frozen. The 4,800-cell confirmatory run is a separate decision and has
not been launched. r7 remains blocked until v3 closure, a Phase 1b
preregistration exists, and a verified split is written.

---

## 6. Falsifiable questions for future work

1. Does a compact register state preserve strict accuracy relative to the
   router-only control at a fixed budget? (The frozen primary question.)
2. Does an adaptive router that escalates on verification failure beat every
   fixed arm on tokens or joules per correct answer at non-inferior accuracy?
3. What is the smallest register state that is sufficient for each category?
   Ablate registers one at a time and measure accuracy loss.
4. Does a real verifier (not a tautological `verify` block) reduce false
   positives without increasing cost more than it saves?
5. Does the same policy survive quantization and batch-1 inference on the
   Android/`llama.cpp` target at matched accuracy?
6. Where is the knee of the accuracy-versus-token curve for each task family?

Until those are measured, they are hypotheses.

---

## 7. What not to claim

- That the model thinks like a human.
- That compact registers are universally better than CoT.
- That a diagnostic subset is the confirmatory v3 result.
- That token reduction automatically becomes energy savings.
- That a 20-watt brain analogy is an energy measurement.
- That the current 1.5B model is a sparse, brain-like, or mobile-ready system.
- That r7 can rescue a null v3 result.
