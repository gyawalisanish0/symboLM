# SymboLM v3: Token Efficiency Under a Correctness Constraint

**A research note on biologically inspired control for small reasoning models**

**Author / Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**Version:** v3 research note, 2026-10-23  
**Status:** framing and protocol; **no v3 model result exists**  
**Protocol:** [`PREREG_VALUATION_V2.md`](PREREG_VALUATION_V2.md)  
**Canonical framing:** [`docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`](docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md)

---

## Abstract

SymboLM asks a narrow systems question: **under a fixed correctness constraint, can a small
language model spend fewer generated tokens — and fewer joules — per correct answer by
allocating computation differently?** The proposed control policy uses a short reflex route
for atomic facts and a bounded deliberative route for everything else. The biological
language (“System 1 / System 2,” “working memory,” “metabolic parsimony”) is a design
analogy, not a claim of biological fidelity. The current artifact is a dense 1.5B
transformer with a LoRA adapter, not a sparse mixture of experts or a brain-like module.

The previous v2.1 public headline — 37.0% overall accuracy and “Format Value Confirmed” —
is **retracted**. That six-arm run stopped at 2,119 of 4,800 cells because the free Kaggle
quota/infrastructure limit ended the session; it is an incomplete run, not a completed
result and not a negative result. Its Arm C traces never emitted `<ans>`, and its scoring
path is not the v3 strict scorer. No v3 adapter exists yet. The r3 SFT kernel failed on a
CRLF/LF hash mismatch; the r4 provenance fix (tag `sft-v3-20261023-r4`) passed the full
Kaggle preflight, but training stopped before an optimizer step because Kaggle supplied
`torchao 0.10.0` and PEFT 0.21.2 requires at least `0.16.0`. That is a packaging failure,
not a data or scientific result. The r5 fix (tag `sft-v3-20261023-r5`) removes the unused
package; SFT still must complete before any training result.

---

## 1. The problem: token bloat is the default

Modern reasoning models often spend hundreds or thousands of generated tokens on routine
queries. That cost is real even when the final answer is correct. It is paid in latency,
memory bandwidth, energy, and money. A model that is 2 percentage points more accurate but
five times more expensive per correct answer is not obviously the better edge system.

The old SymboLM documents treated token compression as a secondary benefit of a symbolic
format. This note reverses the priority:

- **Primary:** generated tokens and joules per **correct** answer.
- **Constraint:** strict accuracy must be non-inferior to the control.
- **Mechanism:** does the control policy reduce deliberation, or merely move tokens into the
  answer channel?

The current harness records total generated tokens (`tokens_emitted`). It does **not** yet
log thinking tokens and answer tokens separately. No document may call total tokens
“thinking tokens” until that split is instrumented.

---

## 2. The control policy

The proposed serving contract has two routes.

### 2.1 Reflex route

For an atomic fact, the model emits:

```text
<reg:bypass>answer</reg:bypass><EOS>
```

This is the “System 1” analogy: a short, terminal response without a deliberative trace.
The evidence for it is narrow. In the old strict 1,000-question rescore, 32 of 1,000
answers were atomic bypasses, and the atomic subset was 32/40. That supports a factual
reflex on a small subset, not general reasoning.

### 2.2 Deliberative route

For a non-atomic problem, the model emits a bounded register trace:

```text
<reg:plan>...</reg:plan>
<reg:deduce>...</reg:deduce>
<reg:verify>...</reg:verify>      # optional
</think>
<ans>...</ans><EOS>
```

This is the “System 2” analogy: explicit intermediate state, a terminal answer channel, and
a structural contract that can be validated. It is not a separate neural module. It is a
text contract learned by the same dense transformer.

### 2.3 What is not in the current system

- No sparse expert routing or mixture of experts.
- No CED encoder, shared global KV cache, or Engram module.
- No separate reflex or working-memory network.
- No Tiny Terminal execution in the v3 SFT run. Arm E in the pilot used a Python sandbox;
  the Tiny Terminal is a proposed deterministic tool, not a property of the current adapter.
- No claim that the format is a reasoning mechanism. It is a serving contract.

The 20×, 80×, and 224× figures in historical documents are projections or narrow pilot
ratios. The “4× token compression × 5× sparse routing = 20×” multiplication is invalid as a
compound claim, and the current model has no sparse routing. See
[`docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`](docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md), §3.2.

---

## 3. Training

The v3 run is **LoRA SFT only**:

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, revision
  `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
- Data: 4,600 targets — 4,200 reasoning traces and 400 atomic bypass traces.
- Split: 4,140 train / 460 validation, seed 42, stratified by target kind.
- Manual Qwen chat formatting with real BOS/EOS, exact prompt-prefix assertion, and `−100`
  prompt labels. The stock chat template is not used for training because it can discard
  the reasoning trace.
- Contrastive-negative rows use the final corrected branch, never the deliberately wrong
  branch.
- 400 dialogue/intent rows are excluded from the math/edge thesis.
- No GRPO, no Tiny Terminal execution, no FSM-constrained decoding, and no MoE.

The efficiency-relevant training choices are compact canonical traces, explicit bypass
examples, terminal-answer discipline, and real EOS. Those are design choices, not yet a
demonstrated efficiency effect. A matched no-format adapter and multiple seeds are required
before a causal training claim is possible.

The historical `training/reward.py` contains length-aware and parsimony reward functions,
and older stages used GRPO. They are not part of the v3 primary run. Do not describe the v3
adapter as GRPO-trained or as a product of the parsimony reward.

---

## 4. Evaluation

The preregistered design is **200 held-out problems × 4 budgets × 6 arms = 4,800 cells**:

| Arm | Contract |
|---|---|
| A | Ordinary chain-of-thought |
| B | Answer-only |
| C | Full registers: bypass / plan / deduce / verify / terminal answer |
| D | Router-only control |
| E | CoT plus a Python sandbox |
| F | Untouched base model |

Budgets are 256, 512, 1024, and 2048 tokens. Every cell is scored by
[`eval/strict_scorer.py`](eval/strict_scorer.py):

- A reasoning answer must be declared in `<ans>…</ans>`.
- An atomic answer must be declared in `<reg:bypass>…</reg:bypass>`.
- Baseline arms may use a closed `\boxed{…}` because their prompt requests it.
- Truncation without EOS is a failure.
- No trailing-number fallback, no `deduce_equality_rhs` fallback, and no acceptance of
  `</reg:verify>` as an answer.

The primary test is C versus D at 256 tokens. The win condition is:

1. The paired 95% CI for `J/correct_D − J/correct_C` strictly excludes zero in the positive
   direction.
2. Strict accuracy is non-inferior to D: the lower bound of the paired 95% CI for
   `Acc_C − Acc_D` is at least −2.0 percentage points.
3. Arm C emits a terminal answer and passes the format validator in at least 90% of cells.

If the run stops early because of Kaggle quota or infrastructure, it is reported as
**incomplete**. The missing cells are not treated as a negative result, and the saved cells
are not treated as confirmatory evidence.

The token-efficiency report includes `tokens_per_correct`, `correct_per_1k_tokens`,
`OAA_t`, `AUC_OAA`, `TokenEff`, and the harmonic overthinking score `O`. These are secondary
and cannot rescue a failed primary test. Energy claims require measured NVML data.

---

## 5. What the evidence currently supports

| Finding | Status |
|---|---|
| A short bypass reflex can answer a subset of atomic facts | **Supported as a narrow mechanism** (32/1,000 strict passes; atomic subset 32/40) |
| The old 1,000-question benchmark was mostly truncated | **Supported** (91.7% hit 512 tokens; 823 loops) |
| Strict v2.1 accuracy | **3.2% (32/1,000)**; mean 494.5 tokens |
| Full register format adds value over router-only | **Not established** (SmolLM2 pilot C 9.5% vs D 10.5%; paired CI includes zero) |
| Answer-only is the cheap floor | **Supported in the SmolLM2 pilot** (15.0% at 8.1 tokens and 25.84 J/correct) |
| Tool use is a positive engineering signal | **Supported in the SmolLM2 pilot** (E 23.0%, 444.30 J/correct vs A 7.0%, 2,052.41 J/correct) |
| Old 37.0% / 39.7% / “Format Value Confirmed” | **Retracted**; incomplete six-arm run |
| 100% MATH500 and cross-family universality | **Retracted**; n=5 anecdotes |
| A v3 token-efficiency result | **Does not exist**; no v3 adapter |

The complete claim ledger is in [`docs/EVIDENCE_STATUS.md`](docs/EVIDENCE_STATUS.md). The
historical audit is in [`EVIDENCE_TRUTH_TABLE.md`](EVIDENCE_TRUTH_TABLE.md) and
[`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md).

---

## 6. Current blocker

The r3 tag (`sft-v3-20261022-r3`) is frozen, and its Kaggle SFT kernel is `ERROR`:

```text
Train hash mismatch:
got      BB50D9CBD56B0D2B515D7B922DF10578F587AA60358B0FF87A11821C4D3D94DC
expected 5D05003B495DE3322568C4B40713CCE376A1AC05AC9B85A50B579A68A49FF63E
```

The mismatch was line endings, not data content. The committed blobs were LF; the r3 tag did
not force LF checkout, so Windows materialized CRLF and the manifest recorded CRLF hashes.
The r4 revision (`sft-v3-20261023-r4`) adds `.gitattributes` LF rules, normalizes the canonical
files, regenerates the manifest/preregistration hashes from LF bytes, and updates the kernel
constants. The rows and split are unchanged. No hash check was bypassed.

The remaining blocker is scientific and operational: **no r5 SFT adapter exists yet**. The
r4 preflight passed, but the run stopped at LoRA injection on the Kaggle `torchao` package
conflict; r5 removes that unused package and makes the package gate require it absent. No
4,800-cell evaluation has been launched. It requires a clean adapter that passes the package
gate and separate explicit approval.

---

## 7. Falsification and next steps

The thesis is falsified for practical purposes if:

- Arm C does not satisfy the primary C-versus-D token/energy test at 256 tokens.
- Arm C loses more than 2 percentage points of strict accuracy versus D.
- More than 10% of Arm C cells are format-invalid.
- Arm E dominates C on both accuracy and energy, making the register layer unnecessary.
- A matched no-format adapter explains the result, or multiple seeds show no stable effect.
- The unique-prompt or duplicate-cluster sensitivity analysis reverses the result.

The next steps are:

1. ✅ Apply the CRLF/LF provenance fix and create `sft-v3-20261023-r4`.
2. ✅ Validate the r4 preflight, diagnose the `torchao` packaging stop, and create the r5 fix.
3. Run fresh SFT on r5 and pass the mixed-kind generation check.
4. Run the 100 × 4 diagnostic (not the primary result).
5. Run the full 200 × 4 experiment only with separate explicit approval.
6. Report strict accuracy, tokens per correct, J/correct, format validity, and the
   preregistered decision — including a negative result if that is what the data show.

---

## 8. Citation

If you cite this work, cite the v3 research note and the preregistration, not the retracted
v2.1 scorecard. The old README and whitepaper are preserved in
[`docs/ARCHIVE_README_2.1.md`](docs/ARCHIVE_README_2.1.md) and
[`docs/ARCHIVE_WHITEPAPER_2.1.md`](docs/ARCHIVE_WHITEPAPER_2.1.md) for audit.

**Author:** Sanish Gyawali (Rupandehi, Nepal)  
**Repository:** [https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM)  
**Date:** 2026-10-23
