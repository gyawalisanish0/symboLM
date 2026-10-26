# Next Training Session: r7 Verified-Trace Repair

**Status:** design plus a completed dry-run trace audit. Not launched. No r7 artifact or
verified split exists.<br>
**Date:** 2026-10-25<br>
**Branch:** `research/rescue-v3`<br>
**Predecessor:** r6, tag `sft-v3-20261023-r6`, commit `c6d13dd0fbbb83a498f7ef722b9c6afbc70f1255`<br>
**Claim ledger:** [`EVIDENCE_STATUS.md`](EVIDENCE_STATUS.md)<br>
**Frozen protocol:** [`../PREREG_VALUATION_V2.md`](../PREREG_VALUATION_V2.md)

> This is a training plan, not a result. It does not authorize a run.
> r6 remains the frozen serving artifact for the v3 experiment. Do not use r7
> to replace or rescue a null v3 result. Do not launch r7 before the v3
> diagnostic path is closed and a separate Phase 1b preregistration exists.

---

## 1. What the CPU sweep changed

The 38-prompt stratified CPU sweep is recorded in
`results/r6_closeout/CPU_SWEEP.json`. It is a diagnostic, not a benchmark.

| Check | Result |
|---|---:|
| Strict correct | 24 / 38 (63.2%) |
| Format valid | 38 / 38 (100%) |
| EOS emitted | 38 / 38 (100%) |
| Truncated | 0 / 38 |
| Mean generated tokens | 113.9 |
| Mean wall-clock per prompt | 44.1 s |
| Tokens per correct (descriptive) | 180.3 |

The important finding is not the 63.2%. It is the failure shape:

- **Format and EOS are clean.** The adapter learned the register contract.
- **Most errors are arithmetic or semantic, not syntactic.**
- **The `verify` block is often a ritual, not a check.** The model emits
  `verify(ans == 7)✓` when the correct answer is `119`, or
  `verify(ans == 40)✓` after writing `c²=4624`. The check is tautological or
  false, yet marked successful.
- **The training data may teach this.** `scripts/prepare_research_sft.py`
  keeps only the final corrected branch of contrastive traces and discards the
  failed branch and backtrack. A model trained only on successful canonical
  traces never sees a check fail.
- **Some benchmark prompts are ambiguous.** `MY2024_064` and `MY2024_065`
  are Diophantine problems with multiple valid positive-integer solutions. On
  `MY2024_065`, the model output 21 is a valid x + y, but the ground truth is
  7. On `MY2024_064`, the model output 25 is a valid x, not the requested
  x + y. The first is a benchmark-audit issue; the second is both ambiguity
  and a model error. Do not treat either as a clean model failure.

The next training session should therefore target **verification validity and
arithmetic reliability**, not add more register syntax.

## 2. r7 objective

Train one 1.5B LoRA adapter, r7, on the same source split and hyperparameters as r6,
minus rejected rows, but on **verified traces** where:

1. the terminal answer is correct;
2. the `verify` block is a real recomputation or invariant check, not the
   tautology `verify(ans == <the answer just written>)✓`;
3. a failed check is allowed to trigger a correction/backtrack, rather than
   being hidden;
4. no unverified trace enters training.

r7 is a repair run, not a new format. It keeps the same v3 register contract.

## 3. Data audit and repair

The audit script is `scripts/audit_training_traces.py`. With `--write` it would produce a
repaired split; the default dry run does not:

```text
data/research_sft_verified/
  train.jsonl
  val.jsonl
  manifest.json
  rejected.jsonl
```

The audit must run over all 4,140 train and 460 validation rows. The manifest
must record the source hashes, source counts, retained/rejected counts by kind,
and audit coverage. Rejected rows go to `rejected.jsonl` with a reason; they are
never silently dropped. The retained split may be smaller than the source split,
but it must never be expanded.

### 3.0 Dry-run audit receipt (2026-10-25)

`scripts/audit_training_traces.py` now audits all 4,140 train / 460 validation rows. The
default is a dry run: it writes `results/r7_verified/AUDIT_DRY_RUN.json` and **does not
write a verified split**. The `--write` path has not been run.

| Split | Audited | Verified reasoning | Bypass contract | Answer-checked-only | Rejected |
|---|---:|---:|---:|---:|---:|
| Train | 4,140 | 2,561 | 360 | 1,187 | 32 |
| Validation | 460 | 303 | 40 | 115 | 2 |
| **Total** | **4,600** | **2,864** | **400** | **1,302** | **34** |

- The 2,864 verified reasoning rows are 982 original-valid rows plus 1,985 repaired rows.
  Reasoning verification validity is 0.2338 before repair and 0.6819 after repair.
  Of the 982 original-valid rows, 103 were removed because their verify arithmetic could
  not be parsed or checked; the final `valid` status count is 879, not 982.
- The 1,302 answer-checked-only rows have the verify block removed and are excluded from
  the verified train/val split by default. They are not called verified.
- The 34 rejected rows are all arithmetic mismatches, mostly successive-discount rows
  where exact arithmetic yields a decimal while the terminal answer is rounded. They are
  recorded in the audit report, not silently dropped.
- The 400 bypass rows are retained only as `verified_bypass_contract`; they are excluded
  from the reasoning verification-validity rate.
- 184 rows carry a heuristic positive-integer ambiguity flag. These are review flags, not
  automatic rejections.
- This receipt is a data-quality audit, not a training run, benchmark, accuracy claim, or
  token-efficiency claim.

### 3.1 Checks

For every row:

- Parse exactly one terminal `<ans>...</ans>` or `<reg:bypass>...</reg:bypass>`.
- Extract the final answer and compare it to `ground_truth` with
  `eval.strict_scorer.verify_answer`.
- Parse `<reg:deduce>` and `<reg:verify>`.
- Flag **tautological verification**: `verify(ans == X)` where `X` is the same
  answer declared in `<ans>`, with no independent recomputation.
- Flag **false verification**: a check expression that evaluates false but is
  marked `✓`.
- Flag **missing derivation**: the final answer is not the result of the last
  equality or stated reduction in `<reg:deduce>`.
- Flag **arithmetic mismatch**: where the trace is a simple arithmetic
  expression, evaluate it with SymPy and compare.
- Flag **ambiguous problem**: multiple valid solutions or underdetermined
  wording. This is a benchmark/data flag, not an automatic rejection.
- Flag **format/loop/truncation** issues with the existing validator.

The audit cannot prove every olympiad-style derivation. It must be honest about
coverage: report how many rows were fully checked, partially checked, or only
answer-checked. Do not call an unchecked row “verified.”

### 3.2 Repair policy

- If the final answer is wrong: reject the row. Do not train on it.
- If the verify block is tautological or false: replace it with a real check
  when a deterministic check exists; otherwise remove the verify block.
- If the derivation is unverifiable but the answer is correct: keep the row only
  in a clearly labeled `answer_checked_only` subset with the verify block
  removed; do not use it for the verification-validity metric.
- If the problem is ambiguous: keep it for training only if the target answer is
  one of the valid solutions, and flag it. Exclude it from strict-accuracy
  claims.
- Never rewrite a target answer to match the model or the benchmark after seeing
  evaluation outcomes.

### 3.3 Contrastive verification data

The source corpus contains failed-branch/backtrack traces that
`prepare_research_sft.py` currently discards. A small, verified contrastive
subset can teach the model that `verify` can fail:

```text
<reg:plan>...</reg:plan>
<reg:deduce>wrong branch ...</reg:deduce>
<reg:verify>verify(...)✗</reg:verify>
<reg:deduce>replan: corrected branch ...</reg:deduce>
<reg:verify>verify(...)✓</reg:verify>
</think>
<ans>correct answer</ans>
```

Rules:

- only include a contrastive row if the final corrected answer passes the strict
  scorer and the failed check is genuinely false;
- cap the contrastive fraction at 10–15% of training rows;
- keep a canonical-only r7a first; add contrastive r7b only after r7a passes its
  gates;
- do not let contrastive traces increase the primary token budget without a
  separate analysis.

## 4. Training configuration

Use a new kernel directory, not the frozen r6 kernel:

```text
experiments/kaggle/sft_v3_verified/
```

r6's kernel validates register targets and calls a register-only generation
gate. Reusing it unchanged would not test verification validity.

Training command, once the verified split exists:

```bash
python -m training.sft_train \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --base_revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --train_file data/research_sft_verified/train.jsonl \
  --val_file data/research_sft_verified/val.jsonl \
  --output_dir /kaggle/working/symbolm_v3_r7_verified \
  --adapter_dir /kaggle/working/symbolm_v3_r7_verified/adapter \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 2e-4 \
  --no_quant \
  --generation_check 2 \
  --strict_generation_check \
  --seed 42 \
  --run_manifest /kaggle/working/symbolm_v3_r7_verified/run_manifest.json \
  --no_drive_backup
```

Keep the same LoRA rank/alpha/dropout, target modules, max sequence length, and
token-delta repair as r6. The adapter artifact must be directly comparable.

### 4.1 Additional generation gate

The r6 gate checks terminal answer, format, EOS, and loop markers. r7 adds:

1. **Verification-validity gate:** on fixed probes, the `verify` block must
   contain a non-tautological check that evaluates true and supports the final
   answer.
2. **Corrupted-answer probe:** give the model a prompt whose tempting answer is
   wrong; the trace must not declare the wrong answer with `✓`. It may correct
   itself or fail, but it must not fake a pass.
3. **Arithmetic probe:** a small set of known arithmetic identities must be
   computed correctly in the deduce/verify blocks.
4. **Format/EOS gate:** unchanged from r6.

A failure is an artifact/pipeline event, not a scientific negative result.

## 5. Evaluation

Do not score r7 under the frozen v3 decision rule. r7 is a separate artifact.

Minimum diagnostic comparison:

- same 38-prompt CPU sweep, frozen selection and prompts;
- same 200-problem benchmark later, with the v3 strict scorer;
- r6 vs r7 on strict accuracy, format validity, EOS, truncation, tokens, and
  **verification validity rate**;
- verification validity is a new secondary metric: fraction of traces whose
  verify block is non-tautological, evaluates true, and agrees with the terminal
  answer;
- paired comparison over the same prompts;
- no accuracy claim from n=38.

The 38-prompt sweep is a screening signal. The 2,400-cell v3 diagnostic remains
the first formal evaluation gate for r6.

## 6. Sequence and gates

1. Finish the CPU sweep and record the receipt. **Done:** 24/38 strict,
   38/38 format, 38/38 EOS, 0 truncation.
2. Audit the 38-prompt failures and the 200-row benchmark for ambiguity.
   **Partially done:** the CPU-sweep failure analysis is in
   `results/r6_closeout/CPU_SWEEP.json`; the trace audit flags 184 rows for
   positive-integer ambiguity review; benchmark rows `MY2024_064` and
   `MY2024_065` remain flagged and unchanged.
3. Harden the harness and run the 2,400-cell v3 diagnostic on r6. **Harness done:**
   `eval/run_contract.py` and `eval/robust_eval.py` pass 108 model-free tests.
   **Diagnostic not run.** r6 remains the only v3 serving artifact.
4. If the diagnostic passes, request separate explicit approval for the
   4,800-cell v3 run.
5. Freeze a Phase 1b preregistration for r7 **before** its first scored cell.
6. Build and preflight the verified split. **Dry-run audit done**; the verified
   split has not been written. Any wrong answer, false check, or unverified row
   in the training set stops the run.
7. Train r7a canonical-only seed 42. Pass the generation and package gates.
8. Run the 38-prompt and a 100-prompt diagnostic. If verification validity does
   not improve, do not add contrastive data; diagnose first.
9. If r7a is clean, train r7b with the capped verified-contrastive subset.
10. Only after r7 is stable, run the matched no-register control (Section 9).

## 7. Stop conditions

Stop and fix the pipeline if any of these occurs:

- a training target has a wrong terminal answer;
- a verify block is false or tautological and was not repaired;
- a row is unverified but labeled verified;
- register tags, `<ans>`, or loop markers appear in the wrong place;
- target truncates before EOS at `max_length=1024`;
- prompt is not a token prefix of the full sequence;
- train/validation ID overlap;
- generation gate lacks a terminal answer or real EOS;
- verification-validity gate fails;
- adapter package contains a `checkpoint-*` entry;
- `torchao` is present or the manifest does not record `"torchao": "absent"`;
- package or adapter hash mismatch;
- loss is non-finite or the run exits nonzero.

## 8. Compute estimate

r6 took 17,544 seconds (about 4.9 hours) for 777 steps on one T4. r7 starts from
the same 4,140/460 split; rejected rows can only reduce it, so expect a similar or
slightly shorter run. The audit and
contrastive filtering add CPU time, not GPU time. Three seeds plus evaluation is
a multi-day Kaggle-quota project. These are planning estimates, not receipts.

## 9. Follow-on: matched no-register control

The causal question remains: did register training data cause the effect, or is
it only the serving contract? After r7 is stable, train a matched no-register
adapter on the **same verified problems**, with the same base, tokenizer, split,
seed, and hyperparameters, using register-stripped verbalized targets ending in
`\boxed{...}`.

Evaluate the 2 × 2 design:

| | Register serving (C) | No-register serving (A/D) |
|---|---|---|
| r7 register-trained | r7-C | r7-A/D |
| control | control-C | control-A/D |

Primary causal quantity: difference-in-differences for tokens per correct answer
(and J/correct when NVML is available), with strict accuracy non-inferiority.
A one-seed screening result is not proof. The confirmatory study needs at least
three seeds for both formats.

## 10. What not to claim

Do not claim:

- r7 is a v3 benchmark result;
- r7 proves register training helps or hurts;
- 24/38 is an accuracy estimate for the 200-row benchmark;
- the Diophantine misses prove the model is wrong when the prompt is ambiguous;
- the verification-validity metric is part of the frozen v3 protocol;
- r7 transfers to 7B, 0.5B, GGUF, or Android;
- one seed is causal evidence;
- a matched control is unnecessary.

The allowed screening statement is:

> “A 38-prompt CPU diagnostic found clean format/EOS behavior but frequent
> arithmetic errors and non-causal verification blocks. A verified-trace repair
> run is designed to test whether repairing verification supervision improves
> strict accuracy without changing the serving contract. No v3 accuracy,
> token-efficiency, or mobile claim is made from this diagnostic.”

## 11. Receipts to produce

Produced dry-run receipt:

- `results/r7_verified/AUDIT_DRY_RUN.json` (exists; no verified split written)

Still to produce:

- `data/research_sft_verified/manifest.json`
- `data/research_sft_verified/rejected.jsonl`
- `results/r7_verified/KERNEL_CHECK.json`
- `results/r7_verified/KERNEL_RECEIPT.json`
- `results/r7_verified/run_manifest.json`
- `results/r7_verified/VERIFICATION_GATE.json`
- adapter SHA-256 and package SHA-256
- raw r7 generations
- the Phase 1b preregistration
- the crossed evaluation receipt, if and only if it is run

Until those exist, r7 is a plan.
