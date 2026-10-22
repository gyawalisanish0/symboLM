# Research Redirect: What Survives, What Dies, and What to Test Next

**Author:** Sanish Gyawali (Rupandehi, Nepal)  
**Date:** 2026-10-22  
**Branch:** `research/rescue-v3`  
**Status:** active rescue. No v3 GPU result exists yet.

---

## 1. The original pitch was larger than the evidence

The early SymboLM story claimed four things at once:

1. A symbolic register language makes a small model reason better.
2. The format sharply compresses reasoning and saves energy.
3. The model reaches strong multi-year benchmark accuracy.
4. The result transfers across model families.

The audit did not support that bundle. It found a broken training format, a lenient scorer, an early-stopped evaluator, a partial GPU run, and a dataset hash mismatch. Some components are real; the headline is not.

The project is therefore being redirected from **“prove SymboLM is a new reasoning paradigm”** to **“test whether a typed-register serving contract is useful at the edge, and report a negative result if it is not.”**

---

## 2. What survives

These are mechanisms or measurements with a clear receipt:

| Finding | Why it survives | Caveat |
|---|---|---|
| Atomic facts can be answered by a short bypass reflex | The strict 1,000-question rescore found 32 correct answers, all atomic bypasses; the atomic subset was 32/40 | This is a narrow factual subset, not general reasoning |
| The old 1,000-question run was mostly truncated | 91.7% of outputs hit the 512-token limit; 823 looped | This is a failure diagnosis, not a capability claim |
| Answer-only is much cheaper than CoT on the SmolLM2 pilot | 8.1 vs 167.6 mean tokens; 20.67× compression | SmolLM2-1.7B, 200 problems, not the flagship |
| The sandbox arm is a positive engineering signal | E 23% vs A 7%; 444 vs 2,052 J/correct | SmolLM2 pilot; it is external arithmetic/tool use, not format value |
| Strict scoring changes the conclusion | The old lenient score was 3.2% strict, not the old headline | The new scorer is the only admissible one for v3 |

## 3. What must die

These claims must not appear in a current abstract, README, grant application, or talk:

- **37.0% Pass@1 overall.** The run stopped at 2,119 / 4,800 cells. The number is a commit message, not a completed result.
- **39.7% reflex accuracy / 822.6 J.** Part of the same incomplete run and the same invalid scoring path.
- **“Format Value Confirmed.”** The Task 2 causal ablation (C 10% vs corrupted-plan 9%, +1 pp) had a confidence interval that included zero. The valuation (C 9.5% vs router-only 10.5%) also failed to establish a format advantage.
- **50.5% on the 1,000-question benchmark.** Legacy lenient extraction; strict rescore is 3.2%.
- **100% MATH500.** n=5 anecdote, not a benchmark.
- **Cross-family LLaMA universality.** n=5 transfer probe; the separate 200-problem SmolLM2 evaluation is not a LLaMA result.
- **Sub-50-token reasoning.** The v2.1 run averaged 494.5 tokens.
- **A “1,955-record complete evaluation.”** The receipts are partial and were scored before the strict scorer existed.

The old README and whitepaper are archived under `docs/` for provenance. They are not current claims.

---

## 4. The redirected thesis

**Route, don’t format: budget-robust reasoning for small models.**

The register language is not assumed to be a reasoning mechanism. It is a **serving contract** with three possible benefits:

1. A short terminal-answer channel that is easy to parse.
2. An explicit place for a router to bypass reasoning on atomic facts.
3. A structure that can be validated and, later, repaired externally.

The question is whether that contract produces more correct answers per joule than the alternatives at the same budget.

### What would count as success

At **256 tokens**, the full register arm (C) must:

- have a paired 95% CI for `J/correct_D − J/correct_C` that strictly excludes zero in the positive direction;
- be non-inferior to the router-only arm (D) on strict accuracy, with a lower 95% CI bound no worse than −2 percentage points;
- emit a terminal answer in at least 90% of cells and pass the format validator in at least 90%.

If those conditions hold, the claim is a **deployment-contract** claim, not a claim that the format makes the model smarter.

### What counts as a negative result

If the CI includes zero, or C is less accurate than D, or the adapter cannot emit terminal answers, the correct conclusion is:

> “The data do not establish format value. Routing-only or external verification is the better explanation.”

That is a legitimate result. It is better than another retracted headline.

---

## 5. Evidence hierarchy

Use this order when deciding what to cite:

### Tier A — admissible for v3 decisions

- `data/held_out_eval_200_v2.jsonl` with SHA-256 `95E2B237AE2B724DC0856AEE7220EBD79436F91630F77BBB3BAD0913127F10ED`.
- `data/research_sft/manifest.json` and its train/validation hashes.
- `eval/strict_scorer.py` and the strict report produced by `eval/robust_eval.py`.
- `run_manifest.json` from the v3 SFT run.
- The preregistered 4,800-cell run, or the explicit statement that it is incomplete.

### Tier B — measured but not comparable

- The SmolLM2-1.7B 200-problem ablation and valuation (A 7%, B 15%, C 10%, D 9%; E 23%).
- The old v2.1 strict rescore (3.2%, 91.7% truncation, 823 loops).
- The partial six-arm run’s strict common-57 at 2048 (C 26.3%, D 1.8%, F 42.1%, A 0%, B 29.8%, E 1.8%; n=57, post-hoc).

These can motivate a hypothesis. They cannot be pooled with the v3 run.

### Tier C — historical or retracted

- The v2.1 README/whitepaper headline numbers.
- The old 37.0/39.7/50.5/100% claims.
- Any score produced by `deduce_equality_rhs`, trailing-number regexes, or an Arm C stop at `</reg:verify>`.

Tier C is preserved for audit, not for promotion.

---

## 6. The experiment that can settle it

The new protocol is [`PREREG_VALUATION_V2.md`](../PREREG_VALUATION_V2.md). In one sentence:

> One adapter, six inference contracts, four budgets, 200 held-out problems, strict declared-answer scoring, and a single primary C-vs-D test at 256 tokens.

The design is intentionally narrow. It does not try to prove general reasoning or SOTA. It asks whether one serving contract is worth its overhead.

### Why six arms

- **A:** ordinary CoT, the default.
- **B:** answer-only, the token-efficiency floor.
- **C:** full registers, the proposed contract.
- **D:** router-only, the format-free control.
- **E:** CoT plus a Python sandbox, the external-verification alternative.
- **F:** untouched base model, the adapter sanity check.

### Why one adapter

Training separate adapters for every arm would be a stronger causal design, but it is a larger experiment. This run tests the deployment question: **given the adapter we can train once, how should it be served?** A matched no-format adapter is the next experiment if C shows a signal.

### Why strict scoring

The old harness could call an answer correct without `<ans>`, and it could stop Arm C before the answer channel. That is not a measurement of the format; it is a measurement of the parser’s optimism. V3 uses declared answers and EOS-aware truncation.

---

## 7. Kill criteria and decision rule

The rescue stops or narrows if:

- The fresh adapter cannot produce terminal answers in the generation check.
- More than 10% of Arm C cells are format-invalid.
- C does not beat D on the primary energy endpoint at 256.
- C loses more than 2 percentage points of strict accuracy versus D.
- The unique-prompt or duplicate-cluster sensitivity analysis reverses the result.
- E dominates C on accuracy and energy, making the register layer unnecessary.
- The adapter is worse than the untouched base model F at the same budget.

If any of these happen, the public claim becomes a **negative systems result**. The repository stays, the receipts stay, and the retraction stays.

---

## 8. What would make this a real paper

Not another bigger claim. Five things:

1. **Matched training.** At least two adapters trained on matched data: register and no-register. Then C vs D is a training-data comparison, not only a prompt comparison.
2. **Multiple seeds.** At least three SFT seeds; report variance, not one lucky run.
3. **External benchmark.** MATH500 or a comparable public benchmark, evaluated with the same strict scorer, not a custom template-only set.
4. **Energy on two machines.** At least one NVIDIA GPU with NVML and one CPU/edge device, with the same model and prompts.
5. **Human or independent audit.** A reviewer checks a random sample of traces against ground truth, especially the 173 unique prompts.

Until then, the project is an audit plus a small preregistered systems experiment. That is a legitimate scope.

---

## 9. Immediate checklist

- [x] Replace fake BOS/EOS with real special tokens.
- [x] Build SFT targets manually; stop using the Qwen chat template for training.
- [x] Remove dialogue rows and wrong contrastive branches.
- [x] Add prompt-prefix and label-mask assertions.
- [x] Build a canonical 4,600-target split with a hash manifest.
- [x] Correct `MY2021_001` to 169.5.
- [x] Add duplicate-prompt flags to the benchmark.
- [x] Add strict declared-answer scoring and EOS-aware truncation.
- [x] Stop Arm C only at `<ans>`/bypass.
- [x] Add run manifests and a post-training generation check.
- [x] Pass the local test suite (55 tests).
- [x] Freeze `PREREG_VALUATION_V2.md`.
- [x] Commit the rescue branch and tag it `sft-v3-20261022-r2`.
- [x] Prepare the Kaggle SFT kernel at `experiments/kaggle/sft_v3/`.
- [ ] Run fresh SFT on Kaggle (kernel prepared, not launched).
- [ ] Run the 100 × 4 diagnostic.
- [ ] Run the full 200 × 4 × 6 experiment.
- [ ] Update README/results with receipts, or publish the negative result.

---

## 10. House rules for claims

1. A number without a command, a hash, and a receipt is not a result.
2. Strict and lenient scores may both be shown, but only strict scores decide.
3. Never call a run complete when it stopped early.
4. Never use “confirmed” for a confidence interval that includes zero.
5. Never compare an adapter arm to a separately trained baseline and call it a controlled ablation.
6. Preserve old artifacts; label them historical rather than deleting them.
7. A negative result is allowed to be the final answer.
