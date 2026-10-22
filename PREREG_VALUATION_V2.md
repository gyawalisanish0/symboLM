# Pre-Registration V2: Budget-Robust Register Routing

**Study title:** Route, Don't Format — a strict, budget-matched evaluation of a typed-register inference contract for a small reasoning model

**Principal investigator / systems architect:** Sanish Gyawali (Rupandehi, Nepal)  
**Repository branch:** `research/rescue-v3`  
**Protocol date:** 2026-10-22  
**Status:** frozen before the v3 GPU run. Any change after the first scored GPU cell is a protocol deviation and must be reported with the result.  
**Supersedes for execution:** [`PREREG_VALUATION.md`](PREREG_VALUATION.md). The old protocol is preserved as a locked historical document, but its dataset hash and execution record do not match the actual old Kaggle run.

---

## 0. Why this protocol exists

The v2.1 public result — **37.0% Pass@1 and “Format Value Confirmed”** — is not a completed preregistered result. The six-arm run stopped at **2,119 / 4,800 cells**, the old receipts were produced with a lenient extraction path, and the old Kaggle manifest used a held-out hash (`575B…`) that does not match the old preregistration (`8369…`). The old Arm C evaluator also stopped at `</reg:verify>`, so it could “close” a trace without ever emitting `<ans>`.

This protocol does not try to rescue that number. It defines a new, smaller, falsifiable test of the only claim that still matters:

> **Given one small model and one fixed token budget, does serving it with typed registers plus a router improve correct answers per joule over serving the same model with ordinary chain-of-thought, answer-only, or a router-only contract?**

The register format is treated as a **control interface**, not as evidence of hidden reasoning ability. If the format does not beat the control, the project will say so.

---

## 1. Locked artifacts

### 1.1 Training corpus

| Artifact | Value |
|---|---|
| Source file | `data/concept_math_contrastive_v2_1.jsonl` |
| Source SHA-256 | `3F7DC86D5C6A0BF47F1C5E4E90F8F317727319BBE58C7F54C5C416D39EC7BD69` |
| Canonical train file | `data/research_sft/train.jsonl` |
| Train SHA-256 | `5D05003B495DE3322568C4B40713CCE376A1AC05AC9B85A50B579A68A49FF63E` |
| Canonical validation file | `data/research_sft/val.jsonl` |
| Validation SHA-256 | `EB854AFF33DD31210D0B4BE61AF0165E5392F44B7B6D472C8C747161062E3EE3` |
| Train / validation counts | 4,140 / 460 |
| Target kinds | 4,200 reasoning, 400 atomic bypass |
| Excluded | 400 dialogue/intent rows |
| Duplicate handling | 350 legitimate source-ID collisions preserved with unique training IDs |
| Contrastive handling | wrong branch removed; final corrected branch retained |
| Seed / split | 42; 90/10 stratified |

The effective register system prompt is fixed:

```text
You are SymboLM. Solve using typed registers: <reg:plan> for invariants, <reg:deduce> for deductive reduction (conclude with ∴ ans = <value>), <reg:verify> for invariant check, then close with </think> and emit <ans><value></ans> for the final answer. If the problem is an atomic fact, emit <reg:bypass><value></reg:bypass> and stop.
```

The training target must end with the tokenizer's real EOS token. The prompt must be an exact token prefix of the training sequence. Prompt tokens are masked with `-100`. These conditions are asserted by `data/symbolic_dataset.py`, not assumed.

### 1.2 Held-out benchmark

| Artifact | Value |
|---|---|
| File | `data/held_out_eval_200_v2.jsonl` |
| SHA-256 | `95E2B237AE2B724DC0856AEE7220EBD79436F91630F77BBB3BAD0913127F10ED` |
| Rows | 200 |
| Unique prompts | 173 |
| Duplicate-prompt groups | 20 (27 redundant rows) |
| Ground-truth overrides | 1: `MY2021_001 = 169.5` |
| Audit report | `data/held_out_eval_200_v2_audit.md` |
| Old clean hash | `8369BD8CAF919F672D8EC87FFF2295FF6380AE9E451F668F9EEEE908F83CF07B` |
| Old Kaggle manifest hash | reported as `575B…`; it is not the old preregistered hash |

`MY2021_001` is a documented correction, not a post-hoc answer change: `123 × 1.5 = 184.5`, then `184.5 − 15 = 169.5`. The original value `169` silently truncated the half-unit.

The 200 rows are the old clean 200 plus this correction and duplicate-prompt flags. The primary analysis is the full 200-row file because that is the file being preregistered. A **unique-prompt sensitivity analysis** on the 173 distinct prompts is reported alongside it. No duplicate row is removed from the primary analysis after seeing outcomes.

---

## 2. Research questions and hypotheses

### Q1 — Format value under a router-only control

At a fixed budget, does the full register contract (C) produce fewer joules per correct answer than a router-only contract (D), without losing more than 2 percentage points of strict accuracy?

- **H1 (format value):** C has a strictly positive paired 95% CI for `J/correct_D − J/correct_C` and is non-inferior to D on strict accuracy.
- **H0 (routing-only):** The paired CI includes zero, or C is worse than D by more than the non-inferiority margin. The register grammar is not the source of value.

### Q2 — The reflex floor

How much strict accuracy does the cheap reflex retain at 256 tokens?

- **H2:** Answer-only (B) and the atomic-bypass subset of C/D are materially cheaper per correct answer than full CoT (A), even if their accuracy is lower.
- This is a descriptive question. It is not used to rescue H1.

### Q3 — External verification

Does a Python-sandbox prompt (E) improve accuracy per joule over plain CoT (A) and over the register arm (C)?

- **H3:** E improves strict accuracy relative to A and reduces joules per correct answer.
- E is an engineering baseline, not evidence that the register grammar itself is valuable.

### Q4 — Is the adapter viable at all?

If the new adapter cannot emit a terminal answer under the strict contract, the experiment stops. The conclusion is then “pipeline failure,” not “format value” and not “routing value.”

---

## 3. Experimental design

### 3.1 Fixed grid

| Factor | Levels |
|---|---|
| Problems | 200 held-out problems |
| Token budgets | 256, 512, 1024, 2048 |
| Arms | A, B, C, D, E, F |
| Decoding | greedy |
| Repeats | 1 generation per cell |
| Total cells | 200 × 4 × 6 = **4,800** |

There is no optional stopping. A 100-problem × 4-budget diagnostic (2,400 cells) may be run first to catch EOS/format failures, but it is **not** the primary result. If the full 4,800-cell run is not completed, the result is reported as incomplete; the diagnostic is not relabeled as the experiment.

### 3.2 Arms

All non-F arms use the **same SFT adapter** produced by the canonical training run. This is deliberate: the experiment asks how that adapter should be *served*, not whether separately trained models differ.

| Arm | Prompt contract | Final answer channel | Adapter |
|---|---|---|---|
| **A** | Ordinary CoT: solve step by step in English; final answer in `\boxed{}` | `\boxed{}` | yes |
| **B** | Answer-only: no explanation; final answer in `\boxed{}` | `\boxed{}` | yes |
| **C** | Full typed registers: `<reg:plan>`, `<reg:deduce>`, `<reg:verify>`, `</think>`, then `<ans>` or atomic bypass | `<ans>...</ans>` / `<reg:bypass>...</reg:bypass>` | yes |
| **D** | Router-only: atomic facts get the bypass prompt; all other problems get ordinary CoT with no registers | bypass for atomic; `\boxed{}` otherwise | yes |
| **E** | Ordinary CoT plus permission to use small Python blocks for arithmetic | `\boxed{}` | yes |
| **F** | Ordinary CoT with the untouched base model; no adapter | `\boxed{}` | no |

**Known limitation, stated up front:** A, B, D, and E are *prompt controls on a register-trained adapter*, not independently trained no-format models. A difference between C and D therefore measures deployment-contract value under one set of weights. It does **not** isolate the causal effect of training on register data. A matched no-format SFT arm is a follow-up experiment, not part of this run.

### 3.3 Prompt identity

Training and evaluation must use the same user content and the same effective system prompt for Arm C. The old harness prefixed evaluation prompts with `Problem: `; that prefix is removed. Arm C is built with the manual Qwen chat builder in `symbolic/prompts.py`, because the stock chat template strips assistant content before `</think>`.

The exact prompt builder and system prompts live in `symbolic/prompts.py` and are considered part of this preregistration. Changing them requires a protocol deviation.

### 3.4 Training protocol

| Parameter | Value |
|---|---|
| Base model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Method | LoRA SFT (no GRPO in this run) |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | Qwen projection modules (`q,k,v,o,gate,up,down`) |
| Adapter artifact | LoRA weights plus the exact resized `embed_tokens` and `lm_head` rows saved via `modules_to_save`; those rows are frozen, not optimized |
| Epochs | 3 |
| Per-device batch | 2–4, selected to fit one T4 |
| Gradient accumulation | chosen so effective batch = 16 |
| Learning rate | 2e-4 |
| Max sequence length | 1024 |
| Seed | 42 |
| Precision | fp16/bf16 on GPU; no 4-bit required for a 1.5B model |
| Auto-resume | **off**; explicit resume only |
| Generation check | 2 validation prompts; strict terminal-answer and real-EOS check enabled |

The training run must write `run_manifest.json` containing dataset hashes, counts, tokenizer special-token IDs, package versions, git commit, and the generation-check traces. A run without that manifest is not admissible.

### 3.5 Decoding and energy measurement

- One active GPU, batch size 1, greedy decoding.
- `max_new_tokens` = the budget.
- `repetition_penalty = 1.15`.
- The tokenizer's real EOS token is passed explicitly.
- Arm C stopping: stop only at `</ans>` or `</reg:bypass>`. Do **not** stop at `</reg:verify>`.
- Baseline stopping: stop at a closed `\boxed{}` because the baseline prompt asks for it.
- Dynamic energy is measured with NVML at 100 ms polling, minus the measured idle baseline. If NVML is unavailable, the run reports accuracy and tokens only and makes **no energy claim**.
- Wall-clock latency is recorded but is secondary.

### 3.6 Scoring

The primary scorer is `eval/strict_scorer.py`.

1. Arm C accepts only a declared `<ans>…</ans>` or `<reg:bypass>…</reg:bypass>` answer.
2. Arms A, B, D (hard problems), E, and F may use a closed `\boxed{…}` because their prompt requests it.
3. A generation that reaches the budget without observing EOS is **truncated → fail**, even if a partial answer string appears. A generation that emits EOS is not truncated.
4. No trailing-integer fallback. No `deduce_equality_rhs` fallback in the primary score. No last-number guessing.
5. Full generated text, EOS flag, token count, format validity, and both strict and legacy scores are stored for every cell.
6. The legacy lenient score is retained only as a diagnostic and is never used for the decision rule.

A format validator checks that Arm C traces are structurally legal: `<reg:plan>…</reg:plan>`, `<reg:deduce>…</reg:deduce>`, optional `<reg:verify>…</reg:verify>`, `</think>`, and a terminal answer. A bypass trace is legal without the reasoning blocks. Loop markers such as repeated `bands:` are format failures.

---

## 4. Primary and secondary analyses

### 4.1 Primary endpoint (one test)

**Arm C vs Arm D at budget 256**, the edge budget that motivates the project.

- Accuracy: strict declared-answer accuracy.
- Cost: dynamic joules per correct answer.
- Paired bootstrap over problems, **B = 10,000**, seed 42, percentile 95% CI.
- Non-inferiority margin: the lower bound of the paired 95% CI for `Acc_C − Acc_D` must be at least **−2.0 percentage points**.
- Format value is supported only if:
  1. the 95% CI for `J/correct_D − J/correct_C` strictly excludes zero in the positive direction, **and**
  2. the accuracy non-inferiority condition holds, **and**
  3. the Arm C format-valid rate is at least 90% and the terminal-answer rate is at least 90%.

This is a single primary test. No alpha correction is applied to it.

### 4.2 Key secondary endpoints

- C vs D at 512, 1024, and 2048 (reported with CIs; exploratory unless stated).
- C vs A and C vs F at 256 and 2048.
- E vs A at 256 and 2048.
- B vs A at 256 (reflex floor).
- Unique-prompt sensitivity analysis on the 173 distinct prompts.
- Cluster-aware sensitivity analysis that resamples the 20 duplicate-prompt groups rather than individual rows.
- Accuracy, EOS rate, format-valid rate, truncation rate, mean tokens, and J/correct for every arm × budget.
- Generation-length and loop diagnostics.

Secondary results are reported honestly even when they contradict H1.

### 4.3 Decision language

If the primary test passes, the allowed claim is:

> “Under one adapter and the preregistered strict scorer, the full register contract reduced joules per correct answer relative to a router-only contract at 256 tokens without losing more than 2 percentage points of strict accuracy.”

If it does not pass, the required claim is:

> “The data do not establish format value. The result is consistent with routing-only or with an underpowered test.”

No wording such as “confirmed,” “SOTA,” “general reasoning,” or “100%” is permitted unless a separate preregistered test supports it.

---

## 5. Stopping, exclusions, and deviations

1. **Dataset mismatch:** the run stops if the input SHA-256 is not `95E2B237AE2B724DC0856AEE7220EBD79436F91630F77BBB3BAD0913127F10ED`. There is no `--allow-dataset-mismatch` in the primary run.
2. **Format failure:** if more than 10% of Arm C cells are format-invalid, the run is a pipeline failure. The result is not interpreted as evidence about format value.
3. **Infrastructure failure:** a cell may be rerun only for a documented infrastructure failure (OOM, CUDA error, truncated log). The original attempt and the replacement are both retained. No rerun is allowed because the answer was wrong.
4. **No optional stopping:** do not stop after a favorable budget or arm. Complete the 4,800 cells or report the run as incomplete.
5. **Deviations:** any change to prompts, scoring, budget grid, model, dataset, or decision rule must be recorded in `DEVIATIONS.md` with a timestamp and reason. A deviation that changes the primary endpoint invalidates the confirmatory claim.
6. **Energy:** no energy claim on a machine without NVML. No estimated joules substituted for measured joules.

---

## 6. What would falsify the project thesis

The rescue is falsified, or must be narrowed, if any of these occur:

- Arm C cannot emit terminal answers reliably under the strict contract.
- C does not beat D on the primary paired energy endpoint at 256 tokens.
- C loses more than 2 percentage points of strict accuracy versus D.
- The unique-prompt sensitivity analysis reverses the primary result.
- E (sandbox) dominates C on both accuracy and energy, which would make the register format an unnecessary layer.
- The adapter performs worse than the untouched base model F at the same budget.

A negative result is publishable as a systems audit. It is not a failure to be hidden by changing the scorer.

---

## 7. Compute and execution plan

| Stage | Cells / examples | Purpose | Admissible as result? |
|---|---:|---|---|
| Dataset smoke test | 4,600 training rows | Token prefix, labels, real EOS | No |
| SFT | 4,140 × 3 epochs | Produce the adapter | No |
| Generation check | 2 validation prompts | Terminal answer and EOS | No |
| Diagnostic | 100 × 4 × 6 = 2,400 | Format/EOS smoke test | No |
| Full run | 200 × 4 × 6 = 4,800 | Primary analysis | Yes |

Kaggle T4 is sufficient for SFT and evaluation, but the full run is a multi-hour job. Checkpointing and asynchronous persistence are required. If the GPU quota ends before 4,800 cells, the run is labeled incomplete.

---

## 8. Reproducibility commands

```bash
# Build the canonical SFT split and audit the benchmark.
python scripts/prepare_research_sft.py
python scripts/audit_ground_truth.py

# Local, model-free regression suite.
python -m unittest discover -s tests

# SFT (example; use one GPU and the manifest-producing script).
python training/sft_train.py \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --train_file data/research_sft/train.jsonl \
  --val_file data/research_sft/val.jsonl \
  --output_dir checkpoints/sft_v3_adapter \
  --epochs 3 --batch_size 2 --grad_accum 8 --no_quant \
  --generation_check 2 --strict_generation_check

# Strict evaluation.
python eval/robust_eval.py \
  --input data/held_out_eval_200_v2.jsonl \
  --adapter checkpoints/sft_v3_adapter \
  --expected-sha256 95E2B237AE2B724DC0856AEE7220EBD79436F91630F77BBB3BAD0913127F10ED \
  --arms A B C D E F --budgets 256 512 1024 2048
```

The exact Kaggle kernel metadata and commands are in `experiments/kaggle/sft_v3/README.md`. The kernel is prepared on tag `sft-v3-20261022-r2` and has not been launched at the time of this freeze.

---

## 9. Signature

**Sanish Gyawali**  
Principal Systems Architect, SymboLM  
Rupandehi, Nepal  
2026-10-22

**Protocol status:** frozen for the v3 GPU run. The next change to this file must be a dated deviation, not a silent edit.
