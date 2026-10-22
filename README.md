# SymboLM v3 — Route, Don't Format

**A scientific rescue of the SymboLM project.**  
**Author / Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**Repository branch:** `research/rescue-v3`  
**Status date:** 2026-10-22  
**Current phase:** clean SFT + strict evaluation pipeline ready; GPU re-run not yet executed.  
**SFT kernel:** [`experiments/kaggle/sft_v3/`](experiments/kaggle/sft_v3/) (tag `sft-v3-20261022`); prepared but not launched.  
**Model repositories:** all four Hugging Face model/checkpoint repositories were made private on 2026-10-22. Public links now return 401; nothing was deleted.

---

> **Read this first.** The v2.1 public headline — *“37.0% / Format Value Confirmed”* — is
> **retracted**. The six-arm run stopped at 2,119 of 4,800 cells; “37.0%” appeared in a
> commit message, not in a completed preregistered result. The Task 2 causal ablation
> found **no format value over a corrupted-plan control**, and the router-only
> valuation also failed to establish a format advantage. The old README and whitepaper are preserved as
> historical documents in [`docs/ARCHIVE_README_2.1.md`](docs/ARCHIVE_README_2.1.md)
> and [`docs/ARCHIVE_WHITEPAPER_2.1.md`](docs/ARCHIVE_WHITEPAPER_2.1.md).

## 1. The thesis we can actually test

**Route, don’t format: budget-robust reasoning for small models.**

The scientific question is not “does a pretty symbolic format make a small model
smart?” The question is:

> Under a fixed token budget, does a **typed register interface plus a router**
> produce more correct answers per joule than ordinary chain-of-thought, answer-only,
> or a router-only control?

The register format is a **control interface**, not a claimed source of reasoning
ability. The possible value is that it gives the model a short, checkable execution
contract and gives the serving system an early-exit signal.

## 2. What survives the audit

| Claim | Status | Evidence |
|---|---|---|
| Atomic facts can be answered by a short bypass reflex | **Supported as a mechanism** | 32/1,000 strict v2.1 passes were atomic bypass answers; the atomic subset was 32/40 |
| The full register format adds value over a router-only control | **Not established** | Valuation: C 9.5% vs D 10.5%; paired 95% CI includes zero. Task 2: C 10% vs corrupted-plan 9%, +1.0 pp; CI includes zero |
| The sandbox arm is a positive engineering result | **Supported in the SmolLM2 pilot** | E 23% vs A 7%; 444 vs 2,052 J per correct answer |
| The v2.1 model is a strong math reasoner | **Retracted** | Strict rescore of the 1,000-question run: 3.2% (32/1,000), 91.7% truncated, 823 looped traces |
| 100% MATH500 / cross-family transfer | **Retracted** | Both were n=5 anecdotes, not benchmarks |
| 37.0% overall accuracy | **Retracted** | Run stopped at 2,119/4,800 (44%); 37.0% is a git commit message |
| “Format Value Confirmed” | **Retracted** | The paired CI includes zero; see [`eval/ABLATION_REAL_REPORT.md`](eval/ABLATION_REAL_REPORT.md) |
| Sub-50-token reasoning | **Not supported** | The v2.1 run averaged 494.5 tokens, not 253.8 |

The honest exploratory signal from the partial six-arm run is narrower: on the 57
cells common to all arms at budget 2048, strict declared-answer accuracy was
C 26.3%, D 1.8%, F 42.1%, A 0%, B 29.8%, E 1.8%. That is **n=57, post-hoc, and not
the preregistered 200 × 4 experiment**. It motivates the re-run; it does not replace it.

## 3. What was broken

The audit found five independent failures, each of which could produce a false positive:

1. **Fake BOS/EOS.** Training wrapped targets in literal
   `<｜begin of sentence｜>` / `<｜end of sentence｜>` strings. Those are **seven ordinary
   tokens**, not the tokenizer’s special tokens. The model never learned the real EOS.
2. **Chat-template stripping.** The published Qwen chat template splits an assistant
   message at `</think>` and discards everything before it. Using
   `apply_chat_template` for SFT would train on the answer only, not the reasoning trace.
3. **Early Arm C stop.** The evaluator stopped Arm C at `</reg:verify>`, so it never
   emitted `<ans>`. The old scorer then harvested the last equality inside
   `<reg:deduce>` and called it correct.
4. **Lenient scoring.** `deduce_equality_rhs` and trailing-number fallbacks turned
   truncated loops into “correct” answers. C’s 296 old “correct” rows had **zero**
   `<ans>` declarations and 52 were truncated.
5. **Stale or partial runs.** A cancelled Kaggle run, a stale checkpoint, and a dataset
   hash mismatch (`575B…` vs the preregistered `8369…`) were reported as completed.

## 4. The repaired pipeline

### Data

- `scripts/prepare_research_sft.py` builds the canonical split from
  `data/concept_math_contrastive_v2_1.jsonl`.
- **4,600 targets:** 4,200 canonical reasoning traces + 400 atomic bypass traces.
- Contrastive-negative rows are converted to their **final corrected branch**, never
  trained on the deliberately wrong branch.
- 400 dialogue/intent rows are excluded: they are outside the math/edge thesis.
- 350 legitimate duplicate source IDs are preserved with unique training IDs.
- Split: **4,140 train / 460 validation**, seed 42, stratified by target kind.
- Manifest and hashes: `data/research_sft/manifest.json`.

### Training

- `data/symbolic_dataset.py` builds the prompt and target manually with the tokenizer’s
  real special tokens:
  `<BOS><system><|User|><prompt><|Assistant|><think>\n<target><EOS>`.
- The prompt is asserted to be a token prefix of the full string; labels mask exactly
  the prompt; every target must end in the real EOS.
- `training/sft_train.py` no longer auto-resumes stale checkpoints. Auto-resume is
  opt-in. It writes `run_manifest.json` with dataset hashes, counts, package versions,
  and post-training generation checks.

### Evaluation

- `eval/strict_scorer.py` accepts only `<ans>…</ans>` and
  `<reg:bypass>…</reg:bypass>` for the register arm. Baseline arms may use a closed
  `\boxed{}` because their prompt requests it.
- Truncation without EOS = **fail**. No trailing-number fallback. No
  `deduce_equality_rhs` fallback in the primary score.
- `eval/robust_eval.py` stops Arm C only at a declared answer. It records full traces,
  EOS, format validity, strict and lenient scores, and the dataset hash preflight.
- The benchmark is `data/held_out_eval_200_v2.jsonl` (SHA-256
  `95E2B237AE2B724DC0856AEE7220EBD79436F91630F77BBB3BAD0913127F10ED`).
- One documented correction: `MY2021_001` is **169.5**, not 169.
- The v2 file also flags 20 duplicate-prompt groups (27 redundant rows); primary
  analysis is n=200, with a unique-prompt sensitivity analysis.

## 5. The next experiment

The new preregistration is [`PREREG_VALUATION_V2.md`](PREREG_VALUATION_V2.md).
The design is a full **200 problems × 4 budgets × 6 arms = 4,800 cells** run with:

- the same 4,600-target SFT split for every register arm;
- identical system prompts in training and evaluation;
- strict declared-answer scoring;
- paired bootstrap on J/correct and a strict accuracy non-inferiority check;
- a pre-specified rule that **no format-value claim** is allowed unless C beats D on
  the paired CI and is not worse by more than 2 pp.

The first run after this rescue should be a 100 × 4 diagnostic to catch formatting or
EOS regressions. That diagnostic is **not** the primary result. The full 200 × 4 run is.

## 6. Quick start

```bash
# 1. Build the canonical SFT split and audit the benchmark
python scripts/prepare_research_sft.py
python scripts/audit_ground_truth.py

# 2. Formatting regression tests (no model download)
python -m unittest discover -s tests

# 3. Train (GPU/TPU; writes run_manifest.json and a generation check)
python training/sft_train.py \
  --train_file data/research_sft/train.jsonl \
  --val_file data/research_sft/val.jsonl \
  --generation_check 2

# 4. Evaluate strictly
python eval/robust_eval.py \
  --input data/held_out_eval_200_v2.jsonl \
  --adapter checkpoints/sft_adapter \
  --expected-sha256 95E2B237AE2B724DC0856AEE7220EBD79436F91630F77BBB3BAD0913127F10ED \
  --arms A B C D E F --budgets 256 512 1024 2048
```

## 7. Repository map

| Path | Purpose |
|---|---|
| `PREREG_VALUATION_V2.md` | Locked protocol for the rescue run |
| `docs/RESEARCH_REDIRECT.md` | What survives, what dies, and why |
| `docs/EVIDENCE_STATUS.md` | Short evidence-status table |
| `EVIDENCE_TRUTH_TABLE.md` | Full claim-by-claim audit |
| `RESULTS_SUMMARY.md` | Audit narrative and strict rescore |
| `eval/strict_scorer.py` | Declared-answer scorer and trace validator |
| `eval/robust_eval.py` | Six-arm, four-budget evaluation harness |
| `data/symbolic_dataset.py` | Manual chat formatting, EOS, label masking |
| `training/sft_train.py` | SFT with run manifest and generation check |
| `scripts/prepare_research_sft.py` | Canonical SFT split builder |
| `scripts/audit_ground_truth.py` | Benchmark v2 audit and override application |
| `experiments/kaggle/sft_v3/` | Kaggle SFT kernel, hash preflight, and packaging gate |
| `symbolm.gbnf` | Corrected continuation grammar for llama.cpp/Ollama |
| `docs/ARCHIVE_README_2.1.md` | Archived old README (historical only) |
| `docs/ARCHIVE_WHITEPAPER_2.1.md` | Archived old whitepaper (historical only) |

## 8. Citation and contact

If you use this work, cite the **v3 research note**, not the retracted v2.1
scorecard. The author is Sanish Gyawali (Rupandehi, Nepal). The repository is
[https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM).

**One rule for collaborators:** do not add a number to this README unless it has a
receipt, a hash, and a command that reproduces it.
