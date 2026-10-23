# SymboLM Reproducibility Guide

**Author / Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**Branch:** `research/rescue-v3`  
**Status date:** 2026-10-23  
**Current status:** the v3 adapter does not exist; the r3 Kaggle SFT kernel is `ERROR` on a
CRLF/LF hash preflight. No v3 benchmark result exists.

This guide describes how to reproduce the **pipeline** and the **audit**, not a result that
has not been produced. Every reported number must have a command, a hash, and a receipt.

---

## 1. What is reproducible today

| Item | Command | Status |
|---|---|---|
| Canonical SFT split | `python scripts/prepare_research_sft.py` | Works; outputs 4,140 / 460 |
| Ground-truth audit | `python scripts/audit_ground_truth.py` | Works; applies `MY2021_001 = 169.5` and duplicate flags |
| Model-free tests | `python -m unittest discover -s tests` | Works |
| Strict scorer | `python -m eval.strict_scorer` / unit tests | Works |
| SmolLM2 pilot token/energy table | See [`VALUATION.md`](VALUATION.md) | Tier B historical measurement |
| Old six-arm run | See audit docs | **Incomplete**; not reproducible as a completed result |
| v3 SFT run | `experiments/kaggle/sft_v3/` | **Blocked** by r3 hash mismatch |
| Full 4,800-cell v3 run | `eval/robust_eval.py` | **Not run**; requires a clean adapter and approval |

---

## 2. Environment

The v3 SFT runtime is pinned in
[`experiments/kaggle/sft_v3/requirements.lock.txt`](experiments/kaggle/sft_v3/requirements.lock.txt):

```text
transformers==5.20.1
peft==0.21.2
accelerate==1.15.0
datasets==5.1.0
sympy==1.14.0
huggingface_hub==2.2.0
tokenizers==0.23.3
safetensors==0.8.0
```

PyTorch is supplied by the Kaggle image; its exact version is recorded in
`run_manifest.json` rather than replaced in the lock file.

Base model pin:

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
```

The four Hugging Face model/checkpoint repositories were made private on 2026-10-22. Public
links return 401; nothing was deleted.

### Local setup

```bash
git clone https://github.com/gyawalisanish0/symboLM.git
cd symboLM
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 3. Build the canonical SFT split

```bash
python scripts/prepare_research_sft.py
```

Expected properties:

- Source: `data/concept_math_contrastive_v2_1.jsonl` (5,000 records).
- Kept: **4,600** targets.
- Kinds: **4,200 reasoning + 400 bypass**.
- Split: **4,140 train / 460 validation**, seed 42, stratified.
- Excluded: **400 dialogue/intent rows**.
- Contrastive negatives are canonicalized to the **final corrected branch**.
- Duplicate source IDs are renamed; no records are dropped.

The manifest is `data/research_sft/manifest.json`. It records the source hash, train/validation
hashes, counts, target-kind counts, filters, and seed.

### Important provenance warning

The r3 manifest hashes were computed over **Windows CRLF** files. The committed git blobs are
**LF** and hash differently. On a Linux/Kaggle checkout, the hash preflight fails:

```text
Train hash mismatch:
got      BB50D9CBD56B0D2B515D7B922DF10578F587AA60358B0FF87A11821C4D3D94DC
expected 5D05003B495DE3322568C4B40713CCE376A1AC05AC9B85A50B579A68A49FF63E
```

Validation has the same issue:

```text
got      3FCFCC4141522062087D0F35FDF2C54226D454A1A71FE59BEA26F42636FDEDCD
expected EB854AFF33DD31210D0B4BE61AF0165E5392F44B7B6D472C8C747161062E3EE3
```

Do not bypass this check. The r4 revision will add `.gitattributes` with LF rules, regenerate
the manifest and protocol hashes from LF bytes, update
`experiments/kaggle/sft_v3/run_sft_kernel.py`, and create a new tag. Until then, r3 is not
usable for a Linux training run.

---

## 4. Validate the benchmark

```bash
python scripts/audit_ground_truth.py
```

Expected properties:

- `data/held_out_eval_200_v2.jsonl`: **200 rows**, **173 unique prompts**.
- **20 duplicate-prompt groups**, **27 redundant rows**.
- `MY2021_001` is corrected from **169** to **169.5**.
- Primary analysis uses all 200 rows; the unique-prompt sensitivity analysis is secondary.

Current CRLF working-tree hash:

```text
95E2B237AE2B724DC0856AEE7220EBD79436F91630F77BBB3BAD0913127F10ED
```

LF-normalized hash (r4 target):

```text
F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707
```

The old clean benchmark hash (`8369BD8…`) and the old Kaggle manifest hash (`575B…`) do not
match. That mismatch is part of the audit trail; do not treat the old run as reproducible
against the current file.

---

## 5. Run the model-free tests

```bash
python -m unittest discover -s tests
```

The suite covers:

- strict declared-answer extraction;
- EOS-aware truncation;
- prompt-prefix assertion and `−100` label masking;
- real BOS/EOS token identity;
- contrastive branch correction;
- route-aware format validation;
- duplicate-prompt flags and ground-truth overrides;
- the SFT manifest and hash preflight helper.

The tests do not download the base model and do not prove a model result.

---

## 6. SFT (once r4 exists)

```bash
python training/sft_train.py \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --base_revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --train_file data/research_sft/train.jsonl \
  --val_file data/research_sft/val.jsonl \
  --output_dir checkpoints/sft_v3_trainer \
  --adapter_dir checkpoints/sft_v3_adapter \
  --generation_check 2 --strict_generation_check
```

The run must write `run_manifest.json` and pass the mixed-kind generation check:

- one reasoning prompt: valid `<reg:plan>/<reg:deduce>/</think>/<ans>` route, declared answer,
  real EOS, no loop marker;
- one bypass prompt: valid `<reg:bypass>` route, declared answer, real EOS.

If either check fails, the adapter is not eligible for evaluation.

---

## 7. Strict evaluation

```bash
python eval/robust_eval.py \
  --input data/held_out_eval_200_v2.jsonl \
  --adapter checkpoints/sft_v3_adapter \
  --expected-sha256 95E2B237AE2B724DC0856AEE7220EBD79436F91630F77BBB3BAD0913127F10ED \
  --arms A B C D E F --budgets 256 512 1024 2048
```

The `--expected-sha256` value above is the current CRLF working-tree hash. It must be replaced
with the LF-normalized hash after r4. The evaluator records:

- full generation text;
- `strict_correct`, `lenient_correct`, and failure reason;
- EOS and truncation;
- route-aware format validity;
- `tokens_emitted` (total generated tokens);
- `tokens_per_correct` and `correct_per_1k_tokens`;
- dynamic joules when NVML is available;
- OAA/AUC_OAA, TokenEff, and overthinking score;
- dataset hash and run manifest.

It does **not** yet record `thinking_tokens` and `answer_tokens` separately. Do not describe
`tokens_emitted` as thinking tokens.

### Primary decision rule

At 256 tokens, Arm C must satisfy all three:

1. Paired 95% CI for `J/correct_D − J/correct_C` strictly excludes zero.
2. Strict accuracy is non-inferior to D: lower 95% CI bound for `Acc_C − Acc_D` is at least
   −2.0 percentage points.
3. Terminal-answer/format-valid rate is at least 90%.

If the run stops early because of Kaggle quota or infrastructure, report it as **incomplete**.
Do not pool partial cells into a headline, and do not call the missing cells a negative result.

---

## 8. What a receipt must contain

A claim is admissible only if the receipt includes:

1. The exact command.
2. The git commit and tag.
3. The LF-normalized dataset SHA-256.
4. The base model name and revision.
5. The adapter path and adapter hash.
6. Package versions and GPU/runtime.
7. Per-cell raw output, strict score, token count, EOS flag, and energy (if measured).
8. The analysis script and random seed.
9. A statement of completeness: full grid, diagnostic subset, or incomplete.

A CSV table with no raw traces is not a receipt.

---

## 9. Common failure modes

- **CRLF/LF mismatch.** The r3 blocker. Fix with r4, do not bypass.
- **Using `apply_chat_template` for SFT.** It can drop the reasoning trace at `</think>`.
- **Fake EOS strings.** `"<｜end of sentence｜>"` is seven ordinary tokens, not the tokenizer EOS.
- **Stopping Arm C at `</reg:verify>`.** It removes the answer channel and makes strict
  scoring impossible.
- **Scoring with `deduce_equality_rhs` or a trailing-number regex.** It turns truncated loops
  into “correct” answers. Diagnostic only.
- **Calling a partial run complete.** The old six-arm run is the cautionary example:
  2,119/4,800 cells, 1,955 saved rows, no completed headline.
- **Treating a Tier B SmolLM2 pilot as the flagship result.** It is a different model and
  a different experiment.
- **Reporting total tokens as thinking tokens.** The split is not instrumented.

---

## 10. References

- [`PREREG_VALUATION_V2.md`](PREREG_VALUATION_V2.md) — frozen protocol.
- [`docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`](docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md) — framing and metric definitions.
- [`docs/EVIDENCE_STATUS.md`](docs/EVIDENCE_STATUS.md) — current claim ledger.
- [`eval/strict_scorer.py`](eval/strict_scorer.py) — declared-answer scorer.
- [`eval/robust_eval.py`](eval/robust_eval.py) — six-arm harness.
- [`data/symbolic_dataset.py`](data/symbolic_dataset.py) — manual chat formatting and labels.
- [`training/sft_train.py`](training/sft_train.py) — SFT run manifest and generation gate.
- [`experiments/kaggle/sft_v3/README.md`](experiments/kaggle/sft_v3/README.md) — Kaggle kernel.
