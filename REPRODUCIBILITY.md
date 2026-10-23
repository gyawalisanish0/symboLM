# SymboLM Reproducibility Guide

**Author / Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**Branch:** `research/rescue-v3`  
**Status date:** 2026-10-23  
**Current status:** the v3 adapter does not exist. The r4 provenance fix is validated: its
Kaggle preflight passed the hashes and all 4,600 rows. The r4 training run then stopped
before an optimizer step because Kaggle supplied `torchao 0.10.0`, below PEFT 0.21.2's
`0.16.0` minimum for LoRA dispatch. This is a packaging failure, not a data or scientific
result. The r5 package-fix revision (tag `sft-v3-20261023-r5`) removes the unused package;
no r5 training run has completed. No v3 benchmark result exists.

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
| v3 SFT run | `experiments/kaggle/sft_v3/` | **Incomplete**; r4 preflight passed, then packaging stopped at LoRA injection (`torchao`); r5 fix prepared |
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
`run_manifest.json` rather than replaced in the lock file. `torchao` is intentionally
absent: PEFT 0.21.2 raises during LoRA dispatch if it finds `torchao < 0.16.0`, and this
run does not use torchao. The notebook removes the image-provided copy, and the package
gate requires `"torchao": "absent"` in the run manifest.

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

### r4 provenance fix

The r3 manifest hashes were computed over **Windows CRLF** files. The committed git blobs
were already **LF**, so a Linux/Kaggle checkout hashed differently and the r3 kernel failed:

```text
Train hash mismatch:
got      BB50D9CBD56B0D2B515D7B922DF10578F587AA60358B0FF87A11821C4D3D94DC
expected 5D05003B495DE3322568C4B40713CCE376A1AC05AC9B85A50B579A68A49FF63E
```

The r4 revision (`sft-v3-20261023-r4`) fixes this without changing any row. It adds
`.gitattributes` LF rules, normalizes the canonical working files, regenerates the manifest
and protocol hashes from LF bytes, and updates the kernel/evaluator constants. The current
hashes are:

```text
source      724581AD5F3DEE4A1A4D2C321C167E13A016F64E023492AF6460D2BF04A3B7A7
train       BB50D9CBD56B0D2B515D7B922DF10578F587AA60358B0FF87A11821C4D3D94DC
validation  3FCFCC4141522062087D0F35FDF2C54226D454A1A71FE59BEA26F42636FDEDCD
```

Do not bypass the hash check. If a checkout produces a different hash, the source of the
difference is line endings or a modified file, not a valid reason to relax the gate.

### r5 package fix

The r4 Kaggle preflight passed the full data and tokenizer gate, but training stopped
before optimization with:

```text
ImportError: Found an incompatible version of torchao. Found version 0.10.0,
but only versions above 0.16.0 are supported
```

PEFT 0.21.2 probes for torchao while building LoRA layers, even though this run never
uses torchao. The r5 revision removes the Kaggle-provided `torchao` before LoRA
injection and requires the run manifest to record `"torchao": "absent"`. It does not
replace PyTorch and does not change any data row or training hyperparameter.

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

Current LF hash:

```text
F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707
```

The old clean benchmark was `8369BD8…` when checked out with CRLF. Its LF bytes are
`575B247…`, which is also the old Kaggle manifest hash. The old mismatch was line endings,
not a different benchmark. Do not treat the old run as reproducible against the current file
without accounting for that difference.

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

## 6. SFT (r5 checkout)

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
  --expected-sha256 F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707 \
  --arms A B C D E F --budgets 256 512 1024 2048
```

The `--expected-sha256` value is the canonical LF hash. The evaluator records:

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

- **CRLF/LF mismatch.** Fixed in r4 with `.gitattributes` and LF hashes; do not bypass the check.
- **Old `torchao` in the Kaggle image.** PEFT 0.21.2 raises on `torchao < 0.16.0` during LoRA
  dispatch. The r5 notebook removes the unused package; the package gate requires it absent.
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
