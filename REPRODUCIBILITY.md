# SymboLM Reproducibility Guide

**Author / Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**Branch:** `research/rescue-v3`  
**Status date:** 2026-10-24  
**Current status:** r6 SFT is complete and gate-eligible; **no v3 benchmark result exists**.
The r4 provenance fix is validated: its Kaggle preflight passed the hashes and all 4,600 rows.
The r4 training run then stopped before an optimizer step because Kaggle supplied `torchao
0.10.0`, below PEFT 0.21.2's `0.16.0` minimum for LoRA dispatch. That was a packaging failure,
not a data or scientific result. The r5 package fix (tag `sft-v3-20261023-r5`) removed the
unused package; r5 then completed 777 training steps but failed the strict generation gate
because its saved `<ans>`, `</ans>`, and `answer` output rows were byte-identical. The r6
revision (tag `sft-v3-20261023-r6`) added trainable deltas for the 47 genuinely added
tokenizer rows and rejected tied marker rows. The r6 run completed 777 steps / 3 epochs with
final eval loss `0.00748848170042038` and passed both the strict mixed-kind generation gate
and the package gate. The adapter SHA-256 is
`204823BC216BBF7329D6C7C5159ACF8F6688FD321AD1652757C2A9A9D066B31C`; receipts are in
`results/r6_closeout/`.

This guide describes how to reproduce the **pipeline** and the **audit**, not a result that
has not been produced. Every reported number must have a command, a hash, and a receipt.

---

## 1. What is reproducible today

| Item | Command | Status |
|---|---|---|
| Canonical SFT split | `python scripts/prepare_research_sft.py` | Works; outputs 4,140 / 460 |
| Ground-truth audit | `python scripts/audit_ground_truth.py` | Works; applies `MY2021_001 = 169.5` and duplicate flags |
| r7 trace audit (dry run) | `python scripts/audit_training_traces.py` | Works; writes `results/r7_verified/AUDIT_DRY_RUN.json`, no verified split |
| Model-free tests | `python -m unittest discover -s tests` | Works; 109 tests pass |
| Strict scorer | `python -m eval.strict_scorer` / unit tests | Works |
| SmolLM2 pilot token/energy table | See [`VALUATION.md`](VALUATION.md) | Tier B historical measurement |
| Old six-arm run | See audit docs | **Incomplete**; not reproducible as a completed result |
| v3 SFT run | `experiments/kaggle/sft_v3/` | **Complete at the pre-evaluation gates**; r6 trained 777 steps and passed the generation and package gates. No benchmark result. |
| Full 4,800-cell v3 run | hardened v3 harness | **Not run**; requires the 2,400-cell diagnostic, a clean adapter, and separate approval |

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

### r5 generation failure and r6 token-delta repair

The r5 run completed 777 steps / 3 epochs with final eval loss
`0.2768503725528717`, then failed the strict generation gate. The reasoning probe
`concept_poly_30` declared no answer (`format_reason: loop_marker:letterSpacing`); the
bypass probe `concept_bypass_183` returned `100`. The saved adapter had byte-identical
frozen `<ans>` (151665), `</ans>` (151666), and base `answer` (9217) rows in both the input
embedding and LM head. A frozen head cannot separate tied logits, so greedy decoding always
selected the lower-id `answer` token.

The r6 revision keeps the full resized `modules_to_save` matrices frozen but trains a small
delta for each of the 47 genuinely added tokenizer rows (`151665`-`151711`). Forward hooks
apply the deltas without copying the 151,712 x 1,536 matrices; `merge_token_deltas` adds
them into the active rows before saving. The training script and the package gate fail
closed if the saved `<ans>`, `</ans>`, and `answer` rows are still exactly tied. The
adapter format and package contract are unchanged. This is an artifact repair, not a change
to the hypothesis, data, split, targets, evaluation protocol, or decision rule.

The r6 run completed 777 steps / 3 epochs with final eval loss
`0.00748848170042038`. The strict generation gate passed on both probes:
`concept_poly_30` returned `<ans>62</ans>` and `concept_bypass_183` returned
`<reg:bypass>100</reg:bypass>`, both with valid format, a terminal answer, and a real EOS.
The package gate passed with `"torchao": "absent"`, merged deltas, and distinct marker rows.
The adapter SHA-256 is
`204823BC216BBF7329D6C7C5159ACF8F6688FD321AD1652757C2A9A9D066B31C`; the bundle SHA-256 is
`C711EF8BF119DC0B8DDF78BF09F6EDB2276996312901936C1EA67BED55FCC573`. Receipts:
`results/r6_closeout/`.

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

## 6. SFT (r6 checkout)

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

If either check fails, the adapter is not eligible for evaluation. For r6, the manifest must
also record `token_delta.present = true`, `token_delta.merged = true`, and
`token_delta.terminal_marker_check.distinct = true`; the package gate independently reads
the saved LM-head rows.

---

## 7. Strict evaluation

> **Harness status (2026-10-25):** `eval/robust_eval.py` and `eval/run_contract.py` are the
> hardened v3 path. Prompt construction and strict scoring are parity-tested against
> `symbolic/prompts.py` and `eval/strict_scorer.py`; dataset/adapter/contract mismatches,
> corrupt checkpoints, prompt/generation exceptions, persistence failures, decoding
> corruption, and incomplete grids abort the run. A checkpoint is resumable only under the
> same dataset, adapter, subset, budgets, run kind, prompt/scorer code, and evaluator code.
> The 2,400-cell diagnostic has **not** been run; do not label a run as v3 until it exists.

```bash
python eval/robust_eval.py \
  --input data/held_out_eval_200_v2.jsonl \
  --adapter checkpoints/sft_v3_r6/adapter \
  --expected-adapter-sha256 204823BC216BBF7329D6C7C5159ACF8F6688FD321AD1652757C2A9A9D066B31C \
  --expected-sha256 F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707 \
  --subset-size 100 --subset-seed 20261025 --run-kind diagnostic \
  --out-dir results/v3_diagnostic \
  --arms A B C D E F --budgets 256 512 1024 2048
```

The `--expected-sha256` value is the canonical LF hash. The full 4,800-cell run omits
`--subset-size`, uses `--run-kind full`, and requires separate explicit approval. The
evaluator records:

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
- **Tied terminal-marker rows.** r5 froze `<ans>`, `</ans>`, and `answer` with byte-identical
  output rows, so greedy decoding could never emit the markers. r6 trains deltas for the 47
  added rows, merges them before save, and both gates reject an exact tie. Do not bypass.
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
