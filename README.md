# SymboLM v3 — Spend Tokens Where They Change the Answer

**Token efficiency first. Correctness is the constraint. Biology is the design analogy.**

**Author / Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**Repository branch:** `research/rescue-v3`  
**Status date:** 2026-10-23  
**Current phase:** documentation audit and **r4 LF provenance fix complete**; **SFT still
must be rerun**; **no v3 adapter and no v3 GPU result**.  
**SFT kernel:** [`experiments/kaggle/sft_v3/`](experiments/kaggle/sft_v3/) (tag `sft-v3-20261023-r4`); the last Kaggle run (r3) is `ERROR`, not queued. No r4 training run has been launched.  
**Model repositories:** all four Hugging Face model/checkpoint repositories were made private on 2026-10-22. Public links return 401; nothing was deleted.

---

> **Read this first.** The v2.1 public headline — *“37.0% / Format Value Confirmed”* — is
> **retracted**. The old six-arm run stopped at **2,119 of 4,800 cells** because the free
> Kaggle quota/infrastructure limit ended the session. It is an **incomplete run**, not a
> completed result and not a negative result. Its saved cells used a lenient scorer that is
> not the v3 scorer. The old README and whitepaper are preserved as historical documents in
> [`docs/ARCHIVE_README_2.1.md`](docs/ARCHIVE_README_2.1.md) and
> [`docs/ARCHIVE_WHITEPAPER_2.1.md`](docs/ARCHIVE_WHITEPAPER_2.1.md).

## 1. The thesis: allocate deliberation, then measure the cost

The scientific question is not “does a pretty symbolic format make a small model smart?”

> Under a fixed correctness constraint, does a **fast reflex route for atomic facts plus a
> bounded deliberative route for everything else** produce more correct answers per generated
> token — and per joule — than ordinary chain-of-thought, answer-only, or a router-only
> control?

The primary dependent variable is **token efficiency of correct reasoning**. Accuracy is a
**non-inferiority constraint**: an arm that is cheaper because it solves fewer problems has
not won. When NVML is available, the same comparison is made in joules per correct answer.

The register format is a **serving contract**, not a claimed source of reasoning ability.
Its possible value is that it gives the model a short, checkable execution contract and gives
the serving system an early-exit signal. The biological language is a design analogy, not a
mechanism claim. The full framing is in
[`docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`](docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md).

### What “biologically inspired” means here — and what it does not

| Analogy | What is implemented | What is not claimed |
|---|---|---|
| **System 1 / reflex** | `<reg:bypass>answer</reg:bypass>` for atomic facts | A separate fast brain module, or general reasoning ability |
| **System 2 / deliberation** | `<reg:plan>`, `<reg:deduce>`, optional `<reg:verify>`, then `<ans>` | That typed registers cause better reasoning |
| **Bounded working memory** | A short, parseable register trace | A cognitive-science model of working memory |
| **Metabolic parsimony** | Compact SFT targets; length-aware reward code | That parsimony training produced the v3 adapter |
| **Router** | Prompt-level atomic/non-atomic route selection | A learned sparse router or a mixture of experts |
| **Tool use** | Arm E Python blocks; `symbolm/tiny_terminal.py` | That the Tiny Terminal is active in the v3 SFT run |

The current base model is a **dense 1.5B transformer** with a LoRA adapter. There is no
sparse expert routing, shared global KV cache, CED encoder, or Engram module in this
experiment. The “20×” multiplication, the “224×” KV projection, and the “20-watt brain”
comparison are **not measurements** and are not current claims. See §5 of the framing
document.

## 2. What survives the audit

| Claim | Status | Token-efficiency reading |
|---|---|---|
| Atomic facts can be answered by a short bypass reflex | **Supported as a narrow mechanism** | 32/1,000 strict v2.1 passes were atomic bypass answers; the atomic subset was 32/40. Useful for a factual reflex, not general reasoning. |
| The full register format adds value over a router-only control | **Not established** | SmolLM2 pilot: C 9.5% vs D 10.5%; paired 95% CI `[-844.57, +861.30]` includes zero. C 1,265.75 J/correct vs D 1,301.41 J/correct. |
| Answer-only is the cheap floor | **Supported in the SmolLM2 pilot** | B 15.0% at 8.1 mean tokens and 25.84 J/correct. Cheap, but only 15% strict accuracy; not a general replacement for reasoning. |
| The sandbox arm is a positive engineering result | **Supported in the SmolLM2 pilot** | E 23.0% at 120.2 mean tokens and 444.30 J/correct vs A 7.0% at 167.6 tokens and 2,052.41 J/correct. This is tool use, not format value. |
| The v2.1 model is a strong math reasoner | **Retracted** | Strict rescore of the 1,000-question run: 3.2% (32/1,000), 91.7% truncated, 823 looped traces. |
| 100% MATH500 / cross-family transfer | **Retracted** | Both were n=5 anecdotes, not benchmarks. |
| 37.0% overall accuracy / “Format Value Confirmed” | **Retracted** | Run stopped at 2,119/4,800 (44%); 37.0% is a commit-message number, not a completed result. |
| Sub-50-token reasoning | **Not supported** | The v2.1 run averaged 494.5 tokens, not 253.8. |

The honest exploratory signal from the partial six-arm run is narrower: on the 57 cells
common to all arms at budget 2048, strict declared-answer accuracy was C 26.3%, D 1.8%,
F 42.1%, A 0%, B 29.8%, E 1.8%. That is **n=57, post-hoc, and not the preregistered
200 × 4 experiment**. It motivates the re-run; it does not replace it. It is also not a
token-efficiency result, because the old Arm C traces lack `<ans>` and the old token counts
come from a scoring path that is not the v3 strict scorer.

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
   hash mismatch were reported as completed. The old six-arm run is the main example:
   2,119/4,800 cells, 1,955 saved rows, and a headline that was never a completed result.

## 4. The corrected pipeline

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
- The r3 remediation pins the base model name, immutable base revision, and exact
  runtime packages, rejects a dirty or unverified git checkout, and has no branch
  fallback. The tokenizer extension is rebuilt if its recorded base revision does
  not match.
- The r4 provenance revision adds `.gitattributes` LF rules for the canonical data,
  normalizes the working files to LF, and records LF byte hashes in the SFT and
  benchmark manifests, the preregistration, the evaluator, and the Kaggle kernel.
- The SFT artifact initializes both the input and output rows for genuinely added
  symbols, saves both `embed_tokens` and `lm_head` through `modules_to_save`, and keeps
  Trainer checkpoints separate from the final adapter. The package gate reads the
  actual safetensors keys rather than trusting `adapter_config.json` alone.
- The post-training gate checks one reasoning and one bypass prompt: each must emit a
  declared answer of the expected route, valid register structure, and the real EOS
  token, with no loop marker.
- **Not in this run:** GRPO, Tiny Terminal execution, FSM-constrained decoding, or a
  mixture-of-experts layer. Those are historical or future work.

### Evaluation

- `eval/strict_scorer.py` accepts only `<ans>…</ans>` and
  `<reg:bypass>…</reg:bypass>` for the register arm. Baseline arms may use a closed
  `\boxed{}` because their prompt requests it.
- Truncation without EOS = **fail**. No trailing-number fallback. No
  `deduce_equality_rhs` fallback in the primary score.
- `eval/robust_eval.py` does **not** stop early at an answer in the strict run; it
  lets generation reach the real EOS or the token budget, so EOS is not structurally
  removed from every answered trace. It records full traces, EOS, route-aware format
  validity, strict and lenient scores, token counts, joules, and the dataset hash
  preflight.
- `tokens_emitted` is total generated tokens. The harness does **not** yet log a
  separate thinking-token/answer-token split; no document may call total tokens
  “thinking tokens.”
- The benchmark is `data/held_out_eval_200_v2.jsonl` (SHA-256
  `F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707`; r4 LF bytes).
- One documented correction: `MY2021_001` is **169.5**, not 169.
- The v2 file also flags 20 duplicate-prompt groups (27 redundant rows); primary
  analysis is n=200, with a unique-prompt sensitivity analysis.

## 5. Current status and the r4 provenance fix

The pipeline is documented and locally testable, but the v3 SFT adapter does **not**
exist. The last Kaggle SFT kernel (r3) is `ERROR`, not queued. Its preflight failure was
a line-ending mismatch:

```text
Train hash mismatch:
got      BB50D9CBD56B0D2B515D7B922DF10578F587AA60358B0FF87A11821C4D3D94DC
expected 5D05003B495DE3322568C4B40713CCE376A1AC05AC9B85A50B579A68A49FF63E
```

The committed r3 data blobs were already LF, but the r3 tag did not force LF checkout.
Windows `core.autocrlf=true` materialized CRLF, the manifest recorded CRLF hashes, and
the Linux kernel hashed the LF files. The r4 tag (`sft-v3-20261023-r4`) fixes this with
`.gitattributes`, LF-normalized working files, regenerated manifests, and updated hash
constants. The rows, split, and targets are unchanged. No hash check was bypassed.

The canonical r4 hashes are:

```text
source      724581AD5F3DEE4A1A4D2C321C167E13A016F64E023492AF6460D2BF04A3B7A7
train       BB50D9CBD56B0D2B515D7B922DF10578F587AA60358B0FF87A11821C4D3D94DC
validation  3FCFCC4141522062087D0F35FDF2C54226D454A1A71FE59BEA26F42636FDEDCD
benchmark   F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707
```

**Until a clean r4 adapter passes its package gate, there is no v3 result.** The 4,800-cell
evaluation also requires separate explicit approval; it has not been launched.

## 6. The next experiment

The preregistration is [`PREREG_VALUATION_V2.md`](PREREG_VALUATION_V2.md). The design is a
full **200 problems × 4 budgets × 6 arms = 4,800 cells** run with:

- the same 4,600-target SFT split for every register arm;
- identical system prompts in training and evaluation;
- strict declared-answer scoring;
- paired bootstrap on J/correct and a strict accuracy non-inferiority check;
- a pre-specified rule that **no format-value claim** is allowed unless C beats D on
  the paired CI and is not worse by more than 2 pp;
- a token-efficiency report (tokens per correct answer, OAA/AUC_OAA, TokenEff, and
  joules per correct answer when NVML is available).

The first run after this rescue should be a 100 × 4 diagnostic to catch formatting or
EOS regressions. That diagnostic is **not** the primary result. The full 200 × 4 run is.

## 7. Quick start

```bash
# 1. Build the canonical SFT split and audit the benchmark
python scripts/prepare_research_sft.py
python scripts/audit_ground_truth.py

# 2. Formatting regression tests (no model download)
python -m unittest discover -s tests

# 3. Train (GPU/TPU; writes run_manifest.json and a generation check)
python training/sft_train.py \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --base_revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --train_file data/research_sft/train.jsonl \
  --val_file data/research_sft/val.jsonl \
  --output_dir checkpoints/sft_v3_trainer \
  --adapter_dir checkpoints/sft_v3_adapter \
  --generation_check 2 --strict_generation_check

# 4. Evaluate strictly
python eval/robust_eval.py \
  --input data/held_out_eval_200_v2.jsonl \
  --adapter checkpoints/sft_v3_adapter \
  --expected-sha256 F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707 \
  --arms A B C D E F --budgets 256 512 1024 2048
```

The commands above use the r4 LF hashes. They still require a clean adapter; do not run
the evaluation before SFT and the package gate pass.

## 8. Repository map

| Path | Purpose |
|---|---|
| `docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md` | Canonical token-efficiency and biology-analogy framing |
| `docs/MARKDOWN_AUDIT.md` | File-by-file audit of every first-party Markdown document |
| `PREREG_VALUATION_V2.md` | Frozen protocol for the rescue run |
| `docs/RESEARCH_REDIRECT.md` | What survives, what is retracted, and why |
| `docs/EVIDENCE_STATUS.md` | Short evidence-status table |
| `EVIDENCE_TRUTH_TABLE.md` | Full claim-by-claim historical audit |
| `RESULTS_SUMMARY.md` | Audit narrative and strict rescore |
| `VALUATION.md` | SmolLM2 pilot token/energy numbers |
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

## 9. Citation and contact

If you use this work, cite the **v3 research note**, not the retracted v2.1 scorecard.
The author is Sanish Gyawali (Rupandehi, Nepal). The repository is
[https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM).

**One rule for collaborators:** do not add a number to this README unless it has a
receipt, a hash, and a command that reproduces it. Do not add a token-efficiency claim
unless the strict correctness gate and the paired interval support it.
