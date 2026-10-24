# SymboLM v3 SFT Kernel

**Kernel:** `sanishgyawali/symbolm-v3-sft-rescue`  
**Purpose:** train one clean LoRA adapter on the canonical 4,600-target SFT split.  
**Not a result:** this kernel does not evaluate format value, routing value, or benchmark accuracy.

> [!NOTE]
> **Current status (2026-10-24): r5 trained but failed the generation gate; r6 is the
> token-delta repair revision and is running on Kaggle with no result yet.** The r4 tag
> (`sft-v3-20261023-r4`) passed the full Kaggle preflight. Its training stopped before an
> optimizer step because Kaggle supplied `torchao 0.10.0`, below PEFT 0.21.2's `0.16.0`
> minimum; that was a packaging failure, not a data or scientific result. The r5 tag
> (`sft-v3-20261023-r5`) removed the unused `torchao` and then **completed training**:
> 777 steps / 3 epochs, final eval loss `0.2768503725528717`. It **failed the strict
> generation gate** and was not packaged. The reasoning probe `concept_poly_30` declared
> no answer (`format_reason: loop_marker:letterSpacing`), while the bypass probe
> `concept_bypass_183` returned `100` correctly. The cause is concrete: the saved r5
> adapter had byte-identical, frozen `<ans>`, `</ans>`, and `answer` output rows. Their
> logits were permanently tied, so greedy decoding always chose the lower-id `answer`
> token and the model emitted `answer62answer` instead of `<ans>62</ans>`. That is a real
> bug, not a scientific result. The r6 tag (`sft-v3-20261023-r6`) adds trainable deltas
> for the 47 genuinely added tokenizer rows, keeps the resized `modules_to_save` matrices
> frozen during training, merges the deltas before save, and makes both the training
> script and package gate reject exactly tied marker rows. r6 is running; no result
> exists yet. The kernel is efficiency-oriented, but it is not evidence that training
> improves tokens per correct answer. Token efficiency is evaluated only after a clean
> adapter passes
> the package gate.

## Files

| File | Purpose |
|---|---|
| `kernel-metadata.json` | Kaggle kernel definition (private, GPU, internet) |
| `symbolm_v3_sft.ipynb` | Generated notebook; the Kaggle `code_file` |
| `make_notebook.py` | Regenerates the notebook deterministically |
| `run_sft_kernel.py` | Hash preflight, full dataset smoke test, SFT launch, package gate |
| `package_bundle.py` | Optional self-contained Kaggle dataset bundle |
| `requirements.lock.txt` | Exact runtime pins used by the canonical notebook and package gate |
| `upload/` | Generated bundle output; ignored by git |

## What the kernel does

Before LoRA injection it removes the Kaggle image's unused `torchao` package. PEFT 0.21.2
raises if it finds `torchao < 0.16.0`, even though this run does not use torchao; the repair
removes the failure before LoRA injection instead of changing the pinned PyTorch stack. The
run manifest records `"torchao": "absent"` and the package gate requires that value.

1. Verifies the pinned base model/revision and the canonical SFT train/validation
   hashes against `data/research_sft/manifest.json`.
2. Builds the exact extended tokenizer that training will use.
3. Walks **all 4,140 training rows** and asserts:
   - the prompt is an exact token prefix of the full sequence;
   - prompt labels are exactly `-100`, and every target token is supervised;
   - the sequence begins with the tokenizer's real BOS and ends with its real EOS;
   - the target passes the same register-contract validator as evaluation, with the
     expected reasoning/bypass route and a terminal declared answer;
   - the split is exactly 3,780 reasoning + 360 bypass rows.
4. Runs the full 460-row validation check (420 reasoning + 40 bypass).
5. Trains with:
   - `--no_quant` (1.5B model, one T4);
   - batch 2, grad-accum 8, effective batch 16;
   - 3 epochs, lr `2e-4`, seed 42;
   - auto-resume **off**;
   - `--generation_check 2 --strict_generation_check`.
6. Initializes the input **and output** rows for genuinely added symbols, then trains a
   small delta on those rows. The tokenizer extension adds 47 genuinely new rows,
   including `<ans>`/`</ans>`. A frozen output row cannot be distinguished from an
   identical anchor row, so r6 gives every added row a trainable delta, merges the
   deltas into the exact resized `embed_tokens` and `lm_head` rows before save, and
   still writes the matrices through `modules_to_save`. Existing base tokens such as
   `ans` are not overwritten.
7. Refuses to package unless one reasoning and one bypass generation check each emit a
   nonempty declared answer of the expected route, valid register structure, and the
   real EOS token, with no loop marker. It also verifies the actual safetensors keys,
   checks that the saved `<ans>`, `</ans>`, and `answer` LM-head rows are not exactly
   tied, and keeps the Trainer checkpoint tree out of the archive.

## Repository inputs

The notebook prefers the optional bundle dataset:

```text
/kaggle/input/symbolm-v3-sft-bundle/repo_bundle.tar.gz
```

If it is absent, it clones the exact pinned tag:

```text
sft-v3-20261023-r6
```

> **r4/r5/r6 note:** r4 enforces LF canonical data; its Kaggle preflight passed. r5 keeps
> the same data and hashes and removes the image-provided `torchao 0.10.0` before LoRA
> injection. r5 training completed but failed the generation gate because frozen `<ans>`,
> `</ans>`, and `answer` output rows were byte-identical. r6 fixes that with trainable
> token deltas and a stricter package check. The r3 tag is frozen but not usable for a
> Linux training run because of the CRLF/LF hash mismatch. There is no branch fallback;
> do not point a training run at r3 or r5.

There is **no branch fallback**. A missing tag, a tag/commit mismatch, a missing bundle
manifest, or a bundle hash mismatch is a hard failure. Every training command requires
both `--expected_git_head` and `--expected_git_tag`.

### Optional bundle upload

```bash
python experiments/kaggle/sft_v3/package_bundle.py
kaggle datasets create -p experiments/kaggle/sft_v3/upload
```

Then add this to `kernel-metadata.json`:

```json
"dataset_sources": ["sanishgyawali/symbolm-v3-sft-bundle"]
```

The bundle is not required when the tag is public. Do not add the dataset source until
the dataset exists, or Kaggle will reject the kernel push.

## Local verification (no GPU, no training)

```bash
# Regenerate the notebook from source.
python experiments/kaggle/sft_v3/make_notebook.py

# Model-free regression suite.
python -m unittest discover -s tests

# Data/tokenizer/target preflight. This downloads the tokenizer and checks all 4,600 rows.
python experiments/kaggle/sft_v3/run_sft_kernel.py \
  --mode check \
  --base_revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --expected_git_head "$(git rev-parse HEAD)" \
  --expected_git_tag sft-v3-20261023-r6
```

The last command must print `Dataset smoke test passed` and write
`KERNEL_CHECK.json`. It does not train.

## Push and run

```bash
# From the repository root, after committing and pushing the tag:
kaggle kernels push -p experiments/kaggle/sft_v3
```

Or use the Kaggle MCP tool with the kernel directory. The kernel is private and starts
a GPU run. **Do not launch the 4,800-cell evaluation until the SFT package gate passes.**

Expected output under `/kaggle/working/symbolm_v3_sft/`:

```text
adapter/
  adapter_config.json
  adapter_model.safetensors
  tokenizer files
trainer/
  checkpoint-*           # retained for audit; never included in the adapter tarball
run_manifest.json
KERNEL_RECEIPT.json
symbolm_v3_sft_adapter.tar.gz
```

## Stop conditions

Stop and fix the pipeline if any of these occurs:

- base revision, tag, or commit mismatch;
- data hash or split-kind mismatch;
- any preflight assertion failure;
- training exits nonzero;
- `run_manifest.json` is not `completed`;
- either generation check lacks an expected-route declared answer, valid format, or real EOS;
- a loop marker appears;
- `adapter_config.json` does not list `embed_tokens` and `lm_head`, or the actual
  safetensors keys lack either tensor;
- the saved LM-head rows for `<ans>`, `</ans>`, and `answer` are exactly tied, or the
  run manifest does not record a completed token-delta merge and distinct-row check;
- the adapter directory or tarball contains a `checkpoint-*` entry;
- the runtime package versions differ from `requirements.lock.txt`;
- `torchao` is present, or `run_manifest.json` does not record `"torchao": "absent"`;
- the adapter files or tarball are missing.

A failed generation check means the adapter is **not** admissible, even if the training
loss looks good.

## Runtime

One T4, 777 optimizer steps total (`ceil(4,140 x 3 / 16)`), sequence length 1024.
Expect several hours. Kaggle may reclaim the session; because auto-resume is deliberately
disabled for the primary run, a reclaimed session must be restarted from a fresh output
directory rather than silently resuming a stale checkpoint. For a deliberate resume, use
`training/sft_train.py --resume_from_checkpoint <path>` and document it as a deviation.

## Next stage

1. 100-problem x 4-budget x 6-arm diagnostic.
2. 200-problem x 4-budget x 6-arm strict evaluation.
3. Apply the preregistered C-vs-D decision rule in `PREREG_VALUATION_V2.md`.
