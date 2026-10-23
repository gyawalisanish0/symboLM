# SymboLM v3 SFT Kernel

**Kernel:** `sanishgyawali/symbolm-v3-sft-rescue`  
**Purpose:** train one clean LoRA adapter on the canonical 4,600-target SFT split.  
**Not a result:** this kernel does not evaluate format value, routing value, or benchmark accuracy.

> [!NOTE]
> **Current status (2026-10-23): the r4 provenance fix is applied; no r4 training run has
> been launched.** The last r3 kernel run is `ERROR`: it failed the train-hash preflight
> because the r3 manifest hashes were computed over Windows CRLF files while the committed
> blobs are LF. The data rows are identical; the bytes were not. The r4 tag
> (`sft-v3-20261023-r4`) adds `.gitattributes` LF rules, regenerates the manifest/protocol
> hashes from LF bytes, and updates the kernel constants. Do not bypass the preflight. The
> training run is also efficiency-oriented — compact canonical traces, bypass examples,
> terminal answers, and real EOS — but this kernel is not evidence that the training improves
> tokens per correct answer. Token efficiency is evaluated only after a clean adapter passes
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
6. Initializes the input **and output** rows for genuinely added symbols, then saves the
   exact resized `embed_tokens` and `lm_head` rows with the adapter via
   `modules_to_save`. The tokenizer extension adds `<ans>`/`</ans>`, so an adapter
   containing only LoRA weights would leave those rows different at evaluation.
   Existing base tokens such as `ans` are not overwritten.
7. Refuses to package unless one reasoning and one bypass generation check each emit a
   nonempty declared answer of the expected route, valid register structure, and the
   real EOS token, with no loop marker. It also verifies the actual safetensors keys and
   keeps the Trainer checkpoint tree out of the archive.

## Repository inputs

The notebook prefers the optional bundle dataset:

```text
/kaggle/input/symbolm-v3-sft-bundle/repo_bundle.tar.gz
```

If it is absent, it clones the exact pinned tag:

```text
sft-v3-20261023-r4
```

> **r4 note:** this tag enforces LF canonical data. The previous r3 tag is frozen but is not
> usable for a Linux training run because of the CRLF/LF hash mismatch. There is no branch
> fallback; do not point a training run at r3.

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
  --expected_git_tag sft-v3-20261023-r4
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
- the adapter directory or tarball contains a `checkpoint-*` entry;
- the runtime package versions differ from `requirements.lock.txt`;
- the adapter files or tarball are missing.

A failed generation check means the adapter is **not** admissible, even if the training
loss looks good.

## Runtime

One T4, about 776 optimizer steps total (`4,140 x 3 / 16`), sequence length 1024.
Expect several hours. Kaggle may reclaim the session; because auto-resume is deliberately
disabled for the primary run, a reclaimed session must be restarted from a fresh output
directory rather than silently resuming a stale checkpoint. For a deliberate resume, use
`training/sft_train.py --resume_from_checkpoint <path>` and document it as a deviation.

## Next stage

1. 100-problem x 4-budget x 6-arm diagnostic.
2. 200-problem x 4-budget x 6-arm strict evaluation.
3. Apply the preregistered C-vs-D decision rule in `PREREG_VALUATION_V2.md`.
