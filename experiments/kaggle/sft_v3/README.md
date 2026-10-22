# SymboLM v3 SFT Kernel

**Kernel:** `sanishgyawali/symbolm-v3-sft-rescue`  
**Purpose:** train one clean LoRA adapter on the canonical 4,600-target SFT split.  
**Not a result:** this kernel does not evaluate format value, routing value, or benchmark accuracy.

## Files

| File | Purpose |
|---|---|
| `kernel-metadata.json` | Kaggle kernel definition (private, GPU, internet) |
| `symbolm_v3_sft.ipynb` | Generated notebook; the Kaggle `code_file` |
| `make_notebook.py` | Regenerates the notebook deterministically |
| `run_sft_kernel.py` | Hash preflight, full dataset smoke test, SFT launch, package gate |
| `package_bundle.py` | Optional self-contained Kaggle dataset bundle |
| `upload/` | Generated bundle output; ignored by git |

## What the kernel does

1. Verifies the canonical SFT train/validation hashes against
   `data/research_sft/manifest.json`.
2. Builds the exact extended tokenizer that training will use.
3. Walks **all 4,140 training rows** and asserts:
   - the prompt is an exact token prefix of the full sequence;
   - prompt labels are exactly `-100`;
   - the target ends with the tokenizer's real EOS token;
   - the target declares `<ans>` or `<reg:bypass>`;
   - there are supervised target tokens after the prompt.
4. Runs the full 460-row validation check.
5. Trains with:
   - `--no_quant` (1.5B model, one T4);
   - batch 2, grad-accum 8, effective batch 16;
   - 3 epochs, lr `2e-4`, seed 42;
   - auto-resume **off**;
   - `--generation_check 2 --strict_generation_check`.
6. Refuses to package unless both generation checks emit a terminal answer and the
   real EOS token, with no loop marker.

## Repository inputs

The notebook prefers the optional bundle dataset:

```text
/kaggle/input/symbolm-v3-sft-bundle/repo_bundle.tar.gz
```

If it is absent, it clones the pinned tag:

```text
sft-v3-20261022-r1
```

If the tag is not pushed, it falls back to `research/rescue-v3` and prints a warning.
The receipt records the actual commit, so a fallback is visible.

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
python experiments/kaggle/sft_v3/run_sft_kernel.py --mode check
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
run_manifest.json
KERNEL_RECEIPT.json
symbolm_v3_sft_adapter.tar.gz
```

## Stop conditions

Stop and fix the pipeline if any of these occurs:

- data hash mismatch;
- any preflight assertion failure;
- training exits nonzero;
- `run_manifest.json` is not `completed`;
- either generation check lacks a terminal answer or real EOS;
- a loop marker appears;
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
