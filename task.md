# SymboLM — Task Tracker

**Status date:** 2026-10-23  
**Branch:** `research/rescue-v3`  
**Current phase:** documentation audit complete; the planned **r4** provenance fix is still
pending. **No v3 adapter and no v3 GPU result.**

This tracker records implementation state, not scientific validity. A checked box means the
code or document exists; it does not mean the associated claim is confirmed.

## Historical phases (code exists; old results are retracted)

- [x] Phase 0: project scaffold.
- [x] Phase 1: symbolic DSL, tokenizer extension, converter.
- [x] Phase 2: data ingestion, dataset class, answer checker.
- [x] Phase 3: reward function, SFT, historical GRPO trainer.
- [x] Phase 4: Colab notebooks for setup, data prep, SFT, GRPO, evaluation.
- [x] Phase 5: benchmark, efficiency script, streaming inference, trace renderer.
- [x] Phase 6: config, requirements, original README.

The old v2.1 results produced by these phases are retracted or incomplete. See
[`docs/EVIDENCE_STATUS.md`](docs/EVIDENCE_STATUS.md).

## Phase 7: v3 rescue

### Data and scoring

- [x] Replace fake BOS/EOS with the real tokenizer special tokens.
- [x] Build SFT targets manually; stop using the Qwen chat template for training.
- [x] Assert prompt-prefix identity and mask prompt labels with `−100`.
- [x] Remove dialogue rows and deliberately wrong contrastive branches.
- [x] Build the canonical 4,600-target split (4,200 reasoning + 400 bypass).
- [x] Create the 4,140 / 460 train-validation split with a hash manifest.
- [x] Correct `MY2021_001` to 169.5.
- [x] Flag duplicate-prompt groups in the v2 benchmark.
- [x] Implement strict declared-answer scoring with EOS-aware truncation.
- [x] Remove early Arm C stop at `</reg:verify>` from the strict path.
- [x] Add route-aware format validation.
- [x] Add the six-arm, four-budget harness and overthinking metrics.
- [x] Add run manifests and a mixed-kind post-training generation check.
- [x] Pass the model-free test suite.

### Protocol and framing

- [x] Freeze `PREREG_VALUATION_V2.md` before any v3 scored GPU cell.
- [x] Tag the rescue branch `sft-v3-20261022-r3` (r2 superseded).
- [x] State token efficiency as the primary dependent variable.
- [x] State strict accuracy as a non-inferiority gate.
- [x] State that biological language is a design analogy, not a mechanism claim.
- [x] State that the old six-arm run is incomplete, not a negative result.
- [x] Audit every first-party Markdown file and add `docs/MARKDOWN_AUDIT.md`.

### Provenance and training

- [ ] Add `.gitattributes` with LF rules.
- [ ] Convert canonical data files to LF and regenerate the manifest hashes.
- [ ] Update `PREREG_VALUATION_V2.md`, docs, and kernel constants to LF hashes.
- [ ] Create the `r4` tag.
- [ ] Re-run the Kaggle SFT preflight (current r3 kernel is `ERROR`).
- [ ] Train the v3 adapter and pass the mixed-kind generation check.
- [ ] Package the adapter and pass the package gate.

### Evaluation

- [ ] Run the 100 × 4 × 6 diagnostic (not the primary result).
- [ ] Run the full 200 × 4 × 6 experiment (**requires separate explicit approval**).
- [ ] Report strict accuracy, tokens per correct, J/correct, format validity, and the
      preregistered C-versus-D decision.
- [ ] Publish the result, including a negative result if that is what the data show.

## Non-negotiable rules

1. Token efficiency is the primary lens; strict accuracy is the non-inferiority gate.
2. No format-value claim without C beating D on the paired CI and passing non-inferiority.
3. `tokens_emitted` is total generation, not “thinking tokens.”
4. A partial, quota-stopped run is not a result in either direction.
5. A projection is not a measurement; a biological metaphor is not a mechanism.
6. The 4,800-cell evaluation requires separate explicit approval.
