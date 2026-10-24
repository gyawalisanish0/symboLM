# Markdown Audit: Every First-Party Document

**Audit date:** 2026-10-24  
**Scope:** first-party `.md` files, excluding `.venv/`, `vendor/`, `scratch/`,
archived Kaggle output duplicates, and generated output directories.  
**Files audited:** 66<br>
**Relative-link pass:** zero broken links in the 66-file scope (archived Kaggle output duplicates excluded)

Some audit targets are gitignored in this working tree: the private grant documents under
`research/` and the `checkpoints/` model cards. They were still audited and banner-marked on
disk, but their edits will not appear in a normal `git status` unless they are force-added.
They are not current public claims.

This audit records what each document is, whether it contains current claims,
and what was done. It is not a claim ledger; the claim ledger is
[`EVIDENCE_STATUS.md`](EVIDENCE_STATUS.md). The primary framing is in
[`TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`](TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md).

## How to read the classification

- **Canonical current:** may be cited for the current framing or protocol.
- **Canonical methodology:** reference material on evaluation methods; not a results page.
- **Tier B / audit receipt:** measured or audited, but not a v3 result and not comparable
  to the flagship run without the stated caveat.
- **Design / projection:** proposed mechanism or analytical projection; not a measurement.
- **Archive / historical ops:** retained for provenance; not current guidance.
- **Current operational:** pipeline instructions; not a scientific result.
- **Historical / retracted:** old claim or receipt; a banner now marks it as historical.
- **Audit index:** this file.

## Finding summary

The old repository led with accuracy and “Format Value Confirmed.” The corrected
repo leads with token efficiency of correct reasoning and treats strict accuracy as a
non-inferiority gate. The biological language is a design analogy, not a mechanism
claim. The old six-arm run is incomplete (2,119/4,800; Kaggle quota/infrastructure),
not a completed or negative result. No admissible v3 adapter exists. The r3 SFT kernel is
`ERROR` on a CRLF/LF hash mismatch. The r4 LF provenance fix (tag `sft-v3-20261023-r4`)
passed the full Kaggle preflight, but training stopped before an optimizer step because
Kaggle supplied `torchao 0.10.0`, below PEFT 0.21.2's `0.16.0` minimum. That was a
packaging failure, not a data or scientific result. The r5 fix (tag `sft-v3-20261023-r5`)
removed the unused package and the run completed 777 training steps, but it failed the
strict generation gate: its frozen `<ans>`, `</ans>`, and `answer` output rows were
byte-identical, so greedy decoding could not emit the terminal markers. The r6 repair (tag
`sft-v3-20261023-r6`) trains deltas on the 47 added rows, merges them before save, and
rejects tied rows; no r6 run has completed. The 4,800-cell evaluation has not been
launched.

## File-by-file audit

| File | Classification | Finding / action |
|---|---|---|
| `AUDIT.md` | Historical / retracted | Historical banner added; retains old 50.5% audit row as historical finding. |
| `EVAL_V2_200_REPORT.md` | Historical / retracted | Historical duplicate; banner added. |
| `EVIDENCE_TRUTH_TABLE.md` | Historical / retracted | Token-efficiency/biology banner added; rows 16 and 20 corrected to incomplete/retracted. |
| `GEMINI.md` | Canonical current | Updated session directive; SESSION_CONTEXT is historical; records the r4 preflight pass, the `torchao` packaging stop, the r5 generation-gate failure, and the r6 repair. |
| `OPTIMIZATIONS_RESEARCH.md` | Design / projection | Banner added: engineering options/projections, not measured v3 results. |
| `PREREG_VALUATION.md` | Historical / retracted | Already historical/superseded. |
| `PREREG_VALUATION_V2.md` | Canonical current | Added dated framing, token-efficiency, r4 provenance, r4-outcome/r5 package notes, and the r5-outcome/r6 token-delta note. Hypotheses/endpoints unchanged. |
| `README.md` | Canonical current | Rewritten around token efficiency, strict-accuracy non-inferiority, biology-as-analogy, the quota-stopped old run, r4 provenance, r5 generation failure, and r6 token-delta repair; r6 is running, SFT and the full run are pending. Adds the Android/`llama.cpp` harness status. |
| `REPRODUCIBILITY.md` | Canonical current | Rewritten for v3, strict scoring, the r4 LF provenance fix, the r5 `torchao` fix and generation failure, the r6 token-delta repair, and token-efficiency reporting. |
| `RESULTS_SUMMARY.md` | Historical / retracted | Framing banner and tokens-per-correct table added; incomplete-run language corrected. |
| `SESSION_CONTEXT.md` | Historical / retracted | Correction note added: 20x invalid, 224x projection, old 100%/confirmed claims historical. |
| `VALUATION.md` | Tier B / audit receipt | Added tokens-per-correct table, C-vs-D null result, and biology caveat. Tier B only. |
| `WHITEPAPER.md` | Canonical current | Replaced the historical v2.1 root with a current v3 research note; records the r4 preflight pass, `torchao` packaging stop, r5 generation-gate failure, and r6 repair. Old text preserved in docs/ARCHIVE_WHITEPAPER_2.1.md. |
| `checkpoints/symbolm_v2_adapter/README.md` | Historical / retracted | Historical model card; banner added; repository private. |
| `data/DATASET_CURATION_STRATEGY.md` | Design / projection | 60/20/20 marked historical; v3 4,600 split stated; 100%-immaculate guarantee removed. |
| `data/held_out_eval_200_v2_audit.md` | Tier B / audit receipt | Retained. Documents 200 rows, 173 unique prompts, 20 duplicate groups, MY2021_001=169.5. |
| `docs/AI_GRANT_OPEN_SOURCE_APPLICATION.md` | Historical / retracted | Banner and one-line correction; currently-running/definitive-evaluation claims corrected. Do not submit as-is. |
| `docs/ARCHIVE_README_2.1.md` | Archive / historical ops | Already banner-marked; preserved as historical provenance. |
| `docs/ARCHIVE_WHITEPAPER_2.1.md` | Archive / historical ops | Already banner-marked; preserved as historical provenance. |
| `docs/EVIDENCE_STATUS.md` | Canonical current | Rewritten with token-efficiency hierarchy, pilot ledger, incomplete-run language, and the r4/r5/r6 provenance, package, failure, and repair notes. |
| `docs/HONEST_PITCH_AND_SOCIAL.md` | Historical / retracted | Banner added; currently-running corrected to incomplete; Tier B receipts labeled. |
| `docs/MARKDOWN_AUDIT.md` | Audit index | This file. Lists every first-party Markdown document and the action taken. |
| `docs/RESEARCH_REDIRECT.md` | Canonical current | Rewritten with token efficiency primary, biology as analogy, quota-stop language, decision rules, and the r4/r5/r6 status. |
| `docs/ROADMAP.md` | Design / projection | Long-term plan: symbolic-interface scaling, teacher–student distillation, architecture-level edge efficiency, gates, controls, and resources. Android + `llama.cpp` is the first deployment target; `DomainAI` is the harness. Planning only; no measured claim. |
| `docs/STATE_OF_THE_ART_REASONING_AND_EVALUATION.md` | Canonical methodology | Removed “SymboLM solves” and hardware-receipt claims; reframed as hypotheses; updated the v3 artifact status through the r5 failure and r6 repair. |
| `docs/TINY_TERMINAL_AND_CONTEXT_PRUNING_SPEC.md` | Design / projection | Status changed to not active in v3; removed the outperform-70B claim. |
| `docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md` | Canonical current | New canonical doc: metric definitions, biology caveats, training scope, evidence ledger, claim rules, and the r4/r5/r6 status. |
| `eval/ABLATION_REAL_REPORT.md` | Tier B / audit receipt | Banner added; C 10.0% vs D 9.0% CI includes zero. Not format-value evidence. |
| `eval/ADAPTIVE_ALLOCATION_REPORT.md` | Historical / retracted | Historical receipt; banner added. |
| `eval/AI_ADVERSARIAL_CRITIQUE_LOG.md` | Historical / retracted | Historical audit log; banner added. |
| `eval/BACKTRACKING_STRESS_REPORT.md` | Historical / retracted | Historical receipt; banner added. |
| `eval/CONTAMINATION_AUDIT_REPORT.md` | Tier B / audit receipt | Banner added; contamination audit receipt, not a capability result. |
| `eval/EVAL_1000_MULTI_YEAR_REPORT.md` | Historical / retracted | Historical receipt; banner added; 50.5% is lenient, strict is 3.2%. |
| `eval/EVAL_200_BENCHMARK_REPORT.md` | Historical / retracted | Historical receipt; banner added. |
| `eval/EVAL_CROSS_DOMAIN_REPORT.md` | Historical / retracted | Historical receipt; banner added. |
| `eval/EVAL_CROSS_FAMILY_LLAMA_REPORT.md` | Historical / retracted | Historical receipt; banner added; n=5 transfer probe. |
| `eval/EVAL_V2_1000_REPORT.md` | Historical / retracted | Historical receipt; banner added; 67.6% is lenient, not strict. |
| `eval/EVAL_V2_200_REPORT.md` | Historical / retracted | Historical receipt; banner added. |
| `eval/MATH500_APPLES_TO_APPLES_REPORT.md` | Historical / retracted | Already deprecated; n=5 dry run is not a benchmark. |
| `eval/QUANT_EFFICIENCY_AUDIT_REPORT.md` | Historical / retracted | Historical receipt; banner added. |
| `eval/RESCORING_AUDIT_REPORT.md` | Tier B / audit receipt | Banner added; strict 3.2% and 91.7% truncation. Retained audit receipt. |
| `eval/V2_POST_MORTEM_AND_WORST_CASE_ANALYSIS.md` | Historical / retracted | Historical analysis; banner added. |
| `eval/VERIFIER_FALSIFICATION_REPORT.md` | Tier B / audit receipt | Banner added; methodology receipt, not a capability result. |
| `eval/cross_family_llama/symbolm_llama_1_7b_adapter/README.md` | Historical / retracted | Historical model card; banner added; n=5 probe, not universality. |
| `eval_cross_domain/EVAL_CROSS_DOMAIN_REPORT.md` | Historical / retracted | Historical duplicate; banner added. |
| `experiments/README.md` | Archive / historical ops | Banner added; HF repositories private; old accuracy headlines not current. |
| `experiments/kaggle/sft_v3/README.md` | Current operational | Added r3 `ERROR` history, the r4 preflight pass and `torchao` packaging stop, the r5 generation-gate failure and tied-row diagnosis, the r6 token-delta repair, and the token-efficiency caveat. r6 is running; no result exists yet. |
| `experiments/kaggle/sft_v3/upload/README.md` | Archive / historical ops | Generated r4 bundle metadata; stale and not used by the r6 kernel. Rebuild with `package_bundle.py` on the r6 tag if a bundle is needed. |
| `experiments/lightning/README.md` | Archive / historical ops | Banner added; old SFT/GRPO paths are not the v3 artifact. |
| `research/CLAUDE_EMPIRICAL_AUDIT_BRIEF.md` | Historical / retracted | Historical audit brief; banner added. |
| `research/DEEPSEEK_V4_FLASH_AND_SYMBOLM_CO_DESIGN.md` | Design / projection | 224x labeled assumption-only; co-design softened to hypothetical; no MoE in current model. |
| `research/EMERGENT_VENTURES_GRANT_APPLICATION.md` | Historical / retracted | Historical grant; banner added; do not submit as-is. |
| `research/EMPIRICAL_BENCHMARK_AUDIT_REPORT.md` | Historical / retracted | Historical audit; banner added. |
| `research/GRANT_PROJECT_DOSSIER_OCTOBER_2026.md` | Historical / retracted | Historical grant; banner added; do not submit as-is. |
| `research/HUGGING_FACE_MODEL_CARD_V1.0.md` | Historical / retracted | Historical model card; banner added; repository private. |
| `research/HUGGING_FACE_MODEL_CARD_V2.1_DEEPSEEK.md` | Historical / retracted | Historical model card; banner added; repository private. |
| `research/HUGGING_FACE_MODEL_CARD_V2.1_SMOLLM.md` | Historical / retracted | Historical model card; banner added; repository private. |
| `research/INCIDENT_LOG_POWER_OUTAGE_RUN.md` | Historical / retracted | Historical incident log; banner added. |
| `research/RELEASE_ROADMAP_V1.1_BENCHMARK_AND_QUANT.md` | Historical / retracted | Historical roadmap; banner added. |
| `research/RESEARCHER_PROFILE_AND_GRANT_STRATEGY.md` | Historical / retracted | Historical grant/strategy; banner added; do not submit as-is. |
| `research/STAGE_1_LOCAL_AUDIT_REPORT.md` | Historical / retracted | Historical receipt; banner added. |
| `research/STAGE_1_SFT_RESEARCH_REPORT.md` | Historical / retracted | Historical receipt; banner added. |
| `results/robust_eval/ROBUST_EVAL_REPORT.md` | Historical / retracted | Already warning-marked; incomplete 2,119/4,800 run, C traces lack <ans>. Not a result. |
| `symbolic/GRADIENT_REGISTERS_SPECIFICATION.md` | Design / projection | Status added: research extension, not implemented in v3. |
| `symbolic/SPECIFICATION.md` | Design / projection | Scope note added: v3 uses a subset; token minimization is a goal, not a result. |
| `task.md` | Canonical current | Separates implementation from scientific validity and records the r4 preflight pass, `torchao` packaging stop, r5 generation failure, r6 repair, and remaining SFT/evaluation blockers. |

## Cross-cutting corrections

1. **Token efficiency is the primary lens.** Accuracy is a non-inferiority constraint;
   raw accuracy is not the headline.
2. **`tokens_emitted` is total generation.** It is not “thinking tokens.” The split is
   not yet logged and must be instrumented before a thinking-token-specific claim.
3. **Biology is an analogy.** The current model is a dense 1.5B LoRA adapter, not a
   sparse MoE/CED/Engram system or a brain-like architecture.
4. **The old run is incomplete.** It stopped at 2,119/4,800 due to Kaggle
   quota/infrastructure. It is not a confirmatory or negative result.
5. **20x / 80x / 224x are not current measurements.** 20x is an invalid multiplication;
   80x is a narrow SmolLM2 pilot ratio; 224x is an assumption-only projection.
6. **The r3 hashes were CRLF-based.** The committed blobs are LF; the r3 preflight
   failed on the line-ending difference. The r4 tag fixed provenance with `.gitattributes`
   and LF hashes, and its Kaggle preflight passed. Training then stopped on Kaggle's
   `torchao 0.10.0` before LoRA injection; the r5 tag removes that unused package and
   requires it to be absent. No check may be bypassed.
7. **No format-value claim is allowed** without C beating D on the paired CI and passing
   the strict-accuracy non-inferiority gate.

## Documents intentionally not rewritten

Archived v2.1 documents and old model cards keep their historical text. They are
banner-marked rather than silently rewritten, so the provenance of the old claims is
visible. Grant and pitch documents are marked “do not submit as-is” rather than being
laundered into current applications.

## Verification

```bash
# Re-run the model-free suite.
python -m unittest discover -s tests

# Re-list first-party Markdown files.
python - <<'PY'
from pathlib import Path
skip={'.venv','vendor','scratch','.pytest_cache','__pycache__','.git'}
for p in sorted(Path('.').rglob('*.md')):
    if set(p.parts) & skip: continue
    print(p.as_posix())
PY
```
