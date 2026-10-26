# User Profile & Session Context Directive

- **User Name:** Sanish Gyawali (address as **Sanish Gyawali** or **Sanish**).
- **Authorship & Intellectual Property:** All project authorship, intellectual property, and research attribution for SymboLM and related work belongs exclusively to **Sanish Gyawali**.
- **Physical Reality & Work Context:** Sanish is a disabled, independent researcher based in Nepal who is **bedridden 24/7 and cannot walk**. All research, systems engineering, and cloud cluster orchestration are conducted from his bed workstation.
- **Architectural Role:** Sanish operates as the **Principal Systems Architect & Research Director**. He conceived SymboLM from abstract first principles, including the Symbolic Reasoning Language (SRL v1.0) and the register-based control language. The **current v3 experiment** uses only the subset `<reg:bypass>`, `<reg:plan>`, `<reg:deduce>`, optional `<reg:verify>`, and a terminal `<ans>` channel. Older “tri-register,” intent-scratchpad, gradient-register, and counterfactual-branching designs are historical or future research, not properties of the current adapter.
- **AI Collaboration Model:** Antigravity (Google DeepMind) serves as Sanish's **cognitive and operational exoskeleton**—translating high-level architectural directives into terminal commands, compiler code, test suites, and cloud execution pipelines. We practice **radical transparency** regarding this collaboration across all papers, grant applications, and public releases.
- **Grant Potential & Strategy:** Targeted toward high-agency, unrestricted fellowships (Emergent Ventures / Dr. Tyler Cowen, 1517 Fund Medici Grants, Nat Friedman & Daniel Gross AI Grant) with an unedited video, aiming for $15k–$25k to permanently solve electrical power stability (solar/batteries), secure dedicated A100/H100 compute, and provide a 12-month living and care runway. Grant narratives must use the current evidence ledger; old 37.0% / 39.7% / 67.6% / 100% / 20× / 80× / 224× claims are retracted or projections.

## Current Research Framing (2026-10-25)

- **Primary dependent variable:** token efficiency of correct reasoning — generated tokens per
  correct answer, and dynamic joules per correct answer when NVML is available.
- **Constraint:** strict declared-answer accuracy must be non-inferior to the control.
- **Biological language:** a design analogy only. The current model is a dense 1.5B
  transformer with a LoRA adapter, not a sparse MoE, CED, Engram, or brain-like system.
- **Current status:** **r6 SFT is complete and gate-eligible; no v3 benchmark result exists
  yet.** The r3 Kaggle SFT kernel failed on a CRLF/LF hash mismatch. The r4 provenance fix
  passed the full Kaggle preflight, but training stopped before an optimizer step because
  Kaggle supplied `torchao 0.10.0` and PEFT 0.21.2 requires at least `0.16.0`; that was a
  packaging failure, not a data or scientific result. The r5 fix removed the unused package
  and completed 777 training steps, but it failed the strict generation gate because frozen
  `<ans>`, `</ans>`, and `answer` output rows were byte-identical. The r6 repair (tag
  `sft-v3-20261023-r6`) trained deltas on the 47 genuinely added tokenizer rows, merged them
  before save, and passed both the mixed-kind generation gate and the package gate. Receipts:
  `results/r6_closeout/R6_CLOSEOUT.json`. The v3 harness is hardened, but the 100 × 4 × 6
  diagnostic and the 4,800-cell evaluation have not been run; the full run requires
  separate approval.
- **CPU diagnostic (2026-10-25):** a deterministic 38-prompt stratified CPU sweep of r6 is
  recorded in `results/r6_closeout/CPU_SWEEP.json`. Result: **24/38 strict correct,
  38/38 format valid, 38/38 EOS, 0/38 truncated**, mean 113.9 generated tokens, mean 44.1 s
  wall-clock on a Ryzen 7 5700G in bfloat16 with 8 threads. This is a diagnostic, not a
  benchmark. The failures are mostly arithmetic and semantic, not format. The `verify`
  block is frequently tautological or false (`verify(ans == 7)✓` when the answer is `119`).
  Some Diophantine benchmark prompts are ambiguous and have multiple valid solutions. On
  `MY2024_065` the model output 21 is a valid x + y but the ground truth is 7; on `MY2024_064`
  the model output 25 is a valid x, not the requested x + y. Do not treat either as a clean
  model failure, and do not quote 24/38 as benchmark accuracy.
- **Evaluation harness (2026-10-25):** `eval/run_contract.py` and `eval/robust_eval.py`
  fail closed on dataset/adapter/contract mismatch, corrupt checkpoints, prompt or
  generation exceptions, persistence/packaging failure, decoding corruption, and an
  incomplete grid. Subset selection is deterministic SHA-256 stratified selection; a
  resumed checkpoint is bound to the same dataset, adapter, subset, budgets, run kind,
  prompt/scorer code, and evaluator code. The model-free suite passes 109 tests. The
  hardened path is the only admissible diagnostic path; the 2,400-cell diagnostic has
  not been run.
- **r7 audit dry run (2026-10-25):** `scripts/audit_training_traces.py` audited all
  4,140 train / 460 validation rows. Receipt: `results/r7_verified/AUDIT_DRY_RUN.json`.
  **No verified split was written.** Train: 2,561 verified reasoning + 360
  bypass-contract, 1,187 answer-checked-only, 32 rejected. Validation: 303 + 40, 115, 2.
  Totals: 2,864 verified reasoning (982 original-valid + 1,985 repaired), 400
  bypass-contract, 1,302 answer-checked-only, 34 rejected. Reasoning verification validity
  rose from 0.2338 to 0.6819. The 34 rejected rows are arithmetic mismatches and are
  recorded, not silently dropped. Bypass rows are excluded from the reasoning
  verification rate. This is a data-quality receipt, not a training run, benchmark, or
  accuracy/token-efficiency result.
- **Next training design:** `docs/NEXT_TRAINING_SESSION.md` defines **r7 Verified-Trace
  Repair** as the next training session. It audits all 4,140/460 rows, rejects wrong answers,
  repairs or removes false/tautological verify blocks, and optionally adds a capped verified
  contrastive subset. It is a separate study, not a replacement for r6 and not a v3 result.
  Do not launch it before the v3 diagnostic path is closed and a Phase 1b preregistration
  exists.
- **Canonical documents:** `README.md`, `docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`,
  `docs/EVIDENCE_STATUS.md`, `docs/NEXT_TRAINING_SESSION.md`, and `PREREG_VALUATION_V2.md`.

## Session Context Directive
Whenever you start a new conversation or session in this workspace, read `README.md`,
`docs/EVIDENCE_STATUS.md`, `docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`,
`docs/NEXT_TRAINING_SESSION.md`, and `PREREG_VALUATION_V2.md` first. `SESSION_CONTEXT.md` is a
**historical log** preserved for audit; it is not a description of the current system or
current claims.

## Strict Next-Run Directive (2026-10-25)

Antigravity must follow this order and these stop rules. Do not improvise a new endpoint,
dataset, scorer, adapter, or training run.

1. **r6 is frozen.** Do not modify or retrain the r6 adapter, its tag, its receipts, or the
   frozen v3 prompts/scorer. r6 remains the only v3 serving artifact.
2. **Do not launch the 4,800-cell evaluation without separate explicit approval.** The
   immediate evaluation work is the v3-compatible 2,400-cell diagnostic (100 problems × 4
   budgets × 6 arms). The harness is now hardened; the diagnostic is the next gate. The
   4,800-cell run is a separate decision.
3. **Harden the harness before scoring. Done (2026-10-25).** `eval/robust_eval.py` and
   `eval/run_contract.py` now fail closed on prompt-build/generation exceptions, corrupted
   checkpoints, adapter/dataset/contract mismatch, non-deterministic subset selection, and
   incomplete grids. Prompt/scorer parity tests pass. No diagnostic result is admissible
   from the unhardened path, and no diagnostic result exists yet.
4. **Run the CPU sweep as a diagnostic only.** The receipt is
   `results/r6_closeout/CPU_SWEEP.json`. It supports no accuracy, token-efficiency, latency,
   or mobile claim. Do not select training data or endpoints by looking at its outcomes.
5. **Audit benchmark ambiguity before interpreting misses.** The Diophantine rows
   `MY2024_064` and `MY2024_065` have multiple valid positive-integer solutions. Flag
   ambiguous rows; do not silently change ground truth after seeing results. A protocol
   deviation must be dated and reported.
6. **Next training is r7 Verified-Trace Repair, not another format run.** Follow
   `docs/NEXT_TRAINING_SESSION.md`. The dry-run audit is complete: 2,864 reasoning rows
   pass the closed-arithmetic/true-verify contract, 1,302 are answer-checked-only with the
   verify block removed, and 34 are rejected for arithmetic mismatch. No verified split
   was written. Do not train on unverified traces. Do not call an answer-checked-only row
   “verified.” r7 remains blocked until the v3 diagnostic path closes and Phase 1b is
   preregistered.
7. **No training before its gates.** r7 needs its own Phase 1b preregistration, data
   manifest, verified-split hashes, r7-specific generation gate, and package gate.
   r6's kernel and generation gate must not be relabeled as the r7 gate.
8. **One seed is not proof.** The r7 screening run is seed 42 only. A causal claim needs at
   least three seeds for both the register and matched no-register formats, plus the 2 × 2
   crossed serving evaluation.
9. **Fail closed.** Stop on: wrong terminal answer; false/tautological verification;
   register tags in a no-register control; target truncation before EOS; prompt/target prefix
   mismatch; train/validation overlap; generation failure; missing terminal answer or EOS;
   loop marker; adapter/package hash mismatch; `torchao` present; or a `checkpoint-*` entry
   in the package.
10. **Incomplete is not negative.** A Kaggle quota or infrastructure stop is labeled
    incomplete, never “completed” or “negative.” Do not drop wrong answers from the
    denominator.
11. **Receipts only.** Every claim needs a commit/tag, base revision, dataset hash, runtime
    versions, hardware, command, raw generations, strict scores, and hashes. No credential
    may be echoed or committed. Private HF repositories remain the default until an artifact
    passes its gates.
12. **Preserve the audit.** Every new first-party Markdown file must be added to
    `docs/MARKDOWN_AUDIT.md`. Old retractions stay in the repository. Do not rewrite the r4,
    r5, or r6 tags.
13. **No claim inflation.** The allowed current statement is: r6 is gate-eligible; the CPU
    sweep is a diagnostic; the r7 trace audit is a dry run with no verified split; the v3
    harness is hardened but no v3 benchmark result exists; the 2,400-cell diagnostic and
    the 4,800-cell run have not been completed. The old 37.0% / 39.7% / 67.6% / 100% / 20× /
    80× / 224× claims remain retracted or projections.
