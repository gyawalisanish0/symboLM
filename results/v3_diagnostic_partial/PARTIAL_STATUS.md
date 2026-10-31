# v3 2,400-Cell Diagnostic — Partial Audit

> **STATUS: INCOMPLETE — NOT A RESULT.**
>
> This report describes a run stopped by Kaggle's infrastructure limit. It is
> not a completed diagnostic, not a negative result, and not a benchmark,
> token-efficiency, latency, energy, or mobile claim. Do not use the partial
> accuracy or token tables for any decision.

## Run identity

| Field | Value |
|---|---|
| Run ID | `robust_eval_20261031T022945Z` |
| Run kind | `diagnostic` |
| Repository commit | `17d62d1bc8e26f6a02bee67e777ebf8984a0ca02` |
| Kaggle status | `CANCEL_ACKNOWLEDGED` |
| Expected cells | 2,400 |
| JSONL cells | **2,287** |
| Unique cells | **2,287** |
| Missing cells | **113** |
| Duplicate cells | 0 |
| Wrapper receipt | **Absent** |
| Classification | **Incomplete, infrastructure-stopped** |

## What survived

- `robust_eval_results.jsonl`: 2,287 unique cells, SHA-256
  `1A151C232CFF5B63D4A48AF5EC3C862FF51529DE3B7D57232457CAD971183132`.
- `EVAL_RUN_MANIFEST.json`: SHA-256
  `13F348A65FE376E8E5AD8DC2FB3C1FB39F52844BD10E26353CBC263A7EC8C522`.
- `ROBUST_EVAL_REPORT.md`: generated at 2,280 cells; it is stale relative to
  the JSONL and is reproduced only as an audit artifact.
- Checkpoint tarball:
  `symbolm_robust_eval_receipts_robust_eval_20261031T022945Z.tar.gz`
  (contains the 2,280-cell checkpoint).

The dataset and adapter hashes in the manifest match the frozen values. The
missing cells are concentrated at the end of the deterministic problem order;
there are no extra or duplicate cells.

## Safe observations from completed cells only

- Arm C completed cells: **380 / 400** expected Arm C cells.
- Of those completed Arm C cells: **380 format-valid**, **380 EOS-emitted**,
  **0 truncated**.
- Gate G1 is **not evaluated** because the diagnostic is incomplete and 20 Arm C
  cells are missing. The completed-cell observation above is not a gate pass.
- No strict-accuracy, token-efficiency, joules-per-correct, or Arm C vs D claim
  is made.

## Where the artifacts are

Local download directory:

`C:\Users\user\AppData\Local\Temp\opencode\v3_diagnostic_output\v3_diagnostic\`

Files:

- `ROBUST_EVAL_REPORT.md` — partial report, 2,280 cells
- `EVAL_RUN_MANIFEST.json` — run identity and hashes
- `robust_eval_results.jsonl` — 2,287 cells
- `..\symbolm_robust_eval_receipts_robust_eval_20261031T022945Z.tar.gz` —
  2,280-cell checkpoint

## Next action

A resume kernel is running. It restores the exact 2,287-cell checkpoint,
evaluates only the 113 missing cells, then reruns the full 2,400-cell grid and
receipt checks. The 4,800-cell confirmatory run remains blocked.
