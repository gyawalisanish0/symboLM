# v3 2,400-Cell Diagnostic: Completion and Gate Receipt

**Status:** completed and independently verified. **Diagnostic only**; not the
4,800-cell confirmatory run and not a benchmark result.<br>
**Run ID:** `robust_eval_20261031T151842Z`<br>
**Code commit:** `17d62d1bc8e26f6a02bee67e777ebf8984a0ca02`<br>
**Branch:** `research/rescue-v3`<br>
**Adapter:** r6, `204823BC216BBF7329D6C7C5159ACF8F6688FD321AD1652757C2A9A9D066B31C`<br>
**Dataset:** `F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707`<br>
**Energy path:** NVML dynamic power on Tesla T4<br>
**Verification receipt:** [`VERIFICATION.json`](VERIFICATION.json)

---

## 1. Completion

| Check | Result |
|---|---|
| Expected cells | 2,400 |
| Completed cells | 2,400 |
| Unique cells | 2,400 |
| Missing cells | 0 |
| Extra cells | 0 |
| Wrapper status | `completed` |
| Resume source | private checkpoint dataset, 2,287 prior cells |
| Independent verifier | `status: verified` |

The independent verifier reloaded the frozen benchmark, recomputed the
deterministic subset, checked the completed Cartesian grid, checked every row's
provenance fields, compared the contract hashes against the committed git blobs
at `17d62d1`, and cross-checked the wrapper receipt against the artifacts. It
does not trust the receipt alone.

Verifier checks passed:

`no_wrapper_failure`, `manifest_present`, `manifest_identity`,
`contract_hashes`, `subset_recomputed`, `expected_grid`,
`complete_unique_grid`, `wrapper_receipt`, `report_boundary`,
`local_adapter_hash`.

---

## 2. Gate G1

Gate G1 requires no more than 10% Arm C format-invalid cells.

| Arm C check | Count |
|---|---:|
| Cells | 400 |
| Format valid | 400 |
| Format invalid | 0 |
| EOS emitted | 400 |
| Terminal answer declared | 400 |
| Truncated | 0 |

**Gate G1 passes:** Arm C format-invalid rate is **0.0% (0/400)**, far below the
10% threshold. The four budget rows are identical for all 100 Arm C problems,
so the effective number of unique Arm C generations is 100, not 400; the
format result is 100/100 either way.

Other arms show the expected budget truncation at low budgets. That is not a
Gate G1 failure, which is defined on Arm C.

---

## 3. Descriptive observations

These are diagnostic observations, not confirmatory claims and not a benchmark
result. The subset is 100 problems; the full run is 200 problems.

| Arm | Budget 256 | Budget 512 | Budget 1024 | Budget 2048 |
|---|---:|---:|---:|---:|
| A: ordinary CoT | 16/100 | 51/100 | 64/100 | 64/100 |
| B: answer-only | 29/100 | 64/100 | 77/100 | 77/100 |
| C: full registers | 62/100 | 62/100 | 62/100 | 62/100 |
| D: router-only control | 19/100 | 54/100 | 67/100 | 67/100 |
| E: CoT + Python sandbox | 20/100 | 34/100 | 39/100 | 39/100 |
| F: vanilla base | 2/100 | 31/100 | 48/100 | 74/100 |

Arm C mean generation is **111.3 tokens** and is budget-invariant: it emits EOS
before the 256-token limit on every problem. Its 62/100 strict accuracy is the
same row repeated across all four budget columns. Arm C is token-efficient but
is **not** accuracy-non-inferior to B, F, or D at the higher budgets in this
subset.

The report's preregistered decision-rule table prints "format value supported"
at 256 and 512 and "no format value" at 1024 and 2048. That table is a
diagnostic computation on 100 problems, not the confirmatory 4,800-cell
decision. Do not promote it to a v3 result.

### Repeated budget rows

Greedy generation that emits EOS before the budget produces the same text at
multiple budgets. The number of problems with one unique generated text across
all four budgets is:

| Arm | Problems with one unique text across budgets |
|---|---:|
| A | 33/100 |
| B | 44/100 |
| C | 100/100 |
| D | 33/100 |
| E | 63/100 |
| F | 2/100 |

Budget-level rows are therefore not independent observations for the arms that
stop early. The primary preregistered endpoint uses the 256-token budget only;
secondary budget comparisons must account for this repetition.

---

## 4. Duplicate prompts and contamination

The report records **432/2,400 duplicate-prompt rows**. This is a dataset
annotation, not 432 duplicate result cells:

- 18 of the 100 selected problems carry `duplicate_prompt: true`;
- 18 problems x 6 arms x 4 budgets = 432 rows;
- the selected problems form 16 rendered-prompt-hash groups containing more
  than one problem ID, with at most two IDs per group.

The report's contamination table shows a 0.0 pp all-versus-clean gap for every
arm and budget. The selected subset contains **0** of the 67 IDs flagged by
`eval/contamination_audit_results.json`; the zero gap therefore means no
selected problem is in that index. It does not mean contamination has been
exhaustively ruled out. The duplicate-prompt flags and the contamination index
are separate mechanisms.

---

## 5. Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `EVAL_RUN_MANIFEST.json` | `44C5F0FA13A04AEA92D2CFA2DE67E949DEA3901F7EC5E940951B62D65488FB64` |
| `robust_eval_results.jsonl` | `693A1EDC1C43414B996D24BF66347B6FC9405E37833FBFF798F2A08B1D44AABA` |
| `ROBUST_EVAL_REPORT.md` | `6FA4A0591510B4B4FA3A43DA6CD18A3DECAC323207DFEFB861A50A747F98CE35` |
| `WRAPPER_RECEIPT.json` | see `SHA256SUMS.txt` |
| `VERIFICATION.json` | see `SHA256SUMS.txt` |
| receipts tarball | see `SHA256SUMS.txt` |

The manifest and report hashes match the wrapper receipt. The verifier also
checked the frozen adapter artifact and config hashes.

---

## 6. Claim boundary

Allowed:

> "The 2,400-cell v3 diagnostic completed with all 2,400 unique cells, a
> completed wrapper receipt, and an independent verification receipt. Arm C
> passed Gate G1 with 400/400 format-valid, EOS-valid, terminal-answer cells and
> zero truncations. This is a pipeline and format gate, not a benchmark result."

Not allowed:

- calling this the 4,800-cell confirmatory result;
- reporting Arm C's 62/100 as a v3 benchmark accuracy;
- claiming token-efficiency, latency, energy, or mobile superiority from this
  diagnostic;
- using these outcomes to tune r7 or to rescue a null v3 result;
- treating the 432 duplicate-prompt rows as duplicate result cells;
- treating the 0.0 pp contamination gap as proof that the benchmark is clean.
