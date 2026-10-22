# Evidence Status

**As of:** 2026-10-22  
**Branch:** `research/rescue-v3`  
**Current phase:** clean pipeline and preregistration; **no v3 GPU result yet**  
**Primary protocol:** [`PREREG_VALUATION_V2.md`](../PREREG_VALUATION_V2.md)

This page is the short answer to “what can we actually claim?” The long audit trail is in [`EVIDENCE_TRUTH_TABLE.md`](../EVIDENCE_TRUTH_TABLE.md), [`RESULTS_SUMMARY.md`](../RESULTS_SUMMARY.md), and [`docs/RESEARCH_REDIRECT.md`](RESEARCH_REDIRECT.md).

---

## 1. Status table

| Claim | Status | Evidence | What it means |
|---|---|---|---|
| A short bypass reflex can answer a subset of atomic facts | **Supported as a narrow mechanism** | Strict v2.1 rescore: 32/1,000 passes were atomic bypass; atomic subset 32/40 | Useful for a factual reflex, not general reasoning |
| The old 1,000-question benchmark was mostly truncated | **Supported** | 917/1,000 (91.7%) hit the 512-token limit; 823 looped traces | The old model was not solving the benchmark |
| Strict v2.1 accuracy | **3.2% (32/1,000)** | `eval/RESCORING_AUDIT_REPORT.md` | The old 50.5% / 67.6% numbers are lenient artifacts |
| Task 2 causal ablation: C vs corrupted plan | **C 10.0%, D 9.0%, +1.0 pp** | `eval/ABLATION_REAL_REPORT.md`; SmolLM2-1.7B | CI includes zero; not evidence of format value |
| Valuation: C vs router-only D | **C 9.5%, D 10.5%** | `VALUATION.md`; SmolLM2-1.7B | Paired 95% CI `[-844.57, +861.30]` includes zero; no format value |
| Sandbox arm E | **23.0% vs A 7.0%** | `VALUATION.md`; SmolLM2-1.7B | 444.30 vs 2,052.41 J/correct; positive engineering pilot, not format value |
| Answer-only compression | **20.67×** | Task 2; 8.1 vs 167.6 mean tokens | Cheap, but only 15% strict accuracy |
| Full six-arm 200 × 4 run | **Incomplete** | 2,119/4,800 cells | 37.0% / 39.7% / “Format Value Confirmed” are retracted |
| MATH500 100% | **Retracted** | n=5 anecdote | Not a benchmark |
| Cross-family LLaMA transfer | **Retracted as a universality claim** | n=5 probe; separate SmolLM2 200-problem run | Not evidence of tokenizer/architecture invariance |
| Sub-50-token reasoning | **Not supported** | v2.1 averaged 494.5 tokens | The old 253.8-token figure is not the run mean |
| v3 register value | **Unknown / untested** | No v3 GPU run | The preregistered test has not been executed |

---

## 2. Receipts and hashes

| Artifact | Hash / value |
|---|---|
| Canonical SFT source | `3F7DC86D5C6A0BF47F1C5E4E90F8F317727319BBE58C7F54C5C416D39EC7BD69` |
| Canonical train | `5D05003B495DE3322568C4B40713CCE376A1AC05AC9B85A50B579A68A49FF63E` |
| Canonical validation | `EB854AFF33DD31210D0B4BE61AF0165E5392F44B7B6D472C8C747161062E3EE3` |
| Held-out v2 benchmark | `95E2B237AE2B724DC0856AEE7220EBD79436F91630F77BBB3BAD0913127F10ED` |
| Old clean benchmark | `8369BD8CAF919F672D8EC87FFF2295FF6380AE9E451F668F9EEEE908F83CF07B` |
| Old Kaggle manifest | reported as `575B…`; not equal to the old preregistered hash |
| Ground-truth correction | `MY2021_001 = 169.5` (original 169 was wrong) |
| SFT targets | 4,600 total: 4,200 reasoning + 400 bypass |
| SFT split | 4,140 train / 460 validation |
| Benchmark rows | 200 rows, 173 unique prompts, 20 duplicate groups (27 redundant rows) |

The v2 benchmark is the old clean 200 plus one documented correction and duplicate-prompt flags. It is not a new benchmark; it is an audited version of the old one.

---

## 3. What the partial six-arm run actually showed

The old run stopped at **2,119 / 4,800** cells. A post-hoc strict rescore on the 57 cells common to all arms at budget 2048 found:

| Arm | Strict accuracy |
|---|---:|
| C: full registers | 26.3% |
| D: router-only | 1.8% |
| F: vanilla base | 42.1% |
| A: ordinary CoT | 0.0% |
| B: answer-only | 29.8% |
| E: sandbox | 1.8% |

This is **n=57, post-hoc, and not the preregistered 200 × 4 experiment**. It is a hypothesis generator only. It also shows why the full run matters: the vanilla base F was stronger than the register arm on these cells, which is exactly the kind of result the old headline concealed.

---

## 4. What not to cite

Do not cite any of the following as a current result:

- 37.0% overall accuracy.
- 39.7% reflex accuracy or 822.6 J.
- 50.5% or 67.6% on the 1,000-question benchmark.
- 100% on MATH500.
- “Format Value Confirmed.”
- “Cross-family universality” from an n=5 probe.
- “Sub-50-token reasoning.”
- “1,955 complete evaluation records.”
- Any number produced by `deduce_equality_rhs`, trailing-number regexes, or an Arm C stop at `</reg:verify>`.

These are preserved in the archive for audit, not for promotion. The old README and whitepaper are explicitly labeled historical.

---

## 5. What is pending

1. Freeze and commit `PREREG_VALUATION_V2.md`.
2. Run fresh SFT on the canonical 4,600-target split.
3. Pass the post-training generation check (terminal answer + EOS).
4. Run the 100 × 4 × 6 diagnostic.
5. Run the full 200 × 4 × 6 experiment.
6. Report strict accuracy, J/correct, format validity, and the preregistered C-vs-D decision.

Until step 6, the correct public statement is:

> “The old claims are retracted. The repaired pipeline and preregistration are ready. The v3 experiment has not been run.”

---

## 6. Verification commands

```bash
# Rebuild the canonical split and v2 benchmark.
python scripts/prepare_research_sft.py
python scripts/audit_ground_truth.py

# Model-free regression suite.
python -m unittest discover -s tests

# Check the v2 hash and duplicate flags.
python - <<'PY'
import hashlib, json
p = "data/held_out_eval_200_v2.jsonl"
print(hashlib.sha256(open(p, "rb").read()).hexdigest().upper())
rows = [json.loads(line) for line in open(p, encoding="utf-8")]
print(len(rows), len({r["prompt"] for r in rows}))
PY
```

Expected v2 hash:

```text
95E2B237AE2B724DC0856AEE7220EBD79436F91630F77BBB3BAD0913127F10ED
```

---

## 7. Interpretation rules

- Strict scores decide. Lenient scores are diagnostic only.
- A confidence interval that includes zero is not evidence of value.
- An incomplete run is not a result.
- An n=5 probe is not a benchmark.
- A mechanism that works on atomic facts is not a general reasoning claim.
- A negative result is allowed to be the final answer.
