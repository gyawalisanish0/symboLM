# Evidence Status

**As of:** 2026-10-24  
**Branch:** `research/rescue-v3`  
**Current phase:** **r6 SFT is complete and gate-eligible**; r5 is superseded; **no v3
benchmark result exists yet**<br>
**Primary protocol:** [`PREREG_VALUATION_V2.md`](../PREREG_VALUATION_V2.md)  
**Canonical framing:** [`docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`](TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md)

This page is the short answer to “what can we actually claim?” The long audit trail is in
[`EVIDENCE_TRUTH_TABLE.md`](../EVIDENCE_TRUTH_TABLE.md),
[`RESULTS_SUMMARY.md`](../RESULTS_SUMMARY.md), and
[`docs/RESEARCH_REDIRECT.md`](RESEARCH_REDIRECT.md).

---

## 1. The claim hierarchy

1. **Primary:** token efficiency of correct reasoning — generated tokens per correct answer,
   and dynamic joules per correct answer when NVML is available.
2. **Gate:** strict declared-answer accuracy must be non-inferior to the control. The
   preregistered margin for C versus D at 256 tokens is −2.0 percentage points.
3. **Secondary:** format validity, terminal-answer rate, EOS rate, OAA/AUC_OAA, TokenEff,
   overthinking score, and budget-stress diagnostics.
4. **Not a result:** a partial, quota-stopped run; a post-hoc slice; a lenient score; a
   projection; a biological metaphor.

The v3 benchmark has not been run. The r6 adapter is gate-eligible, but no current claim
about C-versus-D format value or token-efficiency superiority is allowed.

---

## 2. Status table

| Claim | Status | Evidence | Token-efficiency reading |
|---|---|---|---|
| A short bypass reflex can answer a subset of atomic facts | **Supported as a narrow mechanism** | Strict v2.1 rescore: 32/1,000 passes were atomic bypass; atomic subset 32/40 | Useful for a factual reflex, not general reasoning. The old 3-token bypass is a pilot observation, not a v3 result. |
| The old 1,000-question benchmark was mostly truncated | **Supported** | 917/1,000 (91.7%) hit the 512-token limit; 823 looped traces | The old model was not solving the benchmark; its token counts are not a valid efficiency comparison. |
| Strict v2.1 accuracy | **3.2% (32/1,000)** | `eval/RESCORING_AUDIT_REPORT.md` | The old 50.5% / 67.6% numbers are lenient artifacts. Mean generated length was 494.5 tokens. |
| Task 2 causal ablation: C vs corrupted plan | **C 10.0%, D 9.0%, +1.0 pp** | `eval/ABLATION_REAL_REPORT.md`; SmolLM2-1.7B | CI includes zero; not evidence of format value. C 125.0 tokens vs D 136.4 tokens. |
| Valuation: C vs router-only D | **C 9.5%, D 10.5%** | `VALUATION.md`; SmolLM2-1.7B | Paired 95% CI `[-844.57, +861.30]` includes zero; C 141.3 tokens / 1,265.75 J per correct vs D 159.8 tokens / 1,301.41 J per correct. No format value. |
| Sandbox arm E | **23.0% vs A 7.0%** | `VALUATION.md`; SmolLM2-1.7B | E 120.2 tokens / 444.30 J per correct vs A 167.6 tokens / 2,052.41 J per correct. Positive engineering pilot; it is tool use, not format value. |
| Answer-only compression | **20.67×** | Task 2; 8.1 vs 167.6 mean tokens | Cheap, but only 15% strict accuracy. A floor, not a reasoning solution. |
| Full six-arm 200 × 4 run | **Incomplete** | 2,119/4,800 cells; 1,955 rows saved | Stopped mid-grid by Kaggle quota/infrastructure. 37.0% / 39.7% / “Format Value Confirmed” are retracted. |
| Old Arm C efficiency win | **Invalid** | All 800 C traces lack `<ans>`; 296 old “correct” C rows used `deduce_equality_rhs`; 692 contain `</reg:verify>` | The old C token count cannot be used to claim an efficiency win. |
| MATH500 100% | **Retracted** | n=5 anecdote | Not a benchmark. |
| Cross-family LLaMA transfer | **Retracted as a universality claim** | n=5 probe; separate SmolLM2 200-problem run | Not evidence of tokenizer/architecture invariance. |
| Sub-50-token reasoning | **Not supported** | v2.1 averaged 494.5 tokens | The old 253.8-token figure is not the run mean. |
| r6 SFT artifact | **Gate-eligible** | `results/r6_closeout/R6_CLOSEOUT.json` | 777 steps / 3 epochs; final eval loss `0.00748848170042038`; strict generation and package gates passed. This is an artifact status, not a benchmark result. |
| r6 CPU smoke test | **Diagnostic only; 2/2 valid** | `results/r6_closeout/CPU_PROBE.json` | Two greedy probes on a Ryzen 7 5700G, bf16, 8 threads: `295` (114 tokens, 35.8 s) and `26` (13 tokens, 8.4 s). Both emitted EOS, valid format, strict correct. n=2; no accuracy, token-efficiency, or latency claim. |
| v3 register value | **Unknown / untested** | r6 adapter is gate-eligible; no v3 benchmark run | The preregistered test has not been executed. r5 is ineligible after a generation-gate failure. |
| v3 token-efficiency value | **Unknown / untested** | r6 adapter is gate-eligible; no v3 benchmark run | `tokens_emitted` is total generation; thinking-token split is not yet logged. |

---

## 3. What the SmolLM2 pilot actually says about efficiency

The SmolLM2-1.7B pilot is **Tier B**: measured, but not the flagship and not the v3
protocol. Its useful signal is not “registers win.” It is:

| Arm | Strict accuracy | Mean tokens | Dynamic J/correct | Reading |
|---|---:|---:|---:|---|
| A: ordinary CoT | 7.0% | 167.6 | 2,052.41 | Expensive and weak on this base model |
| B: answer-only | 15.0% | 8.1 | 25.84 | Cheapest, low accuracy; the reflex floor |
| C: full registers | 9.5% | 141.3 | 1,265.75 | Lower cost than A, but not better than D |
| D: router-only control | 10.5% | 159.8 | 1,301.41 | The correct control; C-vs-D CI includes zero |
| E: CoT + Python sandbox | 23.0% | 120.2 | 444.30 | Strongest engineering signal; external tool use |

**Correct conclusion:** routing and tool use can reduce cost on this small base model, but
the register format has **not** been shown to add value over routing. The C-versus-D
difference is not statistically distinguishable from zero.

The pilot does not measure `thinking_tokens` separately. Its `Mean tokens` is total
generation, so it cannot support a claim about “thinking-token compression” specifically.

---

## 4. What the partial six-arm run actually showed

The old run stopped at **2,119 / 4,800** cells. A post-hoc strict rescore on the 57 cells
common to all arms at budget 2048 found:

| Arm | Strict accuracy |
|---|---:|
| C: full registers | 26.3% |
| D: router-only | 1.8% |
| F: vanilla base | 42.1% |
| A: ordinary CoT | 0.0% |
| B: answer-only | 29.8% |
| E: sandbox | 1.8% |

This is **n=57, post-hoc, and not the preregistered 200 × 4 experiment**. It is a
hypothesis generator only. It is not a token-efficiency result: the C traces lack `<ans>`
and the old token accounting is not the v3 path. The vanilla base F being stronger than
the register arm on these cells is exactly the kind of result the old headline concealed.

### How to describe the stop

> “The old six-arm run stopped at 2,119 of 4,800 cells because the free Kaggle quota and
> infrastructure limit ended the session. The missing cells are incomplete; the saved cells
> used a scoring path that is not the v3 strict scorer. The run is retained as an
> engineering diagnostic, not as a confirmatory or negative result.”

Do not say the run “failed and therefore disproved” the thesis. Do not say it “completed”
or that the partial numbers are an endpoint result.

---

## 5. Receipts and hashes

| Artifact | Hash / value |
|---|---|
| Canonical SFT source | `724581AD5F3DEE4A1A4D2C321C167E13A016F64E023492AF6460D2BF04A3B7A7` |
| Canonical train | `BB50D9CBD56B0D2B515D7B922DF10578F587AA60358B0FF87A11821C4D3D94DC` |
| Canonical validation | `3FCFCC4141522062087D0F35FDF2C54226D454A1A71FE59BEA26F42636FDEDCD` |
| Held-out v2 benchmark | `F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707` |
| Line endings | LF, enforced by `.gitattributes` |
| Old clean benchmark, CRLF-era | `8369BD8CAF919F672D8EC87FFF2295FF6380AE9E451F668F9EEEE908F83CF07B` |
| Old clean benchmark, LF (old Kaggle manifest) | `575B247066B65E5B8514990280C534BAA9448014E100ADCFE67D385E43B9AA8E` |
| Ground-truth correction | `MY2021_001 = 169.5` (original 169 was wrong) |
| SFT targets | 4,600 total: 4,200 reasoning + 400 bypass |
| SFT split | 4,140 train / 460 validation |
| Benchmark rows | 200 rows, 173 unique prompts, 20 duplicate groups (27 redundant rows) |

**Provenance note.** The r3 data blobs were LF, but the r3 tag did not force LF checkout.
Windows materialized CRLF and the manifest recorded CRLF hashes, so the Linux kernel failed
preflight. The r4 tag (`sft-v3-20261023-r4`) adds `.gitattributes` LF rules, normalizes the
working files, and records the LF hashes above. The rows and targets are unchanged. Do not
bypass the check.

**r4 run outcome, r5 generation failure, and r6 token-delta repair.** The r4 Kaggle preflight
passed the tag/commit, canonical hashes, all 4,140 training rows, all 460 validation rows,
tokenizer, label masks, terminal answers, and EOS checks. Training then stopped before an
optimizer step: Kaggle supplied `torchao 0.10.0`, while PEFT 0.21.2 requires
`torchao >= 0.16.0` for its LoRA dispatch. This is a runtime-packaging failure, not a data
failure or a scientific negative result. The r5 revision (`sft-v3-20261023-r5`) removed the
unused `torchao` before LoRA injection, recorded `"torchao": "absent"` in the run manifest,
and required that state in the package gate.

The r5 run then completed 777 steps / 3 epochs with final eval loss
`0.2768503725528717`, but **failed the strict generation gate**. The reasoning probe
`concept_poly_30` declared no answer (`format_reason: loop_marker:letterSpacing`); the bypass
probe `concept_bypass_183` returned `100` correctly. The saved r5 adapter had byte-identical
frozen `<ans>` (151665), `</ans>` (151666), and base `answer` (9217) rows in both the input
embedding and LM head. Frozen tied logits cannot be separated by greedy decoding, so the
model emitted `answer62answer` instead of `<ans>62</ans>`. This is an adapter bug, not a
scientific result.

The r6 revision (`sft-v3-20261023-r6`) keeps the full resized `modules_to_save` matrices
frozen but trains small deltas on the 47 genuinely added tokenizer rows (`151665`-`151711`),
merges them before save, and makes both the training and package gates reject exactly tied
marker rows. The r6 run completed 777 steps / 3 epochs with final eval loss
`0.00748848170042038` and passed the strict mixed-kind generation gate: `concept_poly_30`
returned `62` and `concept_bypass_183` returned `100`, both with valid format, a terminal
answer, and a real EOS. The package gate passed with `"torchao": "absent"`, merged token
deltas, and distinct terminal-marker rows. The adapter SHA-256 is
`204823BC216BBF7329D6C7C5159ACF8F6688FD321AD1652757C2A9A9D066B31C`; the bundle SHA-256 is
`C711EF8BF119DC0B8DDF78BF09F6EDB2276996312901936C1EA67BED55FCC573`. Receipts:
`results/r6_closeout/`. No data, split, target, hypothesis, or endpoint changed. This is an
artifact gate result, not a v3 benchmark result.

---

## 6. What not to cite

Do not cite any of the following as a current result:

- 37.0% overall accuracy.
- 39.7% reflex accuracy or 822.6 J.
- 50.5% or 67.6% on the 1,000-question benchmark.
- 100% on MATH500.
- “Format Value Confirmed.”
- “Cross-family universality” from an n=5 probe.
- “Sub-50-token reasoning.”
- “1,955 complete evaluation records.”
- “20× compound compute” or “224× serving compression.”
- Any number produced by `deduce_equality_rhs`, trailing-number regexes, or an Arm C stop
  at `</reg:verify>`.
- Any claim that the model is “brain-like,” “biomimetic” in a mechanistic sense, or a
  sparse mixture of experts.

These are preserved in the archive for audit, not for promotion. The old README and
whitepaper are explicitly labeled historical.

---

## 7. What is pending

1. ✅ Freeze and commit `PREREG_VALUATION_V2.md` (tag `sft-v3-20261022-r3`; r2 is superseded).
2. ✅ Prepare the Kaggle SFT kernel at `experiments/kaggle/sft_v3/` with the r3 provenance and artifact fixes.
3. ✅ Apply the r4 LF provenance fix and create tag `sft-v3-20261023-r4`.
4. ✅ Validate the r4 preflight on Kaggle; it passed all data and tokenizer checks.
5. ✅ Complete r5 training (777 steps) and diagnose the strict generation-gate failure:
   frozen `<ans>`, `</ans>`, and `answer` output rows were byte-identical.
6. ✅ Implement the r6 token-delta repair, merge-before-save, and tied-row gate; create tag
   `sft-v3-20261023-r6`.
7. ✅ Run r6 training and pass the mixed-kind post-training generation check: 777 steps /
   3 epochs, final eval loss `0.00748848170042038`; probes returned `62` and `100`.
8. ✅ Package the r6 adapter and pass the package gate, including `"torchao": "absent"`,
   merged token deltas, and distinct terminal-marker rows.
9. ⛔ Run the 100 × 4 × 6 diagnostic. This requires a v3-compatible harness; the existing
   `eval/robust_eval.py` is the v2.1 six-arm path and must not be treated as v3-ready.
10. ⛔ Run the full 200 × 4 × 6 experiment (requires separate explicit approval).
11. ⛔ Report strict accuracy, tokens per correct answer, J/correct, format validity, and the
    preregistered C-vs-D decision.

Until step 11, the correct public statement is:

> “The old claims are retracted. The documentation audit, preregistration, and r4 provenance
> fix are complete. The r4 preflight passed but training stopped on a Kaggle `torchao`
> packaging conflict. The r5 run trained but failed its generation gate because the saved
> terminal-answer rows were exactly tied. The r6 run completed and passed both the
> mixed-kind generation gate and the package gate; its adapter is gate-eligible. No v3
> benchmark result exists yet, and the 4,800-cell experiment has not been run.”

---

## 8. Verification commands

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

Expected v2 LF hash:

```text
F08754D49CFA43A8AFA64382A49C21916A809756B1089DA22827D79819771707
```

The old CRLF-era hash was `95E2B237…`. The difference is line endings, not rows.

---

## 9. Interpretation rules

- Token efficiency is the primary lens; strict accuracy is the non-inferiority gate.
- Strict scores decide. Lenient scores are diagnostic only.
- A confidence interval that includes zero is not evidence of value.
- An incomplete run is not a result, in either direction.
- An n=5 probe is not a benchmark.
- A mechanism that works on atomic facts is not a general reasoning claim.
- A biological metaphor is not a biological mechanism.
- A projection is not a measurement.
- A negative result is allowed to be the final answer.
