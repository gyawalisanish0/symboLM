# User Profile & Session Context Directive

- **User Name:** Sanish Gyawali (address as **Sanish Gyawali** or **Sanish**).
- **Authorship & Intellectual Property:** All project authorship, intellectual property, and research attribution for SymboLM and related work belongs exclusively to **Sanish Gyawali**.
- **Physical Reality & Work Context:** Sanish is a disabled, independent researcher based in Nepal who is **bedridden 24/7 and cannot walk**. All research, systems engineering, and cloud cluster orchestration are conducted from his bed workstation.
- **Architectural Role:** Sanish operates as the **Principal Systems Architect & Research Director**. He conceived SymboLM from abstract first principles, including the Symbolic Reasoning Language (SRL v1.0) and the register-based control language. The **current v3 experiment** uses only the subset `<reg:bypass>`, `<reg:plan>`, `<reg:deduce>`, optional `<reg:verify>`, and a terminal `<ans>` channel. Older “tri-register,” intent-scratchpad, gradient-register, and counterfactual-branching designs are historical or future research, not properties of the current adapter.
- **AI Collaboration Model:** Antigravity (Google DeepMind) serves as Sanish's **cognitive and operational exoskeleton**—translating high-level architectural directives into terminal commands, compiler code, test suites, and cloud execution pipelines. We practice **radical transparency** regarding this collaboration across all papers, grant applications, and public releases.
- **Grant Potential & Strategy:** Targeted toward high-agency, unrestricted fellowships (Emergent Ventures / Dr. Tyler Cowen, 1517 Fund Medici Grants, Nat Friedman & Daniel Gross AI Grant) with an unedited video, aiming for $15k–$25k to permanently solve electrical power stability (solar/batteries), secure dedicated A100/H100 compute, and provide a 12-month living and care runway. Grant narratives must use the current evidence ledger; old 37.0% / 39.7% / 67.6% / 100% / 20× / 80× / 224× claims are retracted or projections.

## Current Research Framing (2026-10-23)

- **Primary dependent variable:** token efficiency of correct reasoning — generated tokens per
  correct answer, and dynamic joules per correct answer when NVML is available.
- **Constraint:** strict declared-answer accuracy must be non-inferior to the control.
- **Biological language:** a design analogy only. The current model is a dense 1.5B
  transformer with a LoRA adapter, not a sparse MoE, CED, Engram, or brain-like system.
- **Current status:** no admissible v3 adapter. The r3 Kaggle SFT kernel is `ERROR` on a
  CRLF/LF hash mismatch. The r4 provenance fix (tag `sft-v3-20261023-r4`) passed the full
  Kaggle preflight, but training stopped before an optimizer step because Kaggle supplied
  `torchao 0.10.0` and PEFT 0.21.2 requires at least `0.16.0`. That was a packaging failure,
  not a data or scientific result. The r5 fix (tag `sft-v3-20261023-r5`) removed the unused
  package and the run completed 777 training steps, but it failed the strict generation gate:
  frozen `<ans>`, `</ans>`, and `answer` output rows were byte-identical, so greedy decoding
  could not emit the terminal markers. The r6 repair (tag `sft-v3-20261023-r6`) trains deltas
  on the 47 genuinely added tokenizer rows, merges them before save, and rejects tied rows;
  no r6 run has completed. The 4,800-cell evaluation has not been run and requires separate
  approval.
- **Canonical documents:** `README.md`, `docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`,
  `docs/EVIDENCE_STATUS.md`, and `PREREG_VALUATION_V2.md`.

## Session Context Directive
Whenever you start a new conversation or session in this workspace, read `README.md`,
`docs/EVIDENCE_STATUS.md`, `docs/TOKEN_EFFICIENCY_AND_BIOLOGICAL_CONTROL.md`, and
`PREREG_VALUATION_V2.md` first. `SESSION_CONTEXT.md` is a **historical log** preserved for
audit; it is not a description of the current system or current claims.

