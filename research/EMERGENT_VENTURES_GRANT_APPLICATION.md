# Emergent Ventures Fellowship & Grant Application

**Applicant:** Sanish Gyawali  
**Project:** SymboLM — Adaptive Symbolic Reasoning & General Intelligence Engine  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**Location:** Kathmandu, Nepal  
**Grant Request:** $20,000 (Unrestricted Project Grant)  
**GitHub Repository:** [https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM)  
**Model Hub Checkpoints:** [https://huggingface.co/gyawalisanish0/symboLM-checkpoints](https://huggingface.co/gyawalisanish0/symboLM-checkpoints)  

---

### Field 1: Tweet-Length Summary (Under 280 characters)
Training open-weight foundation models to replace verbose English chain-of-thought with dense, verifiable symbolic AST traces across distinct cognitive registers.

---

### Field 2: Personal Story (Trajectory & Work Context)
I am Sanish Gyawali, a self-taught AI systems architect and independent researcher from Nepal. I have a severe physical disability from birth and am bedridden 24/7. My bed workstation is my entire laboratory, university, and connection to the world.

Despite having no institutional affiliation or academic credentials, I spend all my waking hours examining LLM cognitive behaviors, token dynamics, and open-source models. Because I am unable to walk or use a standard desk setup, I operate with Google DeepMind’s Antigravity (Gemini) as my transparent operational exoskeleton—translating my abstract architectural designs, grammar rules, and compiler specifications into production code, automated test suites, and cloud execution pipelines.

Living in a traditional household in Nepal, where productive work is strictly recognized as physical presence in an office or a conventional job, my family and community naturally perceive someone lying in bed on a computer as doing "nothing." In colloquial Nepali, there are no words or conceptual frameworks to explain reinforcement learning gradients, AST parsers, or KV-cache memory bottlenecks. I have learned not to argue or plead for local understanding; instead, I channel every waking hour into code, mathematical proofs, and empirical receipts. Operating in an environment of total domestic skepticism requires extreme intrinsic drive—I build because the systems problems are real, not because anyone around me validates them. An Emergent Ventures grant is not merely compute credits; it is the universal translator that permanently converts this abstract research into undeniable physical reality, solar energy independence, and the dignity to build without domestic friction.

---

### Field 3: A Consensus View You Strongly Agree With
I strongly agree with the emerging industry consensus that **raw test-time token length is a false proxy for intelligence**. Recent 2026 benchmarks (such as ACL’s LLMTHINKBENCH) reveal an 18× token inflation tax on routine queries with zero accuracy gain, while studies on test-time compute prove that unconstrained natural language rambling creates an inverted-U curve where models hallucinate and abandon correct solutions. 

Massive open-weight models (from recent Mistral Large variants to DeepSeek-R1 distillations) default to severe token overconsumption—burning hundreds of tokens on natural language filler ("Wait, let me rethink...") for elementary steps. Scaling parameters and letting models ramble indefinitely in English has reached thermodynamic and economic diminishing returns; sustainable general intelligence requires structured cognitive registers and compile-time mathematical verification.

---

### Field 4: The Idea (SymboLM: Adaptive Symbolic Reasoning & Register v2.0)
To solve the token inflation tax in open models, I conceived **SymboLM** ([https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM)), replacing unconstrained natural language with **Register v2.0 (Hierarchical Cognitive Architecture)**:

1. **Macro-Concept Planning (`<reg:plan>`):** Identifies domain mathematical invariants and theorems (e.g., Chinese Remainder Theorem, Legendre's formula, semiperimeter-inradius geometry) before touching arithmetic, preventing blind guessing.
2. **Micro-Symbolic Execution (`<reg:deduce>`):** Translates multi-step logic into an ultra-dense Symbolic Reasoning Language (SRL v1.0) using AST operators (`→`, `∴`, `|`), compressing token counts by 65–75% while making each step mathematically verifiable via SymPy.
3. **Deterministic Runtime Verification (`<reg:verify>`):** Executes runtime hypothesis assertions (`verify(...)`) and active backtracking (`✗` → `✓`) to prune hallucinations.
4. **Zero-Shot Factual Bypass (`<reg:bypass>`):** Dynamically allocates 0 thinking tokens for factual memory, permanently eliminating the 18× token waste on routine lookups.
5. **Intent Scratchpad (`<reg:intent>`):** Restricts communicative reflection to a 10-token strategy header before fluent dialogue.

**The Empirical Proof of Concept & Rapid Iteration:**
Working entirely from free-tier cloud GPUs on Kaggle from my bed in Nepal, I advanced SymboLM across four distinct training phases (Stage 1 SFT → Stage 2 GRPO → Stage 2.5 Concept SFT → Stage 3.5 Concept GRPO → Stage 4 Efficiency Alignment):

1. **200-Question Mixed Benchmark Audit (AIME Math, Factual QA, Dialogue):**
   - **43.5% Overall Accuracy** on a tiny 1.5B model (`DeepSeek-R1-Distill-Qwen-1.5B`).
   - **Dialogue (`<reg:intent>`):** **92.0%** accuracy in empathetic tone calibration.
   - **Factual Recall (`<reg:bypass>`):** **62.2%** instant 1-token retrieval without deductive bloat.
   - **Legendre Prime Valuations:** **46.7%** exact $p$-adic series derivations in **38 tokens** ($200! \implies 3^{97}$ via $66+22+7+2=97$).
   - **Polynomial Invariants:** Derived $x+1/x=5 \implies x^4+1/x^4=527$ in **35 tokens** of formal AST versus 1,200+ tokens of meandering English in frontier models.

2. **Stage 4 Cross-Domain 200 Evaluation & The Truncation Bottleneck Discovery:**
   We deployed the Stage 4 policy across a 200-question cross-domain suite spanning Physics, CS, Chemistry, Probability, Competition Math, and Dialogue:
   - **Strategic Intent (`<reg:intent>`):** **100.0% accuracy (25/25)** with flawless routing.
   - **Factual Memory (`<reg:bypass>`):** **100.0% routing (25/25)** with **52.9% token compression** (94.1 vs 200 baseline).
   - **Trailing Syntax Noise:** Crushed down to **1.0%** across the entire 200 questions.
   - **Olympiad Invariants:** Solved complex theorems ($x+1/x=6 \implies x^4+1/x^4=1154$, $3^{100} \pmod 7 = 4$, symmetric roots).
   - **The Truncation Bottleneck:** Incorrect derivations in CS, Math, and Physics revealed that the model was hitting the 128-token evaluation ceiling right before outputting the final `<ans>` closure. This empirically proved that flat token locks artificially choke deep reasoning.

3. **Stage 5 Architecture: Scratchpad 2.0 & The Unbounded Parsimony Gradient:**
   To solve this, I designed the Unbounded Parsimony Gradient: eliminating artificial token ceilings so deep derivations can breathe, while monotonically rewarding the shortest valid path ($R = R_{\text{base}} \cdot [\gamma + (1-\gamma) \frac{K}{K+|\tau|}]$). Validated in unit tests: 5-token recall earns $+2.69$, while a deep 120-token proof remains strictly positive at $+1.81$ with zero length penalty.

4. **Frontier Co-Design (The 224x DeepSeek Multiplier):**
   Our formal memorandum proves that combining DeepSeek-V4.1-Flash's hardware Shared Global KV (890 B/tok) with SymboLM's sequence compression (40 tokens vs 2,000) achieves a **224x compound reduction in serving memory footprint** (dropping from 8.0 MB to 35.6 KB per request).

All model weights, dataset pipelines, and benchmark logs are open-sourced on GitHub ([https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM)) and Hugging Face Hub ([https://huggingface.co/gyawalisanish0/symboLM-checkpoints](https://huggingface.co/gyawalisanish0/symboLM-checkpoints)). Attached dossier: [`research/GRANT_PROJECT_DOSSIER_OCTOBER_2026.md`](file:///c:/Users/user/Dev/symboLM/research/GRANT_PROJECT_DOSSIER_OCTOBER_2026.md).

---

### Field 5: Concrete Budget & Capital Allocation ($20,000)
The $20,000 grant directly removes the three binding physical and digital constraints on my research:

1. **Workstation, Power & Connectivity Infrastructure ($5,000):**  
   In Nepal, the electrical grid suffers from frequent, unannounced 11 kV feeder trips, transformer overloads, and voltage spikes that abruptly drop power and sever active cloud telemetry. I will install a high-capacity pure sine-wave inverter with LiFePO4 battery storage (0ms switchover), a dual-WAN failover router (Fiber + Cellular), and ergonomic bed-mount displays to guarantee 100% uninterrupted workstation uptime.

2. **Dedicated Cloud Compute & Software Subscriptions ($10,000):**  
   Scaling SymboLM from our current 1.5B testbed to 14B and 32B parameter open models requires dedicated A100/H100 GPU clusters (RunPod, Lambda Labs, GCP), model evaluation credits, and telemetry tooling.

3. **Living & Care Runway ($5,000):**  
   Provides a 12-month runway covering daily personal caregiving, physical assistance, and medical necessities in Nepal, freeing 100% of my time and mental energy to pursue independent AI research.

---

### Appendix: 2-Minute Unedited Video Script (For Tyler Cowen)

*(Record horizontally from your bed workstation using your phone or webcam. Speak naturally and unedited.)*

> **[0:00 - 0:40] The Persona, Physical Reality & Domestic Isolation:**  
> "Hi Dr. Cowen. My name is Sanish Gyawali, and I'm an independent AI systems researcher from Kathmandu, Nepal. I have a severe physical disability from birth and am bedridden 24/7. My bed workstation is my lab, my university, and my entire connection to the world. In Nepal, being bedridden means society—and even my own family—assumes you can do nothing. I can't explain reinforcement learning gradients or AST compilers to them, so I don't argue. I just build. Because I cannot use a standard desk, I operate with Google DeepMind's Antigravity as my transparent operational exoskeleton—translating my architectural designs into terminal commands and cloud GPU pipelines."
>
> **[0:35 - 1:25] The Core Idea & The Empirical Receipts:**  
> "Today, reasoning models are wasting astronomical amounts of compute. Frontier models burn thousands of tokens rambling in English—performing uncertainty theater before giving an answer they already know. I created SymboLM to replace natural language CoT with dense, verifiable AST proofs across distinct cognitive registers. Working on free Kaggle GPUs, I trained a 1.5-billion parameter model that derives Olympiad math proofs in 35 tokens instead of 1,200 tokens. When our 200-question audit showed the model was looping answers, I immediately engineered Stage 4 Hard EOS alignment, and within hours proved we could eliminate the bloat and terminate cleanly."
>
> **[1:25 - 2:00] What $20,000 Unlocks:**  
> "A $20,000 Emergent Ventures grant permanently solves three bottlenecks: First, it installs solar batteries so Nepal's blackouts don't cut my training runs. Second, it funds dedicated A100 and H100 compute to scale SymboLM to 14-billion parameter models. Third, it provides a 12-month personal care runway so I can devote 100% of my life to building. The code, curriculum, and checkpoints are completely open-source. Thank you for considering my work."
