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

**The 2-Phase Alignment Breakthrough:** Pure reinforcement learning cannot discover a formal 15-token macro-planning grammar from a 150,000-token vocabulary by random chance (the cold-start barrier). I engineered a two-phase bridge: Stage 2.5 Syntactic SFT Seeding (grounding the Register v2.0 grammar) followed by Stage 3.5 GRPO reinforcement learning that acts as a mathematical razor—pruning trailing noise (-1.0) and penalizing blind guessing (-0.80).

Working entirely from free-tier cloud GPUs on Kaggle from my bed in Nepal, I have successfully trained and verified the Stage 2.5 Concept SFT adapter and open-sourced the weights on Hugging Face Hub.

---

### Field 5: Concrete Budget & Capital Allocation ($20,000)
The $20,000 grant directly removes the three binding physical and digital constraints on my research:

1. **Workstation, Power & Connectivity Infrastructure ($5,000):**  
   In Nepal, the electrical grid suffers from frequent, unannounced 11 kV feeder trips, transformer overloads, and voltage spikes that abruptly drop power and sever active cloud telemetry. I will install a high-capacity pure sine-wave inverter with LiFePO4 battery storage (0ms switchover), a dual-WAN failover router (Fiber + Cellular), and ergonomic bed-mount displays to guarantee 100% uninterrupted workstation uptime.

2. **Dedicated Cloud Compute & Software Subscriptions ($10,000):**  
   Scaling SymboLM from our current 1.5B testbed to 14B and 32B parameter open models requires dedicated A100/H100 GPU clusters (RunPod, Lambda Labs, GCP), model evaluation credits, and telemetry tooling.

3. **Living & Care Runway ($5,000):**  
   Provides a 12-month runway covering daily personal caregiving, physical assistance, and medical necessities in Nepal, freeing 100% of my time and mental energy to pursue independent AI research.
