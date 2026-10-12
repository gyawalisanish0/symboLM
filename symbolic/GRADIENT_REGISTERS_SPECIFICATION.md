# SymboLM Architecture Specification: Gradient Registers & Scratchpad 2.0

**Principal Systems Architect & Research Director:** Sanish Gyawali  
**AI Systems Exoskeleton:** Antigravity (Google DeepMind)  
**Version:** SRL v2.0-Gradient  
**Date:** 2026-10-12  

---

## 1. Abstract & Motivation

In SymboLM Stages 1 through 4, cognitive registers were implemented as **flat, categorical registers**:
* `<reg:bypass>` (0-step factual retrieval)
* `<reg:deduce>` (micro-AST step)
* `<reg:verify>` (hypothesis assertion)
* `<reg:intent>` (dialogue header)

While Stage 4 successfully demonstrated that a strict length and repetition penalty eliminates verbose natural language bloat (curing the 220-token repetition tail), **a static token clamp (e.g., locking reasoning at $\le 35$ tokens) creates a fundamental bottleneck for complex problems**. 

As formulated by Sanish Gyawali:
> *"The next architecture is a gradient register instead of flat, with a gradient-based reward and punishment mechanism—not locking the token limit at 35 tokens entirely, because some questions need a deeper scratchpad: Scratchpad 2.0."*

A simple arithmetic identity can be resolved in 15 tokens. An Olympiad polynomial invariant requires 35 tokens. However, a multi-phase chemical equilibrium, a Chinese Remainder Theorem system with non-coprime moduli, or a graph shortest-path derivation intrinsically requires **60 to 120 tokens of structured intermediate state**. Choking all queries to 35 tokens induces truncation and arithmetic hallucination on deep problems.

---

## 2. The Cognitive Depth Spectrum ($\Delta$-Hierarchy)

Rather than rigid categorical buckets, SymboLM v2.0 introduces a **continuous Cognitive Depth Spectrum** $\Delta \in \{0, 1, 2, 3\}$:

```
[Direct Retrieval]                                           [Deep Verification]
       Δ0                 Δ1                   Δ2                   Δ3
  (0-15 tokens)      (20-45 tokens)       (50-80 tokens)      (80-150 tokens)
   Zero-Shot          Linear AST            Branching          Scratchpad 2.0
     Bypass           Invariants           Verification         Composition
```

### Depth Tiers:

| Tier | Category | Target Budget ($L_{\text{opt}}$) | Max Elastic Budget ($L_{\text{max}}$) | Architectural Function | Example Domains |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **$\Delta_0$** | **Direct Bypass** | 1–10 tokens | 15 tokens | Zero-shot memory recall; instant emission of atomic entities. | Chemical symbols, unit definitions, historical facts. |
| **$\Delta_1$** | **Linear Invariant** | 20–35 tokens | 45 tokens | Single-theorem algebraic or physical transformations without branching. | Legendre valuations, Ohm's law, half-life calculations. |
| **$\Delta_2$** | **Branching Verification** | 45–65 tokens | 80 tokens | Multi-hypothesis testing (`hyp [...]`), active runtime assertion (`verify(...)✓/✗`). | Diophantine equations, inradius geometry, Bayes probability. |
| **$\Delta_3$** | **Deep Scratchpad 2.0** | 75–120 tokens | 150 tokens | Elastic state tracking, multi-variable recurrence unrolling, algorithmic traces. | Non-coprime CRT, chemical equilibria, graph invariants. |

---

## 3. Scratchpad 2.0 Syntax Grammar

Scratchpad 2.0 provides an explicit, elastic cognitive workspace that declares its intended computational depth dynamically:

```xml
<reg:scratchpad depth="2" domain="physics">
  state(v0=40, theta=30°, g=9.8) 
  | vy0=40*sin(30°)=20 
  | H=(vy0^2)/(2*g)=400/19.6=20.41 
  | verify(H>0)✓ 
  ∴ ans=20.41
</reg:scratchpad>
<ans>20.41</ans>
```

For deep multi-tier reasoning ($\Delta_3$), Scratchpad 2.0 supports nested theorem decomposition:

```xml
<reg:scratchpad depth="3" domain="number_theory">
  target: 5x ≡ 9 (mod 11)
  | hyp [gcd(5,11)=1]✓
  | inv: 5*9=45=4*11+1 ∴ 5^-1 ≡ 9 (mod 11)
  | x ≡ 9*9 = 81 ≡ 4 (mod 11)
  | verify(5*4=20=11+9 ≡ 9 (mod 11))✓
  ∴ ans=4
</reg:scratchpad>
<ans>4</ans>
```

---

## 4. Gradient-Based Reward & Punishment Mechanism

Stage 3 and 4 used step-function penalties. In SymboLM v2.0, reward shaping is **continuous and complexity-proportional**:

$$R(\tau, y^*, C) = R_{\text{correct}}(y^*) \cdot \left[1 + \alpha \cdot \Phi(\tau, C)\right] - \beta \cdot \Psi(\tau, C) - \gamma \cdot \Omega(\tau)$$

Where:

### 1. Correctness Base Reward:
$$R_{\text{correct}}(y^*) = \begin{cases} +2.0 & \text{if } \text{normalize}(\hat{y}) == \text{normalize}(y^*) \\ -0.8 & \text{otherwise} \end{cases}$$

### 2. Gradient Economy Scaling ($\Phi$):
Rewards compact derivations relative to the question's intrinsic complexity tier $C$:
$$\Phi(\tau, C) = \max\left(0.0, \, \frac{L_{\text{optimal}}(C) - |\tau|}{L_{\text{optimal}}(C)}\right)$$
* For $\Delta_0$ (Potassium $\to$ K): $L_{\text{opt}} = 10$. If answered in 4 tokens, $\Phi = 0.60$.
* For $\Delta_3$ (CRT proof): $L_{\text{opt}} = 90$. If proved cleanly in 70 tokens, $\Phi = 0.22$.

### 3. Elastic Length Punishment ($\Psi$):
Instead of a cliff at 35 tokens, punishment activates smoothly only when tokens exceed $L_{\text{max}}(C)$:
$$\Psi(\tau, C) = \text{ReLU}\left(|\tau| - L_{\text{max}}(C)\right) \cdot \lambda(C)$$
* No penalty is incurred as long as $|\tau| \le L_{\text{max}}(C)$.
* Allows deep problems to breathe in Scratchpad 2.0 without penalty!

### 4. Hard Repetition & Debris Penalty ($\Omega$):
$$\Omega(\tau) = \begin{cases} 2.5 & \text{if } \text{detect\_ngram\_repetition}(\tau) \\ 2.0 & \text{if } \text{detect\_web\_debris}(\tau) \\ 0.0 & \text{otherwise} \end{cases}$$

---

### 4.2 The Unbounded Parsimony Gradient (Refined Directive by Sanish Gyawali)

Rather than enforcing any artificial token ceiling ($L_{\text{max}}$), the policy is guided by an **Unbounded Parsimony Gradient**:
* **Zero Token Lock:** No problem is ever penalized or truncated simply because it requires deep derivation. Correct answers are *always* strictly positive ($R > 0$).
* **Latent Knowledge Leverage:** The model already possesses distilled factual and algorithmic knowledge. It should not waste tokens restating textbook definitions.
* **Continuous Monotonic Parsimony:** Shorter valid derivations earn strictly higher rewards, driving GRPO group optimization toward the Minimum Description Length (MDL):

$$R_{\text{parsimony}}(\tau, y^*) = R_{\text{base}} \cdot \left[ \gamma_{\text{floor}} + (1 - \gamma_{\text{floor}}) \cdot \frac{K}{K + |\tau|} \right] - \Omega(\tau) + R_{\text{closure}}$$

Where:
* $R_{\text{base}} = 2.5$, $K = 35$, $\gamma_{\text{floor}} = 0.40$
* **Ultra-compact (5 tokens):** $R = +2.69$ (Taps latent memory without rambling)
* **Compact proof (25 tokens):** $R = +2.47$
* **Multi-step derivation (60 tokens):** $R = +2.08$
* **Deep Olympiad derivation (120 tokens):** $R = +1.81$ (**STILL POSITIVE, NO CLIFF!**)
* **Incorrect answer:** $R = -1.00$

$$\frac{\partial R}{\partial |\tau|} < 0 \quad \text{everywhere for correct answers}$$

In any GRPO sample group ($G=4\dots8$), the candidate rollout that derives the exact answer with the fewest necessary steps achieves the highest positive advantage ($A_i > 0$), naturally training the model to be ultra-dense and parsimonious without locking deep reasoning.

---

## 5. Architectural Verification Plan (Cross-Domain 200 Suite)

The new [`data/eval_200_cross_domain_benchmark.jsonl`](file:///c:/Users/user/Dev/symboLM/data/eval_200_cross_domain_benchmark.jsonl) explicitly annotates each problem with its Cognitive Depth Spectrum tag:
* 80 questions at $\Delta_0$
* 99 questions at $\Delta_1$
* 20 questions at $\Delta_2$
* 1 question at $\Delta_3$

This benchmark will serve as the empirical ground truth for tuning the Gradient Register weights in Stage 5.
