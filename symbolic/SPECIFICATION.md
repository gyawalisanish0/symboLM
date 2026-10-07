# SymboLM Reasoning Language (SRL v1.0) Specification

**Author:** Sanish Gyawali  
**Version:** 1.0.0 (Formal Specification)  
**Status:** Canonical Reference Standard  
**Project:** SymboLM (Version 1.1)

---

## 1. Scope & Design Philosophy

The **SymboLM Reasoning Language (SRL v1.0)** is an intermediate representation (IR) designed for neural deductive reasoning. It replaces ambiguous natural language chain-of-thought with an executable, token-dense, and compiler-verifiable formal scratchpad.

### Core Tenets:
1. **Atomic Expression:** Every logical transition carries maximal semantic density; linguistic connective tissue is eliminated.
2. **Determinism:** Every step can be parsed into an Abstract Syntax Tree (AST) and verified by deterministic computational engines (such as SymPy or Python).
3. **Multi-Register Adaptability:** Supports deductive mathematics, factual bypasses, structured questionnaire state-tracking, and interpersonal intent-planning within a unified syntax.
4. **Token Minimization:** Designed so that all major operators (`→`, `∴`, `|`, `✓`, `let`, `verify`) map to atomic vocabulary tokens.

---

## 2. EBNF Grammar (Extended Backus-Naur Form)

```ebnf
(* Top-level document structure *)
Document        ::= ThinkBlock AnswerBlock?
ThinkBlock      ::= "<think>" [WhiteSpace] Trace [WhiteSpace] "</think>"
AnswerBlock     ::= "<ans>" Value "</ans>"

(* Reasoning Trace *)
Trace           ::= EmptyTrace | ReasoningTrace | IntentTrace

EmptyTrace      ::= (* epsilon / zero tokens *)
ReasoningTrace  ::= StepSequence [Delim] Conclusion
IntentTrace     ::= IntentDirective

(* Sequence & Separator *)
StepSequence    ::= Step (Delim Step)*
Delim           ::= [WhiteSpace] "|" [WhiteSpace] | [WhiteSpace] "\n" [WhiteSpace]

(* Individual Step Variants *)
Step            ::= Declaration 
                  | Transformation 
                  | Verification 
                  | StateFrame 
                  | CaseBranch

(* Step Definitions *)
Declaration     ::= "let(" Identifier "=" Expression ")"
                  | "hyp(" Expression ")"

Transformation  ::= Expression "→" TargetState
TargetState     ::= Expression 
                  | Identifier "=" Expression

Verification    ::= "verify(" BooleanExpr ")" TruthMarker
TruthMarker     ::= "✓" | "✗"

StateFrame      ::= "state(" [StatePair ("," [WhiteSpace] StatePair)*] ")"
StatePair       ::= Identifier "=" Value

CaseBranch      ::= "case(" Expression ")" "[" StepSequence "]"

IntentDirective ::= "intent:" Tag (Delim "strategy:" Tag)* (Delim "tone:" Tag)*

Conclusion      ::= "∴" [WhiteSpace] "ans=" Value

(* Primitives *)
Identifier      ::= [a-zA-Z_][a-zA-Z0-9_]*
Value           ::= Number | Boolean | StringLiteral | Identifier
Number          ::= ["-"] [0-9]+ ("." [0-9]+)? (("/" | "\") [0-9]+)?
Boolean         ::= "True" | "False" | "true" | "false"
Tag             ::= [a-zA-Z0-9_-]+
Expression      ::= (* Valid algebraic, arithmetic, or predicate expression *)
BooleanExpr     ::= Expression ("==" | "!=" | "<=" | ">=" | "<" | ">") Expression
WhiteSpace      ::= (" " | "\t" | "\r" | "\n")+
```

---

## 3. Lexical Conventions & Reserved Tokens

### 3.1 Structural Markers
| Token | Name | Meaning |
|---|---|---|
| `<think>` | `THINK_OPEN` | Opens the reasoning scratchpad. |
| `</think>` | `THINK_CLOSE` | Closes the reasoning scratchpad. |
| `<ans>` | `ANS_OPEN` | Opens the final ground-truth answer. |
| `</ans>` | `ANS_CLOSE` | Closes the final ground-truth answer. |
| `\|` | `DELIMITER` | Separates discrete sequential steps. |
| `∴` | `THEREFORE` | Precedes the conclusion marker (`∴ ans=`). |
| `→` | `IMPLIES` | Maps an operation/premise to its evaluated result. |

### 3.2 Evaluation & Verification Markers
| Token | Name | Meaning |
|---|---|---|
| `✓` | `CHECK_PASS` | Mathematical invariant verified true. |
| `✗` | `CHECK_FAIL` | Invariant failed; triggers backtrack. |

### 3.3 Core Keywords
| Keyword | Syntax | Semantics |
|---|---|---|
| `let` | `let(var = expr)` | Variable assignment / initialization. |
| `hyp` | `hyp(expr)` | Introduces a given problem premise/hypothesis. |
| `verify`| `verify(condition)✓` | Invariant assertion (evaluated by compiler). |
| `state` | `state(k1=v1, k2=v2)`| Key-value state tracking (questionnaires). |
| `case` | `case(cond)[steps]` | Exhaustive case splitting. |
| `intent`| `intent: <tag>` | High-level behavioral planning for dialogue. |

---

## 4. Multi-Register Canonical Examples

### 4.1 Deductive Math Register (GSM8K Arithmetic)
```text
<think>
let(tot=16) | eat-=3→13 | bake-=4→9 | sell(9*2)→18 ∴ ans=18
</think>
<ans>18</ans>
```
*Token count:* 18 tokens.  
*Invariants:* $16 - 3 = 13$, $13 - 4 = 9$, $9 \times 2 = 18$.

### 4.2 Deductive Algebra Register (Symbolic Solution & Check)
```text
<think>
5x-15=3x+25 | -3x→2x-15=25 | +15→2x=40 | /2→x=20 | verify(5*20-15==3*20+25)✓ ∴ ans=20
</think>
<ans>20</ans>
```
*Token count:* 25 tokens.  
*Invariants:* $2x - 15 = 25 \iff 2x = 40 \iff x = 20$. Invariant check evaluated via SymPy.

### 4.3 Zero-Shot Factual Bypass Register
```text
<think></think>
Lima
```
*Token count:* 0 reasoning tokens (instant retrieval, zero overhead).

### 4.4 Questionnaire & State-Tracking Register
```text
<think>
state(fever=102, cough=5d, O2=96%) | eval(O2>=95)→stable | urgency=Med ∴ ans=Med
</think>
<ans>Med</ans>
```

### 4.5 Interpersonal Dialogue Register (Intent Scratchpad)
```text
<think>
intent: validate_frustration | strategy: document_evidence | tone: professional_calm
</think>
That is deeply frustrating. To handle this constructively, I recommend first compiling your commit records...
```

---

## 5. Compiler Error Codes & Verification Semantics

When validating an SRL trace, the compiler emits structured diagnostic codes:

| Error Code | Category | Condition |
|---|---|---|
| `E101` | Syntax | Missing opening or closing tag (`<think>`, `</think>`, `<ans>`). |
| `E102` | Syntax | Missing conclusion marker (`∴ ans=`). |
| `E103` | Syntax | Unbalanced parentheses in `let()`, `hyp()`, or `verify()`. |
| `E104` | Syntax | Missing step separator (`|` or newline). |
| `E201` | Semantic | **Invariant Violation:** The algebraic equality in `A → B` evaluates false in SymPy. |
| `E202` | Semantic | **Verification Failure:** `verify(cond)✓` was marked passed, but `cond` evaluated false. |
| `E203` | Semantic | **Answer Mismatch:** The value in `∴ ans=X` does not match `<ans>X</ans>`. |
| `E301` | Hygiene | Natural language leak detected in the symbolic trace. |

---

## 6. Integration Contract

1. **Training (SFT):** The dataset compiler rejects any candidate trace that yields an `E1xx` or `E2xx` error code.
2. **Reinforcement Learning (GRPO):**
   $$\mathcal{R}_{\text{syntax}} = \begin{cases} +0.2 & \text{if zero errors} \\ -0.5 & \text{if E1xx or E2xx present} \end{cases}$$
3. **Inference Verifier:** Traces failing compilation during Best-of-N search are pruned before majority voting.
