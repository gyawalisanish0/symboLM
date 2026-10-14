"""
SymboLM Concept Mathematics Curriculum Generator (Register v2.0 / v2.1)
========================================================================
Generates a comprehensive 5,000-sample competition-grade dataset grounded in
core mathematical invariants and Register v2.0 cognitive formatting:
  - <reg:plan>    Macro-Concept Planning (invariants, theorems, strategies)
  - <reg:deduce>  Dense Micro-Symbolic Execution (substitutions, AST reductions)
  - <reg:verify>  Invariant Verification (bounds, mod check, parity)
  - <reg:bypass>  Zero-Shot Factual Recall
  - <reg:intent>  Interpersonal & Advisory Dialogue

Invariant Families Covered (12 Core Mathematical & Physical Domains):
  1. Polynomial Invariants & Symmetric Powers (x + 1/x = k -> x^n + 1/x^n)
  2. Modular Congruence Systems & Chinese Remainder Theorem (CRT)
  3. Modular Exponentiation & Euler's Totient Reductions (a^k mod m)
  4. Legendre's Formula for Factorial Prime Valuations (nu_p(N!))
  5. Combinatorial Casework & Derangements (D_n, Stars & Bars)
  6. Geometric Invariants (Right triangle inradius Area = r * s)
  7. Diophantine Factorization & Harmonic Work Rates (x^2 - y^2 = N)
  8. Linear Systems, Equations & Determinants (ax + b = c, 2x2 systems, det(M))
  9. Combinations, Permutations & Binomial Expansions (C(n, k), P(n, k), (x+a)^n)
  10. Discrete Sequences & Recurrences (AP, GP, Lucas numbers L_n)
  11. Real-World Applied Arithmetic (discounts, age ratios, relative speed)
  12. Physics & Kinematics Invariants (v=u+at, v^2=u^2+2as, E_k, Ohm's law)

Author: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Exoskeleton: Antigravity (Google DeepMind)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# =============================================================================
# Helper Mathematical Functions
# =============================================================================

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Returns (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    if a == 0:
        return b, 0, 1
    g, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return g, x, y


def mod_inverse(a: int, m: int) -> int:
    """Computes the modular inverse of a modulo m."""
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"Modular inverse does not exist for {a} mod {m}")
    return (x % m + m) % m


def euler_phi(n: int) -> int:
    """Computes Euler's totient function phi(n)."""
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def derangement(n: int) -> int:
    """Computes the derangement number D_n."""
    if n == 0:
        return 1
    if n == 1:
        return 0
    d_prev2, d_prev1 = 1, 0
    for i in range(2, n + 1):
        curr = (i - 1) * (d_prev1 + d_prev2)
        d_prev2, d_prev1 = d_prev1, curr
    return d_prev1


# =============================================================================
# Family 1: Polynomial Invariants & Symmetric Powers
# =============================================================================

def gen_polynomial_invariant_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(count):
        k = random.randint(3, 9)
        power = random.choice([2, 3, 4, 5])

        u2 = k * k - 2
        u3 = k * k * k - 3 * k
        u4 = u2 * u2 - 2
        u5 = u2 * u3 - k

        if power == 2:
            ans = u2
            target_expr = "x^2 + 1/x^2"
            canon_plan = f"<reg:plan>concept: polynomial_invariant | identity: (x²+1/x²)=(x+1/x)²-2 | target: x²+1/x²</reg:plan>"
            canon_deduce = f"<reg:deduce>let(u={k}) | x²+1/x²=u²-2={k}²-2={u2} ∴ ans={u2}</reg:deduce>"
        elif power == 3:
            ans = u3
            target_expr = "x^3 + 1/x^3"
            canon_plan = f"<reg:plan>concept: polynomial_invariant | identity: (x³+1/x³)=(x+1/x)³-3(x+1/x) | target: x³+1/x³</reg:plan>"
            canon_deduce = f"<reg:deduce>let(u={k}) | x³+1/x³=u³-3u={k}³-3·{k}={k**3}-{3*k}={u3} ∴ ans={u3}</reg:deduce>"
        elif power == 4:
            ans = u4
            target_expr = "x^4 + 1/x^4"
            canon_plan = f"<reg:plan>concept: polynomial_invariant | identity: (x⁴+1/x⁴)=(x²+1/x²)²-2 | target: x⁴+1/x⁴</reg:plan>"
            canon_deduce = f"<reg:deduce>let(u={k}) | x²+1/x²=u²-2={u2} | x⁴+1/x⁴={u2}²-2={u2**2}-2={u4} ∴ ans={u4}</reg:deduce>"
        else:
            ans = u5
            target_expr = "x^5 + 1/x^5"
            canon_plan = f"<reg:plan>concept: polynomial_invariant | identity: (x⁵+1/x⁵)=(x²+1/x²)(x³+1/x³)-(x+1/x) | target: x⁵+1/x⁵</reg:plan>"
            canon_deduce = f"<reg:deduce>let(u={k}) | u₂={u2} | u₃={u3} | x⁵+1/x⁵={u2}·{u3}-{k}={u2*u3}-{k}={u5} ∴ ans={u5}</reg:deduce>"

        canon_verify = f"<reg:verify>verify({ans}>0)✓</reg:verify>"
        prompt = f"If x + 1/x = {k}, find the exact value of {target_expr}."

        samples.append({
            "id": f"concept_poly_{i}",
            "domain": "Invariant Algebra",
            "concept": "polynomial_invariant",
            "prompt": prompt,
            "ground_truth": str(ans),
            "canonical_plan": canon_plan,
            "canonical_deduce": canon_deduce,
            "canonical_verify": canon_verify,
            "full_target": f"<think>\n{canon_plan}\n{canon_deduce}\n{canon_verify}\n</think>\n<ans>{ans}</ans>",
            "type": "exact_integer",
            "complexity": "multi_step"
        })
    return samples


# =============================================================================
# Family 2: Modular Congruence Systems & Chinese Remainder Theorem
# =============================================================================

def gen_chinese_remainder_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    moduli_pairs = [(3, 5), (5, 7), (7, 11), (5, 11), (7, 13), (9, 11)]

    for i in range(count):
        m1, m2 = random.choice(moduli_pairs)
        ans = random.randint(1, m1 * m2 - 1)
        a1 = ans % m1
        a2 = ans % m2

        # Reduction derivation
        inv_m1 = mod_inverse(m1, m2)
        diff = (a2 - a1) % m2
        k_val = (diff * inv_m1) % m2
        reconstructed = a1 + m1 * k_val

        canon_plan = f"<reg:plan>concept: chinese_remainder | moduli: [{m1},{m2}] | method: modular_inverses</reg:plan>"
        canon_deduce = (
            f"<reg:deduce>n≡{a1} mod {m1}→n={m1}k+{a1} | {m1}k+{a1}≡{a2} mod {m2}→"
            f"{m1}k≡{diff} mod {m2}→k≡{k_val} mod {m2} | n={m1}·{k_val}+{a1}={reconstructed} ∴ ans={reconstructed}</reg:deduce>"
        )
        canon_verify = f"<reg:verify>verify({reconstructed} mod {m1}={a1} ∧ {reconstructed} mod {m2}={a2})✓</reg:verify>"
        prompt = f"Find the smallest positive integer n such that n ≡ {a1} (mod {m1}) and n ≡ {a2} (mod {m2})."

        samples.append({
            "id": f"concept_crt_{i}",
            "domain": "Number Theory",
            "concept": "chinese_remainder",
            "prompt": prompt,
            "ground_truth": str(reconstructed),
            "canonical_plan": canon_plan,
            "canonical_deduce": canon_deduce,
            "canonical_verify": canon_verify,
            "full_target": f"<think>\n{canon_plan}\n{canon_deduce}\n{canon_verify}\n</think>\n<ans>{reconstructed}</ans>",
            "type": "exact_integer",
            "complexity": "multi_step"
        })
    return samples


# =============================================================================
# Family 3: Modular Exponentiation & Euler's Totient Reductions
# =============================================================================

def gen_modular_totient_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    bases = [2, 3, 5, 7]
    moduli = [10, 13, 17, 19, 25, 100]

    for i in range(count):
        base = random.choice(bases)
        mod = random.choice(moduli)
        while math.gcd(base, mod) != 1:
            base = random.choice([2, 3, 7, 11])
            mod = random.choice([13, 17, 19, 23])

        phi = euler_phi(mod)
        exp = random.choice([2024, 2025, 2026, 2027, 1000, 500])
        red_exp = exp % phi
        ans = pow(base, exp, mod)

        canon_plan = f"<reg:plan>concept: modular_totient | base: {base} | modulus: {mod} | theorem: φ({mod})={phi}</reg:plan>"
        canon_deduce = (
            f"<reg:deduce>{base}^{exp} mod {mod} | φ({mod})={phi} | {exp}={phi}·{exp//phi}+{red_exp}→"
            f"{base}^{exp}≡{base}^{red_exp} mod {mod} | {base}^{red_exp} mod {mod}={ans} ∴ ans={ans}</reg:deduce>"
        )
        canon_verify = f"<reg:verify>verify({ans}<{mod} ∧ {ans}≥0)✓</reg:verify>"
        prompt = f"Compute the remainder when {base}^{exp} is divided by {mod}."

        samples.append({
            "id": f"concept_totient_{i}",
            "domain": "Number Theory",
            "concept": "modular_totient",
            "prompt": prompt,
            "ground_truth": str(ans),
            "canonical_plan": canon_plan,
            "canonical_deduce": canon_deduce,
            "canonical_verify": canon_verify,
            "full_target": f"<think>\n{canon_plan}\n{canon_deduce}\n{canon_verify}\n</think>\n<ans>{ans}</ans>",
            "type": "exact_integer",
            "complexity": "multi_step"
        })
    return samples


# =============================================================================
# Family 4: Legendre's Formula for Factorial Prime Valuations
# =============================================================================

def gen_legendre_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    primes = [2, 3, 5, 7, 11]

    for i in range(count):
        p = random.choice(primes)
        n = random.randint(50, 2500)

        terms = []
        power = p
        ans = 0
        while power <= n:
            term = n // power
            terms.append(f"⌊{n}/{power}⌋={term}")
            ans += term
            power *= p

        terms_str = " | ".join(terms)
        canon_plan = f"<reg:plan>concept: legendre_formula | prime: {p} | n: {n} | series: Σ⌊{n}/{p}^k⌋</reg:plan>"
        canon_deduce = f"<reg:deduce>{terms_str} | Σ={ans} ∴ ans={ans}</reg:deduce>"
        canon_verify = f"<reg:verify>verify({ans}>0)✓</reg:verify>"
        prompt = f"Find the highest power of {p} that divides {n}! (Legendre's formula)."

        samples.append({
            "id": f"concept_legendre_{i}",
            "domain": "Number Theory",
            "concept": "legendre_formula",
            "prompt": prompt,
            "ground_truth": str(ans),
            "canonical_plan": canon_plan,
            "canonical_deduce": canon_deduce,
            "canonical_verify": canon_verify,
            "full_target": f"<think>\n{canon_plan}\n{canon_deduce}\n{canon_verify}\n</think>\n<ans>{ans}</ans>",
            "type": "exact_integer",
            "complexity": "multi_step"
        })
    return samples


# =============================================================================
# Family 5: Combinatorial Casework & Derangements
# =============================================================================

def gen_combinatorics_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(count):
        is_derange = random.random() > 0.4
        if is_derange:
            n = random.choice([3, 4, 5, 6, 7])
            ans = derangement(n)
            canon_plan = f"<reg:plan>concept: derangement | n: {n} | recurrence: D_n=(n-1)(D_{{n-1}}+D_{{n-2}})</reg:plan>"
            
            steps = []
            d_vals = {0: 1, 1: 0}
            for k in range(2, n + 1):
                d_vals[k] = (k - 1) * (d_vals[k-1] + d_vals[k-2])
                steps.append(f"D_{k}=({k}-1)({d_vals[k-1]}+{d_vals[k-2]})={d_vals[k]}")
            
            canon_deduce = f"<reg:deduce>{' | '.join(steps)} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>0)✓</reg:verify>"
            prompt = f"Find the number of derangements of {n} items (permutations where no element appears in its original position)."
        else:
            # Stars and Bars: Distinct partitions with lower bound
            total = random.randint(10, 20)
            bins = random.randint(3, 4)
            min_each = random.randint(1, 2)
            
            # x_1 + ... + x_k = total with x_i >= min_each
            # equivalent to y_1 + ... + y_k = total - bins * min_each >= 0
            # formula: C(rem + bins - 1, bins - 1)
            rem = total - bins * min_each
            ans = math.comb(rem + bins - 1, bins - 1)
            
            canon_plan = f"<reg:plan>concept: stars_and_bars | bins: {bins} | lower_bound: {min_each} | rem: {rem}</reg:plan>"
            canon_deduce = f"<reg:deduce>rem={total}-{bins}·{min_each}={rem} | C({rem}+{bins}-1,{bins}-1)=C({rem+bins-1},{bins-1})={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>0)✓</reg:verify>"
            prompt = f"In how many ways can {total} identical items be distributed into {bins} distinct containers such that each container receives at least {min_each} items?"

        samples.append({
            "id": f"concept_comb_{i}",
            "domain": "Combinatorics",
            "concept": "combinatorics_recurrence",
            "prompt": prompt,
            "ground_truth": str(ans),
            "canonical_plan": canon_plan,
            "canonical_deduce": canon_deduce,
            "canonical_verify": canon_verify,
            "full_target": f"<think>\n{canon_plan}\n{canon_deduce}\n{canon_verify}\n</think>\n<ans>{ans}</ans>",
            "type": "exact_integer",
            "complexity": "multi_step"
        })
    return samples


# =============================================================================
# Family 6: Geometric Invariants (Inradius, Semiperimeter, Area)
# =============================================================================

def gen_geometry_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    # Primitive and scaled Pythagorean triples (a, b, c)
    triples = [
        (3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (9, 40, 41),
        (6, 8, 10), (10, 24, 26), (15, 20, 25), (12, 16, 20), (18, 24, 30)
    ]

    for i in range(count):
        a, b, c = random.choice(triples)
        P = a + b + c
        s = P // 2
        Area = (a * b) // 2
        r = Area // s  # Since right triangle Area = r * s

        canon_plan = f"<reg:plan>concept: inradius_geometry | theorem: Area=r*s | semiperimeter: s=P/2</reg:plan>"
        canon_deduce = f"<reg:deduce>let(P={P}, r={r}) | s={P}/2={s} | Area=r·s={r}·{s}={Area} ∴ ans={Area}</reg:deduce>"
        canon_verify = f"<reg:verify>verify({Area}>0)✓</reg:verify>"
        prompt = f"In a right triangle with inradius r = {r} and perimeter P = {P}, find the exact area of the triangle."

        samples.append({
            "id": f"concept_geo_{i}",
            "domain": "Geometry Invariant",
            "concept": "inradius_geometry",
            "prompt": prompt,
            "ground_truth": str(Area),
            "canonical_plan": canon_plan,
            "canonical_deduce": canon_deduce,
            "canonical_verify": canon_verify,
            "full_target": f"<think>\n{canon_plan}\n{canon_deduce}\n{canon_verify}\n</think>\n<ans>{Area}</ans>",
            "type": "exact_integer",
            "complexity": "multi_step"
        })
    return samples


# =============================================================================
# Family 7: Diophantine Factorization & Work Rates
# =============================================================================

def gen_diophantine_and_rates_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(count):
        is_diophantine = random.random() > 0.5
        if is_diophantine:
            # x^2 - y^2 = N = (x-y)(x+y)
            # Pick odd N with known factor pairs
            N_candidates = [45, 77, 105, 117, 135, 165, 189, 225]
            N = random.choice(N_candidates)
            
            # Count factor pairs (d1, d2) such that d1 * d2 = N and d1 <= d2 and d1 == d2 mod 2
            valid_pairs = []
            for d1 in range(1, int(math.isqrt(N)) + 1):
                if N % d1 == 0:
                    d2 = N // d1
                    if (d1 % 2) == (d2 % 2):
                        valid_pairs.append((d1, d2))
            ans = len(valid_pairs)

            canon_plan = f"<reg:plan>concept: diophantine_difference_of_squares | identity: x²-y²=(x-y)(x+y)={N} | parity: d₁≡d₂ mod 2</reg:plan>"
            pairs_str = ", ".join([f"({p[0]},{p[1]})" for p in valid_pairs])
            canon_deduce = f"<reg:deduce>factors({N})→{pairs_str} | len={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>0)✓</reg:verify>"
            prompt = f"Find all positive integer solutions (x, y) to x^2 - y^2 = {N}. How many valid pairs exist?"
        else:
            # Staggered work rate
            # Pump A fills in rA hours, Pump B fills in rB hours
            rA = random.choice([4, 6, 8])
            rB = random.choice([6, 8, 12])
            # Pump A runs alone for t1 hours
            t1 = 1
            work_done_A = t1 / rA
            rem_work = 1.0 - work_done_A
            combined_rate = (1 / rA) + (1 / rB)
            t2 = rem_work / combined_rate
            total_time = t1 + t2
            
            # Format cleanly as integer or simple round
            if abs(total_time - round(total_time)) < 1e-4:
                ans_str = str(int(round(total_time)))
            else:
                ans_str = str(round(total_time, 2))

            canon_plan = f"<reg:plan>concept: harmonic_work_rate | rates: [1/{rA}, 1/{rB}] | schedule: staggered</reg:plan>"
            canon_deduce = f"<reg:deduce>W_A={t1}/{rA} | W_rem=1-{t1}/{rA} | t₂=W_rem/(1/{rA}+1/{rB}) | t_total={t1}+t₂={ans_str} ∴ ans={ans_str}</reg:deduce>"
            canon_verify = f"<reg:verify>verify(ans>0)✓</reg:verify>"
            prompt = f"Pump A fills a tank in {rA} hours and Pump B fills it in {rB} hours. If Pump A runs alone for {t1} hour and then Pump B joins, how many total hours does it take to fill the tank?"
            ans = ans_str

        samples.append({
            "id": f"concept_dio_rate_{i}",
            "domain": "Diophantine & Rates",
            "concept": "diophantine_or_rates",
            "prompt": prompt,
            "ground_truth": str(ans),
            "canonical_plan": canon_plan,
            "canonical_deduce": canon_deduce,
            "canonical_verify": canon_verify,
            "full_target": f"<think>\n{canon_plan}\n{canon_deduce}\n{canon_verify}\n</think>\n<ans>{ans}</ans>",
            "type": "exact_integer" if isinstance(ans, int) else "numeric",
            "complexity": "multi_step"
        })
    return samples


# =============================================================================
# Family 8: Linear Systems, Equations & Determinants
# =============================================================================

def gen_linear_algebra_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(count):
        subtype = random.choice(["single", "system", "det"])
        if subtype == "single":
            a = random.randint(2, 9)
            x = random.randint(-15, 25)
            while x == 0:
                x = random.randint(-15, 25)
            b = random.randint(-30, 30)
            c = a * x + b
            b_sign = f"+ {b}" if b >= 0 else f"- {abs(b)}"
            prompt = f"Solve for x: {a}x {b_sign} = {c}."
            ans = x
            canon_plan = f"<reg:plan>concept: linear_equation | form: {a}x{b_sign}={c} | solve: isolate_x</reg:plan>"
            canon_deduce = f"<reg:deduce>{a}x={c}-({b})={c-b} | x={c-b}/{a}={x} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({a}*({x})+({b})=={c})✓</reg:verify>"
        elif subtype == "system":
            x = random.randint(1, 12)
            y = random.randint(1, 12)
            while True:
                a1, b1 = random.randint(1, 5), random.randint(1, 5)
                a2, b2 = random.randint(1, 5), random.randint(1, 5)
                det = a1 * b2 - a2 * b1
                if det != 0:
                    break
            c1 = a1 * x + b1 * y
            c2 = a2 * x + b2 * y
            target_var = random.choice(["x", "y", "x+y"])
            if target_var == "x":
                ans = x
                prompt = f"Solve the system of equations for x: {a1}x + {b1}y = {c1} and {a2}x + {b2}y = {c2}."
            elif target_var == "y":
                ans = y
                prompt = f"Solve the system of equations for y: {a1}x + {b1}y = {c1} and {a2}x + {b2}y = {c2}."
            else:
                ans = x + y
                prompt = f"Solve the system of equations for x + y: {a1}x + {b1}y = {c1} and {a2}x + {b2}y = {c2}."
            canon_plan = f"<reg:plan>concept: linear_system | equations: [{a1}x+{b1}y={c1}, {a2}x+{b2}y={c2}] | target: {target_var}</reg:plan>"
            canon_deduce = f"<reg:deduce>det={a1}·{b2}-{a2}·{b1}={det} | x=({c1}·{b2}-{c2}·{b1})/{det}={x} | y=({a1}·{c2}-{a2}·{c1})/{det}={y} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({a1}·{x}+{b1}·{y}=={c1} ∧ {a2}·{x}+{b2}·{y}=={c2})✓</reg:verify>"
        else:
            a = random.randint(-5, 9)
            b = random.randint(-5, 9)
            c = random.randint(-5, 9)
            d = random.randint(-5, 9)
            det = a * d - b * c
            ans = det
            prompt = f"Find the determinant of the 2x2 matrix [[{a}, {b}], [{c}, {d}]]."
            canon_plan = f"<reg:plan>concept: matrix_determinant | matrix: [[{a},{b}],[{c},{d}]] | formula: ad-bc</reg:plan>"
            canon_deduce = f"<reg:deduce>det={a}·({d})-{b}·({c})={a*d}-({b*c})={det} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify(det=={det})✓</reg:verify>"

        samples.append({
            "id": f"concept_linear_{i}",
            "domain": "Linear Algebra",
            "concept": "linear_algebra",
            "prompt": prompt,
            "ground_truth": str(ans),
            "canonical_plan": canon_plan,
            "canonical_deduce": canon_deduce,
            "canonical_verify": canon_verify,
            "full_target": f"<think>\n{canon_plan}\n{canon_deduce}\n{canon_verify}\n</think>\n<ans>{ans}</ans>",
            "type": "exact_integer",
            "complexity": "multi_step"
        })
    return samples


# =============================================================================
# Family 9: Combinations, Permutations & Binomial Expansions
# =============================================================================

def gen_combinations_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(count):
        subtype = random.choice(["comb", "perm", "binomial"])
        if subtype == "comb":
            n = random.randint(5, 15)
            k = random.randint(2, min(5, n - 1))
            ans = math.comb(n, k)
            prompt = f"From a group of {n} distinct candidates, how many ways can a committee of {k} members be chosen?"
            num_factors = [str(n - j) for j in range(k)]
            den_factors = [str(j + 1) for j in range(k)]
            canon_plan = f"<reg:plan>concept: combination | formula: C(n,k)=n!/(k!(n-k)!) | n: {n} | k: {k}</reg:plan>"
            canon_deduce = f"<reg:deduce>C({n},{k})=({'·'.join(num_factors)})/({'·'.join(den_factors)})={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>0 ∧ {ans}==C({n},{k}))✓</reg:verify>"
        elif subtype == "perm":
            n = random.randint(4, 10)
            k = random.randint(2, min(4, n - 1))
            ans = math.perm(n, k)
            prompt = f"How many distinct permutations of length {k} can be formed from a set of {n} distinct symbols?"
            factors = [str(n - j) for j in range(k)]
            canon_plan = f"<reg:plan>concept: permutation | formula: P(n,k)=n!/(n-k)! | n: {n} | k: {k}</reg:plan>"
            canon_deduce = f"<reg:deduce>P({n},{k})={'·'.join(factors)}={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>0 ∧ {ans}==P({n},{k}))✓</reg:verify>"
        else:
            n = random.randint(4, 8)
            k = random.randint(1, n - 1)
            a = random.randint(2, 3)
            comb = math.comb(n, k)
            pwr = a ** (n - k)
            ans = comb * pwr
            prompt = f"Find the coefficient of x^{k} in the binomial expansion of (x + {a})^{n}."
            canon_plan = f"<reg:plan>concept: binomial_expansion | term: C(n,k)*a^(n-k)*x^k | n: {n} | k: {k} | a: {a}</reg:plan>"
            canon_deduce = f"<reg:deduce>term=C({n},{k})·{a}^{n-k}·x^{k}={comb}·{pwr}·x^{k}={ans}·x^{k} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>0)✓</reg:verify>"

        samples.append({
            "id": f"concept_comb_{i}",
            "domain": "Combinatorics & Binomial",
            "concept": "combinations_permutations",
            "prompt": prompt,
            "ground_truth": str(ans),
            "canonical_plan": canon_plan,
            "canonical_deduce": canon_deduce,
            "canonical_verify": canon_verify,
            "full_target": f"<think>\n{canon_plan}\n{canon_deduce}\n{canon_verify}\n</think>\n<ans>{ans}</ans>",
            "type": "exact_integer",
            "complexity": "multi_step"
        })
    return samples


# =============================================================================
# Family 10: Discrete Sequences & Recurrences
# =============================================================================

def gen_sequences_recurrences_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(count):
        subtype = random.choice(["ap_term", "ap_sum", "gp_term", "lucas"])
        if subtype == "ap_term":
            a1 = random.randint(1, 20)
            d = random.randint(2, 9)
            n = random.randint(10, 40)
            ans = a1 + (n - 1) * d
            prompt = f"In an arithmetic progression, the first term is {a1} and the common difference is {d}. Find the {n}th term."
            canon_plan = f"<reg:plan>concept: arithmetic_progression | a₁: {a1} | d: {d} | target: a_{n}</reg:plan>"
            canon_deduce = f"<reg:deduce>a_{n}=a₁+({n}-1)d={a1}+{n-1}·{d}={a1}+{(n-1)*d}={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}=={a1}+({n}-1)*{d})✓</reg:verify>"
        elif subtype == "ap_sum":
            a1 = random.randint(1, 10)
            d = random.randint(2, 6)
            n = random.choice([10, 12, 14, 16, 20])
            ans = (n * (2 * a1 + (n - 1) * d)) // 2
            prompt = f"Find the sum of the first {n} terms of an arithmetic progression with first term {a1} and common difference {d}."
            canon_plan = f"<reg:plan>concept: arithmetic_series_sum | a₁: {a1} | d: {d} | n: {n} | formula: S_n=n/2*(2a1+(n-1)d)</reg:plan>"
            canon_deduce = f"<reg:deduce>S_{n}={n}/2·(2·{a1}+{n-1}·{d})={n//2}·({2*a1}+{(n-1)*d})={n//2}·{2*a1+(n-1)*d}={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>0)✓</reg:verify>"
        elif subtype == "gp_term":
            a1 = random.randint(1, 4)
            r = random.choice([2, 3])
            n = random.randint(4, 7)
            ans = a1 * (r ** (n - 1))
            prompt = f"In a geometric sequence, the first term is {a1} and the common ratio is {r}. Find the {n}th term."
            canon_plan = f"<reg:plan>concept: geometric_sequence | a₁: {a1} | r: {r} | target: a_{n}</reg:plan>"
            canon_deduce = f"<reg:deduce>a_{n}=a₁·r^{n-1}={a1}·{r}^{n-1}={a1}·{r**(n-1)}={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>0)✓</reg:verify>"
        else:
            lucas_seq = [2, 1]
            for j in range(2, 12):
                lucas_seq.append(lucas_seq[-1] + lucas_seq[-2])
            n = random.randint(3, 10)
            ans = lucas_seq[n]
            prompt = f"The Lucas sequence is defined by L_0 = 2, L_1 = 1, and L_n = L_{{n-1}} + L_{{n-2}} for n >= 2. Find L_{n}."
            steps = [f"L_{k}={lucas_seq[k-1]}+{lucas_seq[k-2]}={lucas_seq[k]}" for k in range(2, n + 1)]
            canon_plan = f"<reg:plan>concept: lucas_recurrence | recurrence: L_n=L_{{n-1}}+L_{{n-2}} | base: [2,1] | target: L_{n}</reg:plan>"
            canon_deduce = f"<reg:deduce>{' | '.join(steps)} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>0)✓</reg:verify>"

        samples.append({
            "id": f"concept_seq_{i}",
            "domain": "Discrete Sequences",
            "concept": "sequences_recurrences",
            "prompt": prompt,
            "ground_truth": str(ans),
            "canonical_plan": canon_plan,
            "canonical_deduce": canon_deduce,
            "canonical_verify": canon_verify,
            "full_target": f"<think>\n{canon_plan}\n{canon_deduce}\n{canon_verify}\n</think>\n<ans>{ans}</ans>",
            "type": "exact_integer",
            "complexity": "multi_step"
        })
    return samples


# =============================================================================
# Family 11: Real-World Applied Arithmetic & Word Problems
# =============================================================================

def gen_arithmetic_word_problem_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(count):
        subtype = random.choice(["discount", "age", "speed"])
        if subtype == "discount":
            price = random.choice([50, 80, 100, 120, 150, 200, 250, 300, 400])
            d1 = random.choice([10, 20, 25, 30, 50])
            d2 = random.choice([10, 20])
            p1 = price * (100 - d1) // 100
            ans = p1 * (100 - d2) // 100
            prompt = f"An item with an original price of ${price} is discounted by {d1}%, and then an additional discount of {d2}% is applied. What is the final price?"
            canon_plan = f"<reg:plan>concept: successive_discounts | original_price: {price} | discounts: [{d1}%, {d2}%]</reg:plan>"
            canon_deduce = f"<reg:deduce>p₁={price}·(1-{d1}/100)={p1} | p₂={p1}·(1-{d2}/100)={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>0 ∧ {ans}<{price})✓</reg:verify>"
        elif subtype == "age":
            s = random.randint(5, 15)
            k = random.choice([3, 4, 5])
            f = k * s
            y = (k - 2) * s
            target = random.choice(["father", "son"])
            if target == "father":
                ans = f
                prompt = f"A father is currently {k} times as old as his son. In {y} years, he will be twice as old as his son. What is the father's current age?"
            else:
                ans = s
                prompt = f"A father is currently {k} times as old as his son. In {y} years, he will be twice as old as his son. What is the son's current age?"
            canon_plan = f"<reg:plan>concept: age_ratio_system | relation: f={k}s | future: f+{y}=2(s+{y}) | target: {target}</reg:plan>"
            canon_deduce = f"<reg:deduce>{k}s+{y}=2s+{2*y}→({k}-2)s={y}→s={s} | f={k}·{s}={f} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({f}=={k}*{s} ∧ {f}+{y}==2*({s}+{y}))✓</reg:verify>"
        else:
            v1 = random.choice([40, 50, 60, 70])
            v2 = random.choice([50, 60, 80])
            t = random.randint(2, 5)
            D = (v1 + v2) * t
            ans = t
            prompt = f"Two trains {D} km apart travel towards each other at constant speeds of {v1} km/h and {v2} km/h respectively. After how many hours will they meet?"
            canon_plan = f"<reg:plan>concept: relative_velocity | distance: {D} | v₁: {v1} | v₂: {v2} | v_rel: v₁+v₂</reg:plan>"
            canon_deduce = f"<reg:deduce>v_rel={v1}+{v2}={v1+v2} | t=D/v_rel={D}/{v1+v2}={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({v1}*{ans}+{v2}*{ans}=={D})✓</reg:verify>"

        samples.append({
            "id": f"concept_word_{i}",
            "domain": "Applied Arithmetic",
            "concept": "arithmetic_word_problem",
            "prompt": prompt,
            "ground_truth": str(ans),
            "canonical_plan": canon_plan,
            "canonical_deduce": canon_deduce,
            "canonical_verify": canon_verify,
            "full_target": f"<think>\n{canon_plan}\n{canon_deduce}\n{canon_verify}\n</think>\n<ans>{ans}</ans>",
            "type": "exact_integer",
            "complexity": "multi_step"
        })
    return samples


# =============================================================================
# Family 12: Physics & Kinematics Invariants
# =============================================================================

def gen_physics_kinematics_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(count):
        subtype = random.choice(["v_at", "v2_u2_2as", "ke", "ohms"])
        if subtype == "v_at":
            u = random.choice([0, 5, 10, 15, 20])
            a = random.randint(2, 8)
            t = random.randint(3, 12)
            ans = u + a * t
            prompt = f"A body moves with initial velocity {u} m/s and constant acceleration {a} m/s^2. What is its velocity after {t} seconds?"
            canon_plan = f"<reg:plan>concept: kinematics_velocity | formula: v=u+at | u: {u} | a: {a} | t: {t}</reg:plan>"
            canon_deduce = f"<reg:deduce>v=u+a·t={u}+{a}·{t}={u}+{a*t}={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>={u})✓</reg:verify>"
        elif subtype == "v2_u2_2as":
            target_v = random.choice([10, 12, 14, 16, 20, 24])
            u = random.choice([0, 2, 4, 6])
            diff = target_v * target_v - u * u
            possible_a = [ca for ca in [1, 2, 3, 4, 5, 6, 8] if diff % (2 * ca) == 0]
            if not possible_a:
                a = 2
                u = 0
                target_v = 10
                diff = 100
            else:
                a = random.choice(possible_a)
            s = diff // (2 * a)
            ans = target_v
            prompt = f"An object with initial speed {u} m/s accelerates at {a} m/s^2 across a distance of {s} meters. What is its final speed in m/s?"
            canon_plan = f"<reg:plan>concept: kinematics_work_energy | formula: v²=u²+2as | u: {u} | a: {a} | s: {s}</reg:plan>"
            canon_deduce = f"<reg:deduce>v²={u}²+2·{a}·{s}={u*u}+{2*a*s}={diff} | v=√{diff}={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}^2=={u}^2+2*{a}*{s})✓</reg:verify>"
        elif subtype == "ke":
            m = random.choice([2, 4, 6, 8, 10, 12])
            v = random.choice([3, 4, 5, 6, 8, 10])
            ans = (m * v * v) // 2
            prompt = f"Calculate the kinetic energy in Joules of an object of mass {m} kg moving at a velocity of {v} m/s."
            canon_plan = f"<reg:plan>concept: kinetic_energy | formula: E_k=0.5*m*v^2 | m: {m} | v: {v}</reg:plan>"
            canon_deduce = f"<reg:deduce>E_k=0.5·{m}·{v}²={m//2}·{v*v}={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>0)✓</reg:verify>"
        else:
            I = random.randint(2, 10)
            R = random.choice([5, 10, 12, 15, 20, 25, 30, 50])
            ans = I * R
            prompt = f"A circuit carries a current of {I} Amperes through a resistor of {R} Ohms. What is the potential difference (voltage) across the resistor in Volts?"
            canon_plan = f"<reg:plan>concept: ohms_law | formula: V=I*R | I: {I} | R: {R}</reg:plan>"
            canon_deduce = f"<reg:deduce>V=I·R={I}·{R}={ans} ∴ ans={ans}</reg:deduce>"
            canon_verify = f"<reg:verify>verify({ans}>0 ∧ {ans}=={I}*{R})✓</reg:verify>"

        samples.append({
            "id": f"concept_phys_{i}",
            "domain": "Physics Invariants",
            "concept": "physics_kinematics",
            "prompt": prompt,
            "ground_truth": str(ans),
            "canonical_plan": canon_plan,
            "canonical_deduce": canon_deduce,
            "canonical_verify": canon_verify,
            "full_target": f"<think>\n{canon_plan}\n{canon_deduce}\n{canon_verify}\n</think>\n<ans>{ans}</ans>",
            "type": "exact_integer",
            "complexity": "multi_step"
        })
    return samples


# =============================================================================
# Register 0: Factual Bypass Seeds (Expanded 55+ Scientific & Academic Seeds)
# =============================================================================

FACTUAL_BYPASS_SEEDS = [
    ("What is the capital of Peru?", "Lima"),
    ("What is the chemical symbol for Tungsten?", "W"),
    ("Who wrote the novel 'Pride and Prejudice'?", "Jane Austen"),
    ("What is the boiling point of water at standard sea level in Celsius?", "100"),
    ("In which year did the Apollo 11 moon landing occur?", "1969"),
    ("What is the atomic number of Carbon?", "6"),
    ("Which planet in our solar system is known as the Red Planet?", "Mars"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the currency of Japan?", "yen"),
    ("What is the capital of Australia?", "Canberra"),
    ("What gas do plants absorb during photosynthesis?", "carbon dioxide"),
    ("How many continents are on Earth?", "7"),
    ("What is the hardest natural mineral on Mohs scale?", "diamond"),
    ("Who formulated universal gravitation?", "Isaac Newton"),
    ("What is the capital city of Canada?", "Ottawa"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("Which ocean is the largest by surface area?", "Pacific Ocean"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is the speed of light in vacuum in km/s approximately?", "300000"),
    ("What is the largest organ of the human body?", "skin"),
    ("What is the atomic number of Hydrogen?", "1"),
    ("What is the atomic number of Oxygen?", "8"),
    ("What is the atomic number of Nitrogen?", "7"),
    ("What is the atomic number of Helium?", "2"),
    ("What is the atomic number of Gold?", "79"),
    ("What is the chemical symbol for Potassium?", "K"),
    ("What is the chemical symbol for Sodium?", "Na"),
    ("What is the chemical symbol for Iron?", "Fe"),
    ("What is the chemical symbol for Silver?", "Ag"),
    ("What is the chemical symbol for Lead?", "Pb"),
    ("What is the chemical symbol for Mercury?", "Hg"),
    ("What is the chemical symbol for Tin?", "Sn"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the chemical formula for carbon dioxide?", "CO2"),
    ("What is the chemical formula for methane?", "CH4"),
    ("What is the freezing point of water at standard pressure in Celsius?", "0"),
    ("What is the SI unit of electrical resistance?", "Ohm"),
    ("What is the SI unit of force?", "Newton"),
    ("What is the SI unit of energy?", "Joule"),
    ("What is the SI unit of frequency?", "Hertz"),
    ("What is the SI unit of electrical capacitance?", "Farad"),
    ("What is the SI unit of pressure?", "Pascal"),
    ("What is the SI unit of magnetic flux density?", "Tesla"),
    ("Who formulated the theory of general relativity?", "Albert Einstein"),
    ("Who proposed the uncertainty principle in quantum mechanics?", "Werner Heisenberg"),
    ("Which planet is the largest planet in our solar system?", "Jupiter"),
    ("Which planet is closest to the Sun?", "Mercury"),
    ("What is the name of the galaxy containing our Solar System?", "Milky Way"),
    ("How many natural satellites does Earth have?", "1"),
    ("What organelle is known as the powerhouse of the cell?", "mitochondria"),
    ("What molecule carries genetic instructions in all living organisms?", "DNA"),
    ("What is the normal human body temperature in Celsius?", "37"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Nepal?", "Kathmandu"),
    ("What is the currency of the United Kingdom?", "pound sterling"),
    ("What is the highest mountain peak on Earth?", "Mount Everest"),
    ("What is the longest river in the world?", "Nile River"),
    ("In what year did World War II end?", "1945"),
]

def gen_factual_bypass_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    idx = 0
    while len(samples) < count:
        for q, a in FACTUAL_BYPASS_SEEDS:
            if len(samples) >= count:
                break
            samples.append({
                "id": f"concept_bypass_{idx}",
                "domain": "Factual Memory",
                "concept": "factual_bypass",
                "prompt": q,
                "ground_truth": a,
                "canonical_plan": "",
                "canonical_deduce": "",
                "canonical_verify": "",
                "full_target": f"<reg:bypass>{a}</reg:bypass>",
                "type": "short_answer",
                "complexity": "zero_shot"
            })
            idx += 1
    return samples


# =============================================================================
# Register 3: Intent Dialogue Seeds
# =============================================================================

DIALOGUE_SEEDS = [
    (
        "A colleague took credit for my presentation idea in front of leadership. How should I approach them?",
        "intent: assert-boundaries | tone: professional | strategy: direct-private-dialogue",
        "Schedule a private 1-on-1 meeting. Frame the discussion constructively: express appreciation for the positive reception, but firmly clarify that collaborative contributions must be recognized openly."
    ),
    (
        "A client is frustrated that our server went down for 2 hours during peak business. How do I reply?",
        "intent: de-escalate | tone: empathetic | strategy: acknowledge-and-remediate",
        "I sincerely apologize for the disruption during your peak operational hours. We fully understand the business impact. The issue was resolved within 2 hours, and here is our immediate plan to prevent recurrence."
    ),
    (
        "My friend keeps canceling our plans at the last minute. What is the kindest way to address this?",
        "intent: communicate-feelings | tone: caring | strategy: non-accusatory-check-in",
        "Hey, I really value our friendship and love spending time together. I noticed plans have had to be rescheduled often recently. Is everything okay on your end? Let's choose a time that truly works for you."
    ),
    (
        "How do I decline an invitation to speak at a conference because I am overbooked?",
        "intent: decline-gracefully | tone: appreciative | strategy: clear-refusal-with-alternative",
        "Thank you so much for the invitation to speak. Due to existing project commitments, I am unable to participate this term. I appreciate you thinking of me and wish you a successful conference."
    ),
    (
        "A junior engineer submitted a pull request with multiple subtle bugs. How should I review it?",
        "intent: mentor | tone: encouraging | strategy: constructive-educational-feedback",
        "Great effort on tackling this feature! I noticed a couple of boundary edge cases in the data validation layer. Let's walk through them together so we can get this pull request ready to merge."
    ),
]

def gen_intent_dialogue_samples(count: int) -> List[Dict[str, Any]]:
    samples = []
    idx = 0
    while len(samples) < count:
        for q, intent_scratch, reply in DIALOGUE_SEEDS:
            if len(samples) >= count:
                break
            samples.append({
                "id": f"concept_dialogue_{idx}",
                "domain": "Intent Dialogue",
                "concept": "intent_scratchpad",
                "prompt": q,
                "ground_truth": reply,
                "canonical_plan": f"<reg:intent>{intent_scratch}</reg:intent>",
                "canonical_deduce": "",
                "canonical_verify": "",
                "full_target": f"<reg:intent>{intent_scratch}</reg:intent>\n{reply}",
                "type": "dialogue",
                "complexity": "intent_driven"
            })
            idx += 1
    return samples


# =============================================================================
# Dataset Assembly & CLI
# =============================================================================

def build_full_concept_curriculum(total_count: int = 5000) -> List[Dict[str, Any]]:
    """
    Constructs a balanced Register v2.1 Curriculum (5,000 samples):
      - 84% (4,200 samples): Competition-Grade Concept Mathematics & Science (12 Invariant Families, 350 each)
      - 8% (400 samples): Factual Bypass (<reg:bypass>)
      - 8% (400 samples): Intent Scratchpad Dialogue (<reg:intent>)
    """
    num_families = 12
    bypass_count = max(50, int(total_count * 0.08))
    dialogue_count = max(50, int(total_count * 0.08))
    math_count = total_count - bypass_count - dialogue_count
    per_family = math_count // num_families
    remainder = math_count - (per_family * num_families)
    if remainder > 0:
        dialogue_count += remainder

    print(f"\n[CURRICULUM] Generating {total_count} Register v2.1 samples:")
    print(f"  - Concept Math & Science: {per_family * num_families} samples ({per_family} x {num_families} Invariant Families)")
    print(f"  - Zero-Shot Factual Bypass: {bypass_count} samples")
    print(f"  - Intent Dialogue: {dialogue_count} samples")

    dataset: List[Dict[str, Any]] = []
    dataset.extend(gen_polynomial_invariant_samples(per_family))
    dataset.extend(gen_chinese_remainder_samples(per_family))
    dataset.extend(gen_modular_totient_samples(per_family))
    dataset.extend(gen_legendre_samples(per_family))
    dataset.extend(gen_combinatorics_samples(per_family))
    dataset.extend(gen_geometry_samples(per_family))
    dataset.extend(gen_diophantine_and_rates_samples(per_family))
    dataset.extend(gen_linear_algebra_samples(per_family))
    dataset.extend(gen_combinations_samples(per_family))
    dataset.extend(gen_sequences_recurrences_samples(per_family))
    dataset.extend(gen_arithmetic_word_problem_samples(per_family))
    dataset.extend(gen_physics_kinematics_samples(per_family))
    dataset.extend(gen_factual_bypass_samples(bypass_count))
    dataset.extend(gen_intent_dialogue_samples(dialogue_count))

    random.shuffle(dataset)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build SymboLM Register v2.0/v2.1 Concept Curriculum")
    parser.add_argument("--count", type=int, default=5000, help="Total number of samples")
    parser.add_argument("--output", type=str, default="./data/concept_math_curriculum.jsonl", help="Output file path")
    args = parser.parse_args()

    data = build_full_concept_curriculum(args.count)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(out_p, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[COMPLETE] Successfully generated {len(data)} samples saved to: {out_p}")


if __name__ == "__main__":
    main()
