"""
SymboLM 200-Question Mixed Evaluation Benchmark Generator
=========================================================
Generates exactly 200 curated, contamination-resistant evaluation questions:
  - 130 Competition & Invariant Mathematics (7 Invariant Families + Arithmetic)
  -  45 General Knowledge / Factual QA (Testing <reg:bypass> Zero-Shot Routing)
  -  25 Communicative & Intent Dialogue (Testing <reg:intent> Scratchpads)

Author: Sanish Gyawali
AI Systems Exoskeleton: Antigravity (Google DeepMind)
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

RANDOM_SEED = 2026
random.seed(RANDOM_SEED)

ROOT_DIR = Path(__file__).resolve().parent.parent

# =============================================================================
# 1. MATHEMATICAL QUESTION GENERATORS (130 Questions)
# =============================================================================

def gen_polynomial_symmetric(count: int = 20) -> List[Dict[str, Any]]:
    """Generates x + 1/x = k questions for x^2, x^3, x^4."""
    samples = []
    k_values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    for i, k in enumerate(k_values[:count]):
        target_p = (i % 3) + 2  # 2, 3, or 4
        # Recurrence: T_0 = 2, T_1 = k, T_n = k * T_{n-1} - T_{n-2}
        t = [2, k]
        for n in range(2, target_p + 1):
            t.append(k * t[n-1] - t[n-2])
        ans = t[target_p]
        
        sup = {2: "²", 3: "³", 4: "⁴"}[target_p]
        samples.append({
            "id": f"math_poly_{i+1:02d}",
            "category": "Math: Polynomial Invariants",
            "prompt": f"If x + 1/x = {k}, find the exact value of x^{target_p} + 1/x^{target_p}.",
            "ground_truth": str(ans),
            "expected_register": "reg:deduce",
            "english_baseline_tokens": 180 + target_p * 40
        })
    return samples


def gen_crt_modular_systems(count: int = 20) -> List[Dict[str, Any]]:
    """Generates systems of congruences solvable via CRT."""
    samples = []
    mod_pairs = [
        (3, 5), (5, 7), (7, 11), (3, 7), (5, 11),
        (7, 13), (9, 11), (4, 7), (5, 9), (8, 11),
        (3, 11), (5, 13), (7, 17), (11, 13), (4, 9),
        (5, 8), (7, 9), (11, 17), (13, 17), (9, 13)
    ]
    for i, (m1, m2) in enumerate(mod_pairs[:count]):
        M = m1 * m2
        target_x = random.randint(1, M - 1)
        r1 = target_x % m1
        r2 = target_x % m2
        samples.append({
            "id": f"math_crt_{i+1:02d}",
            "category": "Math: Modular Congruence & CRT",
            "prompt": f"Find the smallest positive integer n such that n ≡ {r1} (mod {m1}) and n ≡ {r2} (mod {m2}).",
            "ground_truth": str(target_x),
            "expected_register": "reg:deduce",
            "english_baseline_tokens": 220
        })
    return samples


def gen_modular_exp(count: int = 20) -> List[Dict[str, Any]]:
    """Generates modular exponentiation problems a^b mod m."""
    samples = []
    configs = [
        (2, 50, 13), (3, 40, 11), (5, 30, 7), (2, 70, 17), (3, 50, 19),
        (7, 25, 11), (2, 100, 13), (5, 45, 19), (3, 80, 23), (2, 60, 31),
        (3, 45, 13), (7, 35, 13), (2, 85, 29), (5, 60, 31), (3, 100, 17),
        (2, 90, 19), (7, 40, 19), (3, 75, 7), (5, 50, 11), (2, 120, 23)
    ]
    for i, (base, exp, mod) in enumerate(configs[:count]):
        ans = pow(base, exp, mod)
        samples.append({
            "id": f"math_modexp_{i+1:02d}",
            "category": "Math: Modular Exponentiation",
            "prompt": f"Compute the remainder when {base}^{exp} is divided by {mod}.",
            "ground_truth": str(ans),
            "expected_register": "reg:deduce",
            "english_baseline_tokens": 240
        })
    return samples


def gen_legendre_valuations(count: int = 15) -> List[Dict[str, Any]]:
    """Generates prime valuation problems nu_p(N!)."""
    samples = []
    configs = [
        (100, 3), (100, 5), (100, 7), (150, 5), (200, 3),
        (200, 7), (250, 11), (300, 5), (500, 3), (500, 7),
        (1000, 5), (1000, 11), (2026, 7), (2026, 5), (2026, 13)
    ]
    for i, (N, p) in enumerate(configs[:count]):
        val = 0
        k = p
        while k <= N:
            val += N // k
            k *= p
        samples.append({
            "id": f"math_legendre_{i+1:02d}",
            "category": "Math: Legendre Valuation",
            "prompt": f"Find the highest power of {p} that divides {N}!.",
            "ground_truth": str(val),
            "expected_register": "reg:deduce",
            "english_baseline_tokens": 190
        })
    return samples


def gen_combinatorics(count: int = 15) -> List[Dict[str, Any]]:
    """Generates derangements and binomial combinations."""
    samples = []
    # Derangements D_n
    d_table = {1: 0, 2: 1, 3: 2, 4: 9, 5: 44, 6: 265, 7: 1854}
    for n in range(3, 8):
        samples.append({
            "id": f"math_comb_d{n}",
            "category": "Math: Combinatorics & Recurrences",
            "prompt": f"Find the number of derangements of {n} items (permutations with no fixed points).",
            "ground_truth": str(d_table[n]),
            "expected_register": "reg:deduce",
            "english_baseline_tokens": 210
        })
    
    # Combinations C(n, k)
    comb_configs = [
        (10, 3), (12, 4), (15, 2), (14, 5), (16, 3),
        (20, 2), (18, 3), (11, 4), (13, 3), (21, 2)
    ]
    for i, (n, k) in enumerate(comb_configs[:count - len(samples)]):
        ans = math.comb(n, k)
        samples.append({
            "id": f"math_comb_c{i+1:02d}",
            "category": "Math: Combinatorics & Recurrences",
            "prompt": f"Compute the binomial coefficient C({n}, {k}).",
            "ground_truth": str(ans),
            "expected_register": "reg:deduce",
            "english_baseline_tokens": 160
        })
    return samples


def gen_geometry_invariants(count: int = 20) -> List[Dict[str, Any]]:
    """Generates geometric invariant problems: Area = r * s, circles, cylinders."""
    samples = []
    # Right triangle inradius triples: (r, s, Area)
    # Area = r * s, P = 2*s
    triples = [
        (6, 36, 216), (4, 20, 80), (3, 15, 45), (5, 30, 150),
        (2, 12, 24), (8, 48, 384), (7, 42, 294), (10, 60, 600),
        (9, 54, 486), (12, 72, 864)
    ]
    for i, (r, s, area) in enumerate(triples):
        P = 2 * s
        samples.append({
            "id": f"math_geom_inradius_{i+1:02d}",
            "category": "Math: Geometric Invariants",
            "prompt": f"In a right triangle with inradius r = {r} and perimeter P = {P}, find the area of the triangle.",
            "ground_truth": str(area),
            "expected_register": "reg:plan",
            "english_baseline_tokens": 260
        })
    
    # Cylinder / rectangle geometry
    cyl_configs = [
        (3, 10), (4, 12), (5, 8), (6, 14), (7, 10),
        (2, 20), (8, 6), (5, 15), (4, 25), (10, 5)
    ]
    for i, (r, h) in enumerate(cyl_configs[:count - len(samples)]):
        # Base Area = r^2, volume / pi = r^2 * h
        v_pi = r * r * h
        samples.append({
            "id": f"math_geom_cyl_{i+1:02d}",
            "category": "Math: Geometric Invariants",
            "prompt": f"A cylinder has radius r = {r} and height h = {h}. What is its volume divided by pi?",
            "ground_truth": str(v_pi),
            "expected_register": "reg:deduce",
            "english_baseline_tokens": 180
        })
    return samples


def gen_diophantine_workrates(count: int = 20) -> List[Dict[str, Any]]:
    """Generates difference of squares x^2 - y^2 = N and harmonic work rates."""
    samples = []
    # Diophantine: x^2 - y^2 = N -> (x - y)(x + y) = N
    dio_pairs = [
        (15, 8), (17, 12), (20, 13), (25, 16), (30, 19),
        (35, 22), (40, 25), (28, 17), (24, 15), (18, 9)
    ]
    for i, (x, y) in enumerate(dio_pairs):
        N = x * x - y * y
        samples.append({
            "id": f"math_diophantine_{i+1:02d}",
            "category": "Math: Diophantine & Work Rates",
            "prompt": f"If positive integers x and y satisfy x - y = {x - y} and x² - y² = {N}, find the value of x.",
            "ground_truth": str(x),
            "expected_register": "reg:deduce",
            "english_baseline_tokens": 200
        })
    
    # Work rates: A takes T1 days, B takes T2 days -> Together T = (T1 * T2)/(T1 + T2)
    work_pairs = [
        (6, 12, 4), (10, 15, 6), (12, 24, 8), (20, 30, 12), (15, 30, 10),
        (8, 24, 6), (12, 60, 10), (14, 35, 10), (18, 36, 12), (20, 80, 16)
    ]
    for i, (t1, t2, tgt) in enumerate(work_pairs[:count - len(samples)]):
        samples.append({
            "id": f"math_workrate_{i+1:02d}",
            "category": "Math: Diophantine & Work Rates",
            "prompt": f"Worker A can complete a task in {t1} days and Worker B can complete it in {t2} days. How many days will it take working together?",
            "ground_truth": str(tgt),
            "expected_register": "reg:deduce",
            "english_baseline_tokens": 220
        })
    return samples


# =============================================================================
# 2. GENERAL KNOWLEDGE & FACTUAL QA (45 Questions -> <reg:bypass>)
# =============================================================================

FACTUAL_QA_ITEMS = [
    # Chemistry & Elements
    ("What is the atomic number of Carbon?", "6"),
    ("What is the atomic number of Oxygen?", "8"),
    ("What is the atomic number of Gold?", "79"),
    ("What is the atomic number of Iron?", "26"),
    ("What is the atomic number of Helium?", "2"),
    ("What is the chemical symbol for Tungsten?", "W"),
    ("What is the chemical symbol for Potassium?", "K"),
    ("What is the chemical symbol for Sodium?", "Na"),
    ("What is the chemical symbol for Silver?", "Ag"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the chemical formula for carbon dioxide?", "CO2"),
    
    # Physics & Constants
    ("What is the boiling point of water at standard sea level in Celsius?", "100"),
    ("What is the freezing point of water at sea level in Celsius?", "0"),
    ("Approximately what is the speed of light in vacuum in thousands of km/s?", "300"),
    ("What fundamental particle carries a negative electric charge?", "electron"),
    ("What fundamental particle has no electric charge?", "neutron"),
    
    # Geography & Capitals
    ("What is the capital city of Peru?", "Lima"),
    ("What is the capital city of Australia?", "Canberra"),
    ("What is the capital city of Canada?", "Ottawa"),
    ("What is the capital city of Japan?", "Tokyo"),
    ("What is the capital city of France?", "Paris"),
    ("What is the capital city of Germany?", "Berlin"),
    ("What is the capital city of Brazil?", "Brasilia"),
    ("What is the capital city of Egypt?", "Cairo"),
    ("What is the capital city of Nepal?", "Kathmandu"),
    ("Which ocean is the largest by surface area on Earth?", "Pacific Ocean"),
    ("How many continents are there on Earth?", "7"),
    ("Which continent has the largest land area?", "Asia"),
    ("What is the tallest mountain above sea level on Earth?", "Mount Everest"),
    
    # Biology & Earth Science
    ("What gas do green plants absorb from the air during photosynthesis?", "carbon dioxide"),
    ("What is the hardest natural mineral on Mohs hardness scale?", "diamond"),
    ("What is the largest organ of the human body?", "skin"),
    ("What organ in the human body pumps blood?", "heart"),
    ("How many bones are in the adult human skeleton?", "206"),
    
    # History, Space & Literature
    ("In which year did the Apollo 11 moon landing take place?", "1969"),
    ("Which planet in our solar system is known as the Red Planet?", "Mars"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("Who formulated the universal law of gravitation?", "Isaac Newton"),
    ("Who discovered penicillin in 1928?", "Alexander Fleming"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("Who wrote the play 'Hamlet'?", "William Shakespeare"),
    ("Who wrote the novel 'Pride and Prejudice'?", "Jane Austen"),
    ("What is the official currency of Japan?", "yen"),
    ("What is the official currency of the United Kingdom?", "pound")
]

def gen_factual_bypass(count: int = 45) -> List[Dict[str, Any]]:
    samples = []
    for i, (q, a) in enumerate(FACTUAL_QA_ITEMS[:count]):
        samples.append({
            "id": f"gk_bypass_{i+1:02d}",
            "category": "Factual QA: Zero-Shot Bypass",
            "prompt": q,
            "ground_truth": a,
            "expected_register": "reg:bypass",
            "english_baseline_tokens": 120
        })
    return samples


# =============================================================================
# 3. INTENT DIALOGUE & ADVISORY (25 Questions -> <reg:intent>)
# =============================================================================

INTENT_DIALOGUE_ITEMS = [
    ("A colleague took credit for my presentation idea in front of leadership. How should I approach them?",
     "Schedule a private 1-on-1 meeting. Frame the discussion constructively: express appreciation for the positive reception, but firmly clarify that collaborative contributions must be recognized openly.",
     "assert-boundaries"),
    ("A client is frustrated that our server went down for 2 hours during peak business. How do I reply?",
     "I sincerely apologize for the disruption during your peak operational hours. We fully understand the business impact. The issue was resolved within 2 hours, and here is our immediate plan to prevent recurrence.",
     "de-escalate"),
    ("My friend keeps canceling our plans at the last minute. What is the kindest way to address this?",
     "Hey, I really value our friendship and love spending time together. I noticed plans have had to be rescheduled often recently. Is everything okay on your end? Let's choose a time that truly works for you.",
     "communicate-feelings"),
    ("How do I decline an invitation to speak at a conference because I am overbooked?",
     "Thank you so much for the invitation to speak. Due to existing project commitments, I am unable to participate this term. I appreciate you thinking of me and wish you a successful conference.",
     "decline-gracefully"),
    ("A junior engineer submitted a pull request with multiple subtle bugs. How should I review it?",
     "Great effort on tackling this feature! I noticed a couple of boundary edge cases in the data validation layer. Let's walk through them together so we can get this pull request ready to merge.",
     "mentor-constructively"),
    ("An investor is asking why our product has lower raw benchmark scores than competitors. How should we answer?",
     "Our architecture prioritizes deterministic reliability, token efficiency, and sub-2-second local execution over brute-forced benchmark vanity scores. We deliver verified proofs at 75% lower compute cost.",
     "strategic-differentiation"),
    ("A team member consistently interrupts others in design reviews. How should the team lead intervene?",
     "In our reviews, every perspective is critical. Let's ensure each speaker finishes their walkthrough before we open the floor for critique and follow-up questions.",
     "facilitate-discussion"),
    ("How do I ask my manager for a compensation review based on recent technical deliveries?",
     "I would like to schedule time to discuss my compensation. Over the past year, I led the delivery of several key systems that significantly improved team throughput. Let's review my impact and market alignment.",
     "advocate-compensation"),
    ("A customer wants a full refund after using our software for 45 days past the 30-day policy. How do we respond?",
     "Thank you for contacting us. While our refund policy applies within the first 30 days, we want to ensure you get value from the product. What specific technical roadblocks did you encounter that we can resolve?",
     "policy-enforcement-with-support"),
    ("A stakeholder wants an unplanned high-risk feature deployed on Friday evening. How do you respond?",
     "We understand this feature is exciting, but our deployment safety policy restricts high-risk production updates before weekends. We will stage and verify it thoroughly for first deployment Monday morning.",
     "risk-management"),
    ("How do you gently tell a coworker their communication style is coming across as overly harsh?",
     "I value your directness and dedication to quality. Sometimes, however, the phrasing in Slack can feel dismissive to team members. Focusing feedback on the code rather than the person helps keep morale high.",
     "peer-feedback"),
    ("A vendor sent an invoice that is 20% higher than the agreed estimate. What is the proper email response?",
     "Thank you for sending the invoice. We noticed the total exceeds the agreed purchase order estimate by 20%. Could you provide an itemized breakdown explaining the discrepancy before we process payment?",
     "clarification-dispute"),
    ("How do I tell my direct report that their project deadline is slipping without discouraging them?",
     "Looking at the current milestone velocity, we are trending behind the projected release date. What external blockers are slowing you down, and how can we re-scope tasks to ensure a smooth launch?",
     "coaching-velocity"),
    ("A candidate was not selected after a technical interview. How do we write a polite rejection?",
     "Thank you for taking the time to interview with our engineering team. While we were impressed with your background, we have chosen to proceed with another candidate whose experience more closely matches this specific opening.",
     "respectful-rejection"),
    ("A user leaves an angry 1-star review claiming our app crashed during an important workflow. How do we reply?",
     "We are very sorry this crash interrupted your workflow. Reliability is our top priority. We have flagged this error to our engineering team. Please contact support@domain.com so we can help restore your session.",
     "public-support"),
    ("How do you decline a promotion offer because the role shifts entirely to administrative management?",
     "I am deeply honored by the offer. However, my long-term career ambition and greatest strengths lie in principal systems architecture rather than people management. I would love to continue driving technical impact.",
     "career-alignment"),
    ("How do you ask for time off for medical treatment while maintaining privacy?",
     "I am writing to request medical leave from [Start Date] to [End Date] for a scheduled health procedure and recovery. I will ensure all critical deliverables are handed off to the team prior to my departure.",
     "professional-leave"),
    ("How do you notify users about scheduled server maintenance without causing panic?",
     "We will be performing a scheduled infrastructure upgrade this Sunday between 02:00 and 04:00 UTC. The application will be temporarily offline for approximately 30 minutes to improve system stability and speed.",
     "maintenance-notice"),
    ("How do you address scope creep from a client who keeps adding features to a fixed-price contract?",
     "We would love to build these additional features! Because they fall outside our original scope of work, let's create a secondary addendum detailing the updated timeline and budget for these enhancements.",
     "scope-management"),
    ("How do you ask a senior engineer for code review feedback when they are very busy?",
     "Hi [Name], when you have a brief moment, could you take a look at PR #123? It touches the core auth pipeline and I'd really value your architectural perspective on the session handling changes.",
     "request-review"),
    ("How do you apologize to a colleague for missing an internal project sync?",
     "I apologize for missing our sync this morning; an urgent production triage came up unexpectedly. I reviewed your notes in the document and have left my comments on items 3 and 4.",
     "accountability"),
    ("How do you decline an unsolicited sales pitch on LinkedIn professionally?",
     "Thank you for reaching out. We are not currently in the market for this type of solution, but I appreciate you keeping us in mind.",
     "brief-decline"),
    ("How do you announce the departure of a beloved team member to the department?",
     "We want to announce that [Name] will be moving on to a new opportunity starting next month. We are immensely grateful for their contributions to our systems and culture, and wish them tremendous success.",
     "farewell-announcement"),
    ("How do you inform a customer that a feature they requested has been deprioritized?",
     "Thank you for your feedback regarding [Feature]. While we recognize its value, our current engineering roadmap is focused on core performance and reliability. We will re-evaluate this feature in Q3.",
     "roadmap-transparency"),
    ("How do you ask a peer to keep a conversation strictly confidential?",
     "Before we discuss this matter, I'd like to ensure this conversation remains strictly between us, as it involves sensitive organizational planning that has not yet been announced.",
     "confidentiality-preface")
]

def gen_intent_dialogue(count: int = 25) -> List[Dict[str, Any]]:
    samples = []
    for i, (q, ans, intent) in enumerate(INTENT_DIALOGUE_ITEMS[:count]):
        samples.append({
            "id": f"intent_dialogue_{i+1:02d}",
            "category": "Dialogue: Intent Scratchpad",
            "prompt": q,
            "ground_truth": ans,
            "expected_register": "reg:intent",
            "english_baseline_tokens": 150
        })
    return samples


# =============================================================================
# MAIN BENCHMARK COMPILER
# =============================================================================

def build_eval_benchmark() -> List[Dict[str, Any]]:
    benchmark = []
    
    # 1. Math (130 samples)
    math_poly = gen_polynomial_symmetric(20)
    math_crt = gen_crt_modular_systems(20)
    math_exp = gen_modular_exp(20)
    math_leg = gen_legendre_valuations(15)
    math_comb = gen_combinatorics(15)
    math_geom = gen_geometry_invariants(20)
    math_dio = gen_diophantine_workrates(20)
    
    math_all = math_poly + math_crt + math_exp + math_leg + math_comb + math_geom + math_dio
    assert len(math_all) == 130, f"Expected 130 math samples, got {len(math_all)}"
    
    # 2. General Knowledge / Factual Bypass (45 samples)
    gk_all = gen_factual_bypass(45)
    assert len(gk_all) == 45, f"Expected 45 GK samples, got {len(gk_all)}"
    
    # 3. Intent Dialogue (25 samples)
    dialogue_all = gen_intent_dialogue(25)
    assert len(dialogue_all) == 25, f"Expected 25 Dialogue samples, got {len(dialogue_all)}"
    
    # Assemble and interleave cleanly
    benchmark = math_all + gk_all + dialogue_all
    assert len(benchmark) == 200, f"Expected exactly 200 samples, got {len(benchmark)}"
    
    # Interleave categories so evaluation tests adaptive register switching
    # Seeded shuffle with stable reproducibility
    random.seed(42)
    random.shuffle(benchmark)
    
    # Re-index
    for idx, item in enumerate(benchmark, start=1):
        item["eval_id"] = f"EVAL-200-{idx:03d}"
        
    return benchmark


def main():
    out_file = ROOT_DIR / "data" / "eval_200_mixed_benchmark.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    benchmark = build_eval_benchmark()
    
    with open(out_file, "w", encoding="utf-8") as f:
        for item in benchmark:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Breakdown
    categories = {}
    for item in benchmark:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1
        
    print(f"[OK] Successfully built 200-question mixed benchmark saved to: {out_file}")
    print("\nCategory Distribution:")
    for cat, cnt in sorted(categories.items()):
        print(f"  - {cat:35s}: {cnt:3d} questions")
    print(f"  Total: {len(benchmark)} questions")


if __name__ == "__main__":
    main()
