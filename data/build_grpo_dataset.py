"""
SymboLM Stage 2 GRPO Curriculum Dataset Builder
================================================
Constructs a balanced 1,200-sample Tri-Register Curriculum for GRPO reinforcement learning:
  - 60% (720 samples): Symbolic Deductive Reasoning (Multi-step math, algebra, logic)
  - 20% (240 samples): Zero-Shot Factual Bypass (Direct QA with 0 thinking tokens)
  - 20% (240 samples): Intent-Scratchpad Dialogue (Interpersonal & empathetic communication)

Author: Sanish Gyawali
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import config

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Register 2: Factual Bypass Seeds (Direct QA)
# ---------------------------------------------------------------------------
FACTUAL_SEEDS = [
    ("What is the capital of Peru?", "Lima"),
    ("What is the chemical symbol for Tungsten?", "W"),
    ("Who wrote the novel 'Pride and Prejudice'?", "Jane Austen"),
    ("What is the boiling point of water at standard sea level in Celsius?", "100"),
    ("In which year did the Apollo 11 moon landing occur?", "1969"),
    ("What is the largest organ of the human body?", "skin"),
    ("What is the atomic number of Carbon?", "6"),
    ("Which planet in our solar system is known as the Red Planet?", "Mars"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the currency of Japan?", "yen"),
    ("What is the capital of Australia?", "Canberra"),
    ("What gas do plants absorb during photosynthesis?", "carbon dioxide"),
    ("How many continents are on Earth?", "7"),
    ("What is the hardest natural mineral on Mohs scale?", "diamond"),
    ("Who formulated the laws of motion and universal gravitation?", "Isaac Newton"),
    ("What is the capital city of Canada?", "Ottawa"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("Which ocean is the largest by surface area?", "Pacific Ocean"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is the speed of light in vacuum approximately in km/s?", "300000"),
]

# ---------------------------------------------------------------------------
# Register 3: Intent-Scratchpad Dialogue Seeds
# ---------------------------------------------------------------------------
DIALOGUE_SEEDS = [
    (
        "A colleague took credit for my presentation idea in front of leadership. How should I approach them?",
        "intent: assert-boundaries | tone: professional | strategy: direct-private-dialogue",
        "Schedule a private 1-on-1 meeting. Frame the discussion constructively: mention that you were glad the idea was received well, but clarify that in the future you expect collaborative credit to be acknowledged openly."
    ),
    (
        "A client is extremely frustrated that our server went down for 2 hours during peak business. How do I reply?",
        "intent: de-escalate | tone: empathetic | strategy: acknowledge-and-remediate",
        "I sincerely apologize for the disruption caused during your peak hours. We understand the direct impact on your business. Our team resolved the root cause within 2 hours, and here is our immediate action plan to prevent recurrence."
    ),
    (
        "My friend keeps canceling our plans at the last minute. What is the kindest way to address this?",
        "intent: communicate-feelings | tone: caring | strategy: non-accusatory-check-in",
        "Hey, I value our friendship and love spending time with you. I noticed plans have had to be rescheduled often recently. Is everything okay on your end? Let's pick a time that genuinely works for both of us."
    ),
    (
        "How do I decline an invitation to speak at a conference because I am completely overbooked?",
        "intent: decline-gracefully | tone: appreciative | strategy: clear-refusal-with-alternative",
        "Thank you so much for the invitation to speak. Unfortunately, due to existing commitments, I am unable to participate this term. I appreciate you thinking of me and wish you a fantastic event."
    ),
    (
        "A junior engineer on my team submitted a pull request with multiple subtle bugs. How should I review it?",
        "intent: mentor | tone: encouraging | strategy: constructive-educational-feedback",
        "Great effort on tackling this feature! I noticed a couple of edge cases in the data validation layer that could lead to unexpected behavior. Let's walk through them together so we can get this ready to merge."
    ),
]


def generate_synthetic_deductive_prompts(count: int) -> List[Dict[str, Any]]:
    """Generate synthetic multi-step arithmetic & algebra prompts."""
    items = []
    for i in range(count):
        ptype = i % 4
        if ptype == 0:
            # Multi-stop train arithmetic
            init = random.randint(50, 150)
            off1 = random.randint(10, 30)
            on1 = random.randint(15, 40)
            off2 = random.randint(5, 20)
            on2 = random.randint(10, 25)
            ans = init - off1 + on1 - off2 + on2
            prompt = (
                f"A train starts its journey with {init} passengers. At the first stop, {off1} passengers get off "
                f"and {on1} passengers board. At the second stop, {off2} passengers get off and {on2} board. "
                f"How many passengers are on the train now?"
            )
            items.append({
                "id": f"deductive_train_{i}",
                "register": "symbolic_deductive",
                "prompt": prompt,
                "answer": str(ans),
                "expected_type": "numeric"
            })
        elif ptype == 1:
            # Linear equation: a*x + b = c
            x_val = random.randint(2, 25)
            a = random.randint(2, 9)
            b = random.randint(1, 50)
            c = a * x_val + b
            prompt = f"Solve for x: {a}x + {b} = {c}"
            items.append({
                "id": f"deductive_linear_{i}",
                "register": "symbolic_deductive",
                "prompt": prompt,
                "answer": str(x_val),
                "expected_type": "numeric"
            })
        elif ptype == 2:
            # Resource allocation (apples/inventory)
            orig = random.randint(30, 120)
            sell_rate = random.randint(3, 8)
            days = random.randint(2, 5)
            received = random.randint(15, 45)
            ans = orig - (sell_rate * days) + received
            prompt = (
                f"A bakery has {orig} loaves of bread. Each day for {days} days, they sell {sell_rate} loaves. "
                f"On the {days}th day, the baker bakes {received} fresh loaves. How many loaves are left?"
            )
            items.append({
                "id": f"deductive_bakery_{i}",
                "register": "symbolic_deductive",
                "prompt": prompt,
                "answer": str(ans),
                "expected_type": "numeric"
            })
        else:
            # Exponential / Power comparison
            base = random.choice([2, 3, 5])
            exp = random.randint(3, 7)
            val = base ** exp
            offset = random.randint(1, 10)
            test_val = val - offset if random.random() > 0.5 else val + offset
            ans = "true" if val > test_val else "false"
            prompt = f"Is {base}^{exp} strictly greater than {test_val}?"
            items.append({
                "id": f"deductive_exp_{i}",
                "register": "symbolic_deductive",
                "prompt": prompt,
                "answer": ans,
                "expected_type": "boolean"
            })
    return items


def generate_factual_bypass_prompts(count: int) -> List[Dict[str, Any]]:
    """Generate 240 factual bypass prompts."""
    items = []
    idx = 0
    while len(items) < count:
        for q, a in FACTUAL_SEEDS:
            if len(items) >= count:
                break
            items.append({
                "id": f"factual_bypass_{idx}",
                "register": "factual_bypass",
                "prompt": q,
                "answer": a,
                "expected_type": "short_answer"
            })
            idx += 1
    return items


def generate_dialogue_prompts(count: int) -> List[Dict[str, Any]]:
    """Generate 240 intent-scratchpad dialogue prompts."""
    items = []
    idx = 0
    while len(items) < count:
        for q, intent, resp in DIALOGUE_SEEDS:
            if len(items) >= count:
                break
            items.append({
                "id": f"intent_dialogue_{idx}",
                "register": "intent_dialogue",
                "prompt": q,
                "answer": resp,
                "target_intent": intent,
                "expected_type": "dialogue"
            })
            idx += 1
    return items


def load_gsm8k_prompts(target_count: int) -> List[Dict[str, Any]]:
    """Attempt to load real GSM8K problems from local cache or datasets."""
    items = []
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        for i, ex in enumerate(ds):
            if len(items) >= target_count:
                break
            match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", ex["answer"])
            if not match:
                continue
            ans = match.group(1).replace(",", "").strip()
            items.append({
                "id": f"gsm8k_{i}",
                "register": "symbolic_deductive",
                "prompt": ex["problem"],
                "answer": ans,
                "expected_type": "numeric"
            })
    except Exception as e:
        print(f"[Notice] Offline or datasets unavailable ({e}), using high-precision synthetic pool.")
    return items


def build_curriculum(output_path: Path):
    """Compile exactly 1,200 curated prompts across the 60/20/20 mix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Building SymboLM Stage 2 GRPO Curriculum (1,200 Prompts)")
    print(f"Target Output: {output_path}")
    print("=" * 65)

    # 1. Register 1: Deductive Reasoning (720 samples)
    target_deductive = 720
    gsm8k_items = load_gsm8k_prompts(target_deductive)
    remaining_deductive = target_deductive - len(gsm8k_items)
    synth_items = generate_synthetic_deductive_prompts(remaining_deductive)
    deductive_items = gsm8k_items + synth_items
    random.shuffle(deductive_items)
    deductive_items = deductive_items[:target_deductive]
    print(f"[OK] Register 1 (Symbolic Deductive) : {len(deductive_items)} prompts")

    # 2. Register 2: Factual Bypass (240 samples)
    target_bypass = 240
    bypass_items = generate_factual_bypass_prompts(target_bypass)
    print(f"[OK] Register 2 (Factual Bypass)    : {len(bypass_items)} prompts")

    # 3. Register 3: Intent Dialogue (240 samples)
    target_dialogue = 240
    dialogue_items = generate_dialogue_prompts(target_dialogue)
    print(f"[OK] Register 3 (Intent Dialogue)   : {len(dialogue_items)} prompts")

    # Total Curriculum
    all_prompts = deductive_items + bypass_items + dialogue_items
    random.shuffle(all_prompts)
    print(f"\nTotal Curriculum Size: {len(all_prompts)} prompts")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_prompts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] Saved successfully to {output_path}!")


def main():
    parser = argparse.ArgumentParser(description="Build GRPO Curriculum")
    parser.add_argument(
        "--output",
        type=str,
        default="./data/grpo_curriculum.jsonl",
        help="Destination JSONL path"
    )
    args = parser.parse_args()
    build_curriculum(Path(args.output).resolve())


if __name__ == "__main__":
    main()
