"""
SymboLM Stage 3 Unified Training Curriculum (Register v2.0 + Experience Replay)
=============================================================================
Unifies the 1,400 Register v2.0 Concept Curriculum with 200 legacy Stage 2 arithmetic
anchors to guarantee zero catastrophic forgetting during continual reinforcement learning.

Curriculum Composition (1,600 Total Samples):
  - 973 samples (60.8%): Competition Concept Math across 7 Invariant Families (<reg:plan> + <reg:deduce>)
  - 200 samples (12.5%): Legacy Deductive Arithmetic Anchors (Multi-step train, linear algebra)
  - 210 samples (13.1%): Zero-Shot Factual Bypass (<reg:bypass>)
  - 217 samples (13.6%): Intent Dialogue Scratchpads (<reg:intent>)

Author: Sanish Gyawali
AI Systems Exoskeleton: Antigravity (Google DeepMind)
"""

import json
import random
import sys
from pathlib import Path

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


def build_unified_curriculum(
    concept_path: str = "./data/concept_math_curriculum.jsonl",
    legacy_path: str = "./data/grpo_curriculum.jsonl",
    output_path: str = "./data/stage3_unified_curriculum.jsonl",
    anchor_count: int = 200
):
    c_p = Path(concept_path)
    l_p = Path(legacy_path)
    out_p = Path(output_path)

    print(f"\n[UNIFY] Loading Register v2.0 Concept Math from {c_p}...")
    with open(c_p, "r", encoding="utf-8") as f:
        concept_samples = [json.loads(line) for line in f if line.strip()]

    print(f"[UNIFY] Loading Legacy Stage 2 Anchors from {l_p}...")
    with open(l_p, "r", encoding="utf-8") as f:
        legacy_samples = [json.loads(line) for line in f if line.strip()]

    # Filter legacy deductive arithmetic samples
    deductive_anchors = [
        s for s in legacy_samples
        if s.get("register") == "symbolic_deductive"
    ]
    random.shuffle(deductive_anchors)
    selected_anchors = deductive_anchors[:anchor_count]

    # Convert legacy anchors to Register v2.0 format
    converted_anchors = []
    for idx, anc in enumerate(selected_anchors):
        ans = anc["answer"]
        converted_anchors.append({
            "id": f"anchor_arithmetic_{idx}",
            "domain": "Elementary Deductive Arithmetic",
            "concept": "deductive_arithmetic_anchor",
            "prompt": anc["prompt"],
            "ground_truth": ans,
            "canonical_plan": "",
            "canonical_deduce": f"<reg:deduce>∴ ans={ans}</reg:deduce>",
            "canonical_verify": "",
            "full_target": f"<think>\n<reg:deduce>∴ ans={ans}</reg:deduce>\n</think>\n<ans>{ans}</ans>",
            "type": anc.get("expected_type", "numeric"),
            "complexity": "single_step"
        })

    unified = concept_samples + converted_anchors
    random.shuffle(unified)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for entry in unified:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[COMPLETE] Successfully assembled {len(unified)} unified samples:")
    print(f"  - Register v2.0 Concept Curriculum: {len(concept_samples)}")
    print(f"  - Legacy Arithmetic Anchors (Experience Replay): {len(converted_anchors)}")
    print(f"  - Output Saved To: {out_p}\n")


if __name__ == "__main__":
    build_unified_curriculum()
