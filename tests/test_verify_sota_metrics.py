import json
import pytest
from pathlib import Path
from eval.verify_sota_metrics import verify_and_compute

def test_verify_and_compute(tmp_path: Path):
    test_jsonl = tmp_path / "test_receipts.jsonl"
    records = [
        {"arm": "A", "budget": 256, "is_correct": False, "tokens_emitted": 256, "dynamic_joules": 150.0},
        {"arm": "A", "budget": 2048, "is_correct": True,  "tokens_emitted": 800, "dynamic_joules": 450.0},
        {"arm": "B", "budget": 256, "is_correct": True,  "tokens_emitted": 6,   "dynamic_joules": 15.0},
        {"arm": "B", "budget": 2048, "is_correct": True,  "tokens_emitted": 6,   "dynamic_joules": 15.0},
    ]
    with open(test_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    res = verify_and_compute(test_jsonl)
    assert res["records_verified"] == 4
    metrics = {f"{m['arm']}_{m['budget']}": m for m in res["metrics"]}
    
    # Arm A choked at 256 tokens: 0% acc, while at 2048 it had 100% acc -> BCF should be 1.0
    assert metrics["A_256"]["acc"] == 0.0
    assert metrics["A_2048"]["acc"] == 1.0
    assert metrics["A_256"]["bcf"] == 1.0

    # Arm B had 100% accuracy at both budgets with only 6 tokens -> BCF should be 0.0
    assert metrics["B_256"]["acc"] == 1.0
    assert metrics["B_2048"]["acc"] == 1.0
    assert metrics["B_256"]["bcf"] == 0.0
    # Overthinking score should be near 1.0 for Arm B
    assert metrics["B_256"]["overthinking_score"] > 0.9
