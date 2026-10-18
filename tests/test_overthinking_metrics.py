import pytest
from eval.robust_eval import compute_overthinking_metrics

def test_compute_overthinking_metrics_empty():
    res = compute_overthinking_metrics([], arm="A", budget=256)
    assert res["n"] == 0
    assert res["accuracy"] == 0.0
    assert res["auc_oaa"] == 0.0

def test_compute_overthinking_metrics_synthetic():
    # 4 synthetic records: 2 correct (one fast with 50 tokens, one slow with 500 tokens), 2 incorrect
    records = [
        {"arm": "A", "budget": 512, "is_correct": True,  "tokens_emitted": 50},
        {"arm": "A", "budget": 512, "is_correct": True,  "tokens_emitted": 500},
        {"arm": "A", "budget": 512, "is_correct": False, "tokens_emitted": 120},
        {"arm": "A", "budget": 512, "is_correct": False, "tokens_emitted": 300},
    ]
    res = compute_overthinking_metrics(records, arm="A", budget=512, thresholds=[64, 128, 256, 512, 1024], target_budget=256.0)
    
    assert res["n"] == 4
    assert res["accuracy"] == 0.5  # 2/4
    # Mean tokens: (50 + 500 + 120 + 300) / 4 = 970 / 4 = 242.5
    assert res["mean_tokens"] == 242.5
    
    # OAA at 64 tokens: only 1 problem is correct AND <= 64 tokens -> 1/4 = 0.25
    assert res["oaa_by_threshold"]["OAA_64"] == 0.25
    # OAA at 512 tokens: both correct problems have tokens <= 512 (50 and 500) -> 2/4 = 0.5
    assert res["oaa_by_threshold"]["OAA_512"] == 0.5
    
    # AUC should be positive and bounded in [0, 1]
    assert 0.0 < res["auc_oaa"] <= 0.5
    
    # Token efficiency: min(1.0, 256 / 242.5) = 1.0
    assert res["token_efficiency"] == 1.0
    
    # Harmonic Overthinking score: 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 0.6667
    assert round(res["overthinking_score"], 3) == 0.667

def test_overthinking_penalty_on_verbose_model():
    # Model that generates 1500 tokens for every correct solve
    verbose_records = [
        {"arm": "CoT", "budget": 2048, "is_correct": True, "tokens_emitted": 1500},
        {"arm": "CoT", "budget": 2048, "is_correct": True, "tokens_emitted": 1600},
    ]
    res_verbose = compute_overthinking_metrics(verbose_records, arm="CoT", budget=2048, target_budget=256.0)
    
    # Model that generates 10 tokens for every correct solve (Reflex)
    reflex_records = [
        {"arm": "Reflex", "budget": 2048, "is_correct": True, "tokens_emitted": 8},
        {"arm": "Reflex", "budget": 2048, "is_correct": True, "tokens_emitted": 12},
    ]
    res_reflex = compute_overthinking_metrics(reflex_records, arm="Reflex", budget=2048, target_budget=256.0)
    
    # Both have 100% accuracy, but reflex must have dramatically higher Overthinking Score
    assert res_verbose["accuracy"] == 1.0
    assert res_reflex["accuracy"] == 1.0
    assert res_reflex["overthinking_score"] > res_verbose["overthinking_score"]
    assert res_reflex["auc_oaa"] > res_verbose["auc_oaa"]

def test_budget_choke_factor():
    from eval.robust_eval import compute_budget_choke_factor
    # Complete choke: acc drops from 30% at 2048 to 0% at 256
    assert compute_budget_choke_factor(0.0, 0.30) == 1.0
    # No degradation: acc is 25% at 2048 and 25% at 256
    assert compute_budget_choke_factor(0.25, 0.25) == 0.0
    # Inverted choke (model does BETTER with shorter budget, e.g. doesn't overthink)
    assert compute_budget_choke_factor(0.33, 0.23) < 0.0

