from evp.scoring.scoring import resource_cost_from_units, score_experiments


def test_resource_cost_mapping():
    assert resource_cost_from_units("Low") == 1
    assert resource_cost_from_units("Medium") == 2
    assert resource_cost_from_units("High") == 3
    assert resource_cost_from_units("Unknown") == 3


def test_score_experiments_ranking():
    experiments = [
        {"id": "a", "compute_units": "High", "novelty_score": 9},
        {"id": "b", "compute_units": "Low", "novelty_score": 6},
    ]
    result = score_experiments(experiments)
    assert result["recommended_experiment_id"] == "b"
    assert result["experiments"][0]["id"] == "b"
