from typing import Dict, List


RESOURCE_COSTS = {"Low": 1, "Medium": 2, "High": 3}


def resource_cost_from_units(units: str) -> int:
    return RESOURCE_COSTS.get(units, 3)


def score_experiments(experiments: List[Dict]) -> Dict:
    scored = []
    for exp in experiments:
        impact_score = exp.get("novelty_score", 0)
        cost = resource_cost_from_units(exp.get("compute_units", "High"))
        value = (impact_score / cost) if cost else 0
        scored.append({**exp, "impact_score": impact_score, "resource_cost": cost, "value": value})

    ranked = sorted(scored, key=lambda e: e["value"], reverse=True)
    recommended_id = ranked[0]["id"] if ranked else None
    return {"experiments": ranked, "recommended_experiment_id": recommended_id}
