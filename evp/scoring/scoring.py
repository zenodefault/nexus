import re
from typing import Dict, List


RESOURCE_COSTS = {"Low": 1, "Medium": 2, "High": 3}


def _to_float(value, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return default
    return default


def resource_cost_from_units(units: str) -> int:
    return RESOURCE_COSTS.get(units, 3)


def score_experiments(experiments: List[Dict]) -> Dict:
    scored = []
    for exp in experiments:
        impact_score = _to_float(exp.get("novelty_score", 0), 0.0)
        cost = resource_cost_from_units(str(exp.get("compute_units", "High")))
        value = (impact_score / cost) if cost else 0
        scored.append({**exp, "impact_score": impact_score, "resource_cost": cost, "value": value})

    ranked = sorted(scored, key=lambda e: e["value"], reverse=True)
    recommended_id = ranked[0]["id"] if ranked else None
    return {"experiments": ranked, "recommended_experiment_id": recommended_id}
