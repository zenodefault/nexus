from typing import Any, Dict, List, TypedDict


class AgentResult(TypedDict, total=False):
    summary: str
    key_findings: List[str]
    limitations: List[str]
    hypotheses: List[Dict[str, Any]]
    compute_units: str
    resource_rationale: str
    novelty_score: int
    expected_gain: float
    impact_rationale: str
