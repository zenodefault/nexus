"""Flat JSON schema definitions for agent outputs."""

AGENT_SCHEMAS = {
    "LiteratureAgent": ["summary", "key_findings", "limitations", "hypotheses"],
    "HypothesisAgent": ["summary", "key_findings", "limitations", "hypotheses"],
    "ResourceEstimatorAgent": ["compute_units", "resource_rationale"],
    "ImpactPredictorAgent": ["novelty_score", "expected_gain", "impact_rationale"],
}
