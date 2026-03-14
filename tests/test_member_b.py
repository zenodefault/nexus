import asyncio

from evp.data.arxiv import build_literature_digest, extract_abstracts
from evp.orchestration.pipeline import run_pipeline
from evp.utils.resource_heuristics import estimate_resource_for_experiment


def test_extract_abstracts_and_digest():
    papers = [
        {
            "title": "Paper A",
            "abstract": "This is an abstract about contrastive learning for MRI.",
            "authors": ["Alice", "Bob"],
        },
        {
            "title": "Paper B",
            "abstract": "Another abstract about efficient networks.",
            "authors": ["Carol"],
        },
    ]

    abstracts = extract_abstracts(papers, max_chars=40)
    assert len(abstracts) == 2
    assert abstracts[0].startswith("Paper A:")

    digest = build_literature_digest(papers, max_papers=1)
    assert "Recent papers:" in digest
    assert "Paper A" in digest
    assert "Authors:" in digest


def test_resource_heuristic_mapping():
    high = estimate_resource_for_experiment({"model": "Vision Transformer pre-training"})
    med = estimate_resource_for_experiment({"title": "CNN with contrastive learning"})
    low = estimate_resource_for_experiment({"title": "Logistic regression baseline"})

    assert high["compute_units"] == "High"
    assert med["compute_units"] == "Medium"
    assert low["compute_units"] == "Low"


def test_pipeline_static_mock_mode(monkeypatch):
    monkeypatch.setenv("EVP_LLM_MODE", "mock_static")
    result = asyncio.run(run_pipeline(topic="Brain MRI", goal="Improve classification accuracy"))

    assert result["recommended_experiment_id"] is not None
    assert len(result["experiments"]) == 3
    assert result["experiments"][0]["value"] >= result["experiments"][1]["value"]
    assert "papers" in result
