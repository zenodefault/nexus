import asyncio

from evp.orchestration.pipeline import run_pipeline


def test_pipeline_mock_shape():
    result = asyncio.run(
        run_pipeline(topic="Brain MRI", goal="Improve classification accuracy")
    )
    assert "experiments" in result
    assert result["recommended_experiment_id"] is not None
    assert len(result["experiments"]) == 3
    exp = result["experiments"][0]
    for field in ["id", "compute_units", "novelty_score", "value"]:
        assert field in exp
