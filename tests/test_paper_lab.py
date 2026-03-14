from evp.lab.paper_audit import build_knowledge_bridge, deconstruct_paper, inspect_consistency


class EmptyLLM:
    def generate(self, prompt: str) -> str:
        return "{}"


def test_deconstruct_fallback_fields():
    text = """
    Abstract
    We report 92.5% accuracy.

    Methodology
    Transfer learning with a lightweight CNN.

    Results
    Accuracy: 90.0% and F1: 88.4

    Conclusion
    The approach improves baseline quality.
    """
    data = deconstruct_paper(text, llm_client=EmptyLLM())
    assert isinstance(data.abstract_summary, str)
    assert isinstance(data.methodology_description, str)
    assert isinstance(data.results_metrics, list)
    assert isinstance(data.conclusion, str)


def test_inspector_and_bridge():
    audit = inspect_consistency("Claimed 99% accuracy", [90.0], llm_client=EmptyLLM())
    assert audit.is_consistent is False
    bridge = build_knowledge_bridge(
        deconstruct_paper("Abstract\nAccuracy 90%\nResults\nAccuracy 89%", llm_client=EmptyLLM())
    )
    assert len(bridge.nodes) >= 2
