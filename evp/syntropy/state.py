from typing import Any, Dict, List
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    topic_a: str
    topic_b: str
    papers_a: List[Dict[str, Any]]
    papers_b: List[Dict[str, Any]]
    concepts_a: List[str]
    concepts_b: List[str]
    connection_path: List[str]
    final_report: str
    trace: List[Dict[str, Any]]
    use_local_papers: bool
    max_results: int
    similarity_threshold: float
