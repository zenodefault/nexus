from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Dict, List

import networkx as nx
from sentence_transformers import SentenceTransformer, util

from evp.data.arxiv import fetch_papers
from evp.data.local import load_local_papers
from evp.data.pubmed import fetch_pubmed_papers
from evp.data.serper import fetch_serper_scholar
from evp.syntropy.state import GraphState

try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:  # pragma: no cover - fallback for older langchain installs
    from pydantic import BaseModel, Field


_EMBEDDER: SentenceTransformer | None = None
_LLM = None


class PaperMethodResults(BaseModel):
    methodology: List[str] = Field(
        description="Core methods, algorithms, or procedures (short phrases)"
    )
    results: List[str] = Field(description="Key results or outcomes (short phrases)")


def _with_trace(state: GraphState, agent: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    trace = list(state.get("trace", []))
    trace.append({"agent": agent, "payload": payload})
    return {**payload, "trace": trace}


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        model_name = os.getenv("SYNTROPY_EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


def _get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM

    mode = os.getenv("SYNTROPY_LLM_MODE", "openai").lower()
    if mode == "mock":
        _LLM = None
        return _LLM

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        _LLM = None
        return _LLM

    if not os.getenv("OPENAI_API_KEY"):
        _LLM = None
        return _LLM

    model_name = os.getenv("SYNTROPY_MODEL", "gpt-4o")
    _LLM = ChatOpenAI(model=model_name, temperature=0)
    return _LLM


def _extract_concepts_heuristic(text: str, max_concepts: int = 10) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "using",
        "used",
        "use",
        "based",
        "via",
        "while",
        "study",
        "paper",
        "results",
        "method",
        "methods",
        "approach",
        "approaches",
        "data",
        "model",
        "models",
        "learning",
        "analysis",
        "system",
        "systems",
        "algorithm",
        "algorithms",
        "framework",
        "frameworks",
        "design",
        "task",
        "tasks",
        "performance",
        "study",
        "experiments",
        "experiment",
        "results",
        "proposed",
        "novel",
    }
    candidates = [t for t in tokens if t not in stopwords]
    if not candidates:
        return []

    counts = Counter(candidates)
    concepts = []
    for term, _ in counts.most_common(max_concepts):
        concepts.append(term.replace("-", " "))
    return concepts


def _extract_method_results_heuristic(text: str, max_items: int = 6) -> tuple[List[str], List[str]]:
    if not text:
        return [], []

    sentences = re.split(r"(?<=[.!?])\\s+", text)
    method_markers = (
        "method",
        "approach",
        "we propose",
        "we present",
        "we use",
        "we train",
        "architecture",
        "algorithm",
        "model",
        "framework",
    )
    result_markers = (
        "result",
        "improve",
        "increase",
        "outperform",
        "accuracy",
        "achieve",
        "%",
        "reduce",
        "reduction",
        "performance",
    )

    methods = [s.strip() for s in sentences if any(k in s.lower() for k in method_markers)]
    results = [s.strip() for s in sentences if any(k in s.lower() for k in result_markers)]

    if not methods:
        methods = _extract_concepts_heuristic(text, max_concepts=max_items)
    if not results:
        results = [s.strip() for s in sentences if any(ch.isdigit() for ch in s)]

    return methods[:max_items], results[:max_items]


def _extract_text(papers: List[Dict[str, Any]]) -> str:
    parts = []
    for paper in papers:
        title = str(paper.get("title", "")).strip()
        summary = str(paper.get("summary") or paper.get("abstract") or "").strip()
        if summary:
            if title:
                parts.append(f"{title}: {summary}")
            else:
                parts.append(summary)
    return "\n".join(parts)


def _dedupe_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for paper in papers:
        key = (paper.get("title") or "").strip().lower()
        if not key:
            key = (paper.get("url") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(paper)
    return out


def archivist_agent(state: GraphState) -> Dict[str, Any]:
    topic_a = state.get("topic_a", "").strip()
    topic_b = state.get("topic_b", "").strip()
    max_results = int(state.get("max_results", 3))
    use_local = bool(state.get("use_local_papers", False))

    papers_a = state.get("papers_a")
    papers_b = state.get("papers_b")
    if papers_a and papers_b:
        payload = {"papers_a": papers_a, "papers_b": papers_b, "source": "upload"}
        return _with_trace(state, "Archivist", payload)

    local_dir = os.getenv("SYNTROPY_LOCAL_PAPERS_DIR") or os.getenv(
        "EVP_LOCAL_PAPERS_DIR", "data/papers"
    )
    if use_local:
        local_papers = load_local_papers(local_dir)
        if local_papers:
            payload = {
                "papers_a": local_papers,
                "papers_b": local_papers,
                "source": f"local:{local_dir}",
            }
            return _with_trace(state, "Archivist", payload)

    use_pubmed = os.getenv("SYNTROPY_USE_PUBMED", "true").lower() == "true"
    use_scholar = os.getenv("SYNTROPY_USE_SCHOLAR", "false").lower() == "true"
    pubmed_email = os.getenv("SYNTROPY_PUBMED_EMAIL") or os.getenv("NCBI_EMAIL")
    pubmed_key = os.getenv("SYNTROPY_PUBMED_API_KEY") or os.getenv("NCBI_API_KEY")

    papers_a = []
    papers_b = []
    if topic_a:
        papers_a.extend(fetch_papers(topic_a, max_results=max_results))
        if use_pubmed:
            papers_a.extend(
                fetch_pubmed_papers(
                    topic_a, max_results=max_results, email=pubmed_email, api_key=pubmed_key
                )
            )
        if use_scholar:
            papers_a.extend(fetch_serper_scholar(topic_a, max_results=max_results))
    if topic_b:
        papers_b.extend(fetch_papers(topic_b, max_results=max_results))
        if use_pubmed:
            papers_b.extend(
                fetch_pubmed_papers(
                    topic_b, max_results=max_results, email=pubmed_email, api_key=pubmed_key
                )
            )
        if use_scholar:
            papers_b.extend(fetch_serper_scholar(topic_b, max_results=max_results))

    papers_a = _dedupe_papers(papers_a)
    papers_b = _dedupe_papers(papers_b)
    payload = {"papers_a": papers_a, "papers_b": papers_b, "source": "mixed"}
    return _with_trace(state, "Archivist", payload)


def deconstructor_agent(state: GraphState) -> Dict[str, Any]:
    text_a = _extract_text(state.get("papers_a", []))
    text_b = _extract_text(state.get("papers_b", []))

    llm = _get_llm()
    if llm is None:
        methods_a, results_a = _extract_method_results_heuristic(text_a)
        methods_b, results_b = _extract_method_results_heuristic(text_b)
        payload = {
            "concepts_a": methods_a,
            "concepts_b": methods_b,
            "methods_a": methods_a,
            "methods_b": methods_b,
            "results_a": results_a,
            "results_b": results_b,
            "note": "heuristic-method-results",
        }
        return _with_trace(state, "Deconstructor", payload)

    structured_llm = llm.with_structured_output(PaperMethodResults)

    res_a = structured_llm.invoke(
        "Extract only the methodology and results from this text. "
        "Return short phrases, ignore background or fluff.\n"
        + text_a
    )
    res_b = structured_llm.invoke(
        "Extract only the methodology and results from this text. "
        "Return short phrases, ignore background or fluff.\n"
        + text_b
    )

    methods_a = [c.strip() for c in res_a.methodology if c.strip()]
    results_a = [c.strip() for c in res_a.results if c.strip()]
    methods_b = [c.strip() for c in res_b.methodology if c.strip()]
    results_b = [c.strip() for c in res_b.results if c.strip()]

    payload = {
        "concepts_a": methods_a,
        "concepts_b": methods_b,
        "methods_a": methods_a,
        "methods_b": methods_b,
        "results_a": results_a,
        "results_b": results_b,
        "note": "llm-method-results",
    }
    return _with_trace(state, "Deconstructor", payload)


def connector_agent(state: GraphState) -> Dict[str, Any]:
    nodes_a = {f"A_{i}": c for i, c in enumerate(state.get("concepts_a", []))}
    nodes_b = {f"B_{i}": c for i, c in enumerate(state.get("concepts_b", []))}

    all_nodes = {**nodes_a, **nodes_b}
    all_labels = list(all_nodes.values())
    all_ids = list(all_nodes.keys())

    if not all_nodes:
        payload = {"connection_path": ["No concepts available to connect."]}
        return _with_trace(state, "Connector", payload)

    embeddings = _get_embedder().encode(all_labels)

    graph = nx.Graph()
    graph.add_nodes_from(all_ids)

    threshold = float(state.get("similarity_threshold", 0.5))
    cos_scores = util.cos_sim(embeddings, embeddings)

    for i in range(len(all_ids)):
        for j in range(i + 1, len(all_ids)):
            if float(cos_scores[i][j]) > threshold:
                graph.add_edge(all_ids[i], all_ids[j])

    path = []
    for id_a in nodes_a.keys():
        for id_b in nodes_b.keys():
            try:
                short_path = nx.shortest_path(graph, source=id_a, target=id_b)
                path = [all_nodes[n] for n in short_path]
                break
            except nx.NetworkXNoPath:
                continue
        if path:
            break

    if not path:
        path = ["No strong connection found (graph disconnected)"]

    payload = {
        "connection_path": path,
        "graph_summary": {
            "nodes": len(all_ids),
            "edges": graph.number_of_edges(),
            "threshold": threshold,
        },
    }
    return _with_trace(state, "Connector", payload)


def grant_writer_agent(state: GraphState) -> Dict[str, Any]:
    path_str = " -> ".join(state.get("connection_path", []))
    topic_a = state.get("topic_a", "")
    topic_b = state.get("topic_b", "")

    llm = _get_llm()
    if llm is None:
        report = (
            "Title: Syntropy Bridge Proposal\n\n"
            f"Executive Summary: Explore how techniques from {topic_a} can inform "
            f"solutions for {topic_b}.\n\n"
            "The Novel Connection: The system found a bridge via\n"
            f"{path_str}.\n\n"
            "Proposed Methodology: Transfer the identified methods across domains, "
            "prototype a cross-domain pilot, and evaluate gains against baseline "
            "approaches."
        )
        payload = {"final_report": report, "note": "heuristic"}
        return _with_trace(state, "GrantWriter", payload)

    prompt = (
        "You are an expert research scientist.\n"
        f"Topic A: {topic_a}\n"
        f"Topic B: {topic_b}\n\n"
        "Our system identified a conceptual bridge between these fields:\n"
        f"BRIDGE: {path_str}\n\n"
        "Write a compelling 'Future Opportunity' research report.\n\n"
        "Structure:\n"
        "1. Title (Catchy)\n"
        "2. Executive Summary (The Why)\n"
        "3. The Novel Connection (Explain the bridge above)\n"
        "4. Proposed Methodology (How to apply A's solution to B's problem)\n"
    )

    response = llm.invoke(prompt)
    payload = {"final_report": response.content, "note": "llm"}
    return _with_trace(state, "GrantWriter", payload)
