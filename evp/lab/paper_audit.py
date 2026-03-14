from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field

from evp.utils.llm import MockLLMClient, safe_json_loads


class PaperDeconstruction(BaseModel):
    abstract_summary: str = Field(default="")
    methodology_description: str = Field(default="")
    results_metrics: List[float] = Field(default_factory=list)
    conclusion: str = Field(default="")


class AuditReport(BaseModel):
    is_consistent: bool
    discrepancies: List[str]
    verdict: str


@dataclass
class BridgeResult:
    nodes: List[str]
    edges: List[Tuple[str, str, float]]
    graph: Any


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract clean text from PDF bytes using PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed. Please install 'PyMuPDF'.") from exc

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: List[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()

    merged = "\n".join(pages)
    return _clean_text(merged)


def deconstruct_paper(text: str, llm_client: Any | None = None) -> PaperDeconstruction:
    """Run extraction chain with schema target and heuristic fallback."""
    client = llm_client or MockLLMClient()
    prompt = _deconstruction_prompt(text)
    raw = client.generate(prompt)
    payload = safe_json_loads(raw)

    if payload:
        payload = _normalize_deconstruction_payload(payload)
        try:
            return PaperDeconstruction(**payload)
        except Exception:
            pass

    return _heuristic_deconstruction(text)


def inspect_consistency(
    abstract_summary: str,
    results_metrics: List[float],
    llm_client: Any | None = None,
) -> AuditReport:
    """Ghost inspector: compare claims against result metrics."""
    client = llm_client or MockLLMClient()
    prompt = _inspector_prompt(abstract_summary, results_metrics)
    raw = client.generate(prompt)
    payload = safe_json_loads(raw)

    if payload and {"is_consistent", "discrepancies", "verdict"}.issubset(payload.keys()):
        try:
            return AuditReport(**payload)
        except Exception:
            pass

    return _heuristic_inspection(abstract_summary, results_metrics)


def build_knowledge_bridge(deconstruction: PaperDeconstruction) -> BridgeResult:
    """Create concept graph using sentence embeddings + NetworkX."""
    try:
        import networkx as nx  # type: ignore
    except ImportError:
        nx = None

    concepts = _extract_concepts(deconstruction)
    graph = nx.Graph() if nx else {"nodes": [], "edges": []}
    for c in concepts:
        if nx:
            graph.add_node(c)
        else:
            graph["nodes"].append(c)

    embeddings = _embed_concepts(concepts)
    edges: List[Tuple[str, str, float]] = []

    if embeddings:
        for i, src in enumerate(concepts):
            for j in range(i + 1, len(concepts)):
                dst = concepts[j]
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                if sim >= 0.42:
                    if nx:
                        graph.add_edge(src, dst, weight=sim)
                    else:
                        graph["edges"].append((src, dst, sim))
                    edges.append((src, dst, sim))
    else:
        # Fallback: connect sequentially to guarantee a visible bridge.
        for i in range(len(concepts) - 1):
            src, dst = concepts[i], concepts[i + 1]
            if nx:
                graph.add_edge(src, dst, weight=0.3)
            else:
                graph["edges"].append((src, dst, 0.3))
            edges.append((src, dst, 0.3))

    return BridgeResult(nodes=concepts, edges=edges, graph=graph)


def _deconstruction_prompt(text: str) -> str:
    clipped = text[:12000]
    return (
        "You are a precision parser. Extract ONLY structured content from this paper text. "
        "Extract ONLY the metrics mentioned in the Abstract vs the Results section. "
        "Ignore general text when capturing metrics; look for numbers and percentages. "
        "Return strict JSON with keys: abstract_summary (string), methodology_description (string), "
        "results_metrics (array of numbers), conclusion (string).\n\n"
        f"Paper Text:\n{clipped}"
    )


def _inspector_prompt(abstract_summary: str, results_metrics: List[float]) -> str:
    return (
        "Compare the Claims in the Abstract against the Data in the Results. "
        "Flag any exaggeration, missing standard deviations, or discrepancies >5%. "
        "Return JSON with keys: is_consistent (bool), discrepancies (list), verdict (text).\n\n"
        f"Abstract Claims:\n{abstract_summary}\n\n"
        f"Results Metrics:\n{results_metrics}"
    )


def _heuristic_deconstruction(text: str) -> PaperDeconstruction:
    abstract = _extract_section(text, ["abstract"], ["introduction", "1."])
    methodology = _extract_section(text, ["method", "methodology"], ["results", "evaluation", "experiments"])
    results = _extract_section(text, ["results", "evaluation", "experiments"], ["discussion", "conclusion"])
    conclusion = _extract_section(text, ["conclusion", "discussion"], ["references", "acknowledgement"])

    metrics = _extract_metrics((abstract + "\n" + results).strip())

    return PaperDeconstruction(
        abstract_summary=_summarize_text(abstract, max_chars=500),
        methodology_description=_summarize_text(methodology, max_chars=700),
        results_metrics=metrics,
        conclusion=_summarize_text(conclusion, max_chars=500),
    )


def _heuristic_inspection(abstract_summary: str, results_metrics: List[float]) -> AuditReport:
    claim_metrics = _extract_metrics(abstract_summary)
    discrepancies: List[str] = []

    if not results_metrics:
        discrepancies.append("No results metrics extracted; cannot verify claims.")

    if claim_metrics and results_metrics:
        best_result = max(results_metrics)
        for c in claim_metrics:
            if c - best_result > 5:
                discrepancies.append(
                    f"Claimed metric {c:.2f} exceeds best reported result {best_result:.2f} by more than 5 points."
                )

    lowered = abstract_summary.lower()
    if ("significant" in lowered or "state-of-the-art" in lowered) and "std" not in lowered and "standard deviation" not in lowered:
        discrepancies.append("Strong claim language detected without explicit standard deviation in abstract text.")

    is_consistent = len(discrepancies) == 0
    verdict = (
        "Claims and results appear broadly consistent."
        if is_consistent
        else "Potential inconsistencies found between abstract claims and results metrics."
    )

    return AuditReport(is_consistent=is_consistent, discrepancies=discrepancies, verdict=verdict)


def _normalize_deconstruction_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    out.setdefault("abstract_summary", "")
    out.setdefault("methodology_description", "")
    out.setdefault("results_metrics", [])
    out.setdefault("conclusion", "")

    normalized_metrics: List[float] = []
    raw = out.get("results_metrics", [])
    if isinstance(raw, list):
        for item in raw:
            val = _to_float(item)
            if val is not None:
                normalized_metrics.append(val)
    elif isinstance(raw, str):
        normalized_metrics = _extract_metrics(raw)
    out["results_metrics"] = normalized_metrics
    return out


def _extract_section(text: str, starts: List[str], stops: List[str]) -> str:
    lower = text.lower()
    start_idx = None
    for s in starts:
        idx = lower.find(f"\n{s}")
        if idx == -1:
            idx = lower.find(s)
        if idx != -1:
            start_idx = idx
            break

    if start_idx is None:
        return ""

    end_idx = len(text)
    for e in stops:
        idx = lower.find(e, start_idx + 1)
        if idx != -1:
            end_idx = min(end_idx, idx)
    return text[start_idx:end_idx].strip()


def _extract_metrics(text: str) -> List[float]:
    if not text:
        return []

    values: List[float] = []
    # percentages and numeric metrics
    for m in re.findall(r"\b\d+(?:\.\d+)?\s*%", text):
        num = _to_float(m)
        if num is not None:
            values.append(num)

    for m in re.findall(r"\b(?:accuracy|f1|auc|bleu|rouge|precision|recall)\s*[:=]?\s*\d+(?:\.\d+)?", text, flags=re.I):
        num = _to_float(m)
        if num is not None:
            values.append(num)

    # general decimals in likely metric range
    for m in re.findall(r"\b\d+\.\d+\b", text):
        num = _to_float(m)
        if num is not None and 0 <= num <= 100:
            values.append(num)

    deduped = sorted({round(v, 3) for v in values})
    return deduped[:50]


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _summarize_text(text: str, max_chars: int) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _clean_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _extract_concepts(deconstruction: PaperDeconstruction) -> List[str]:
    joined = " ".join(
        [
            deconstruction.abstract_summary,
            deconstruction.methodology_description,
            deconstruction.conclusion,
        ]
    ).lower()

    tokens = re.findall(r"[a-z][a-z0-9\-]{2,}", joined)
    stop = {
        "with", "from", "that", "this", "were", "have", "using", "into", "than",
        "their", "there", "which", "model", "paper", "results", "method", "methods",
        "conclusion", "show", "shows", "data", "analysis", "based", "approach",
    }
    kept = [t for t in tokens if t not in stop]

    # preserve order and uniqueness
    uniq: List[str] = []
    for token in kept:
        if token not in uniq:
            uniq.append(token)
    if len(uniq) < 2:
        uniq.extend(["domain_a", "domain_b"])
    return uniq[:16]


def _embed_concepts(concepts: List[str]) -> List[List[float]]:
    if not concepts:
        return []
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return []

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    vectors = model.encode(concepts, normalize_embeddings=True)
    return [list(map(float, v)) for v in vectors]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def audit_to_json(report: AuditReport) -> str:
    return json.dumps(report.model_dump(), indent=2)
