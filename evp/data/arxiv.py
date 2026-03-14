import time
from typing import Any, Dict, List

from evp.utils.logging_utils import get_logger


def fetch_papers(query: str, max_results: int = 8, retries: int = 3) -> List[Dict[str, Any]]:
    """Fetch papers from arXiv and return normalized dicts."""
    logger = get_logger("arxiv")

    try:
        import arxiv
    except ImportError as exc:
        raise RuntimeError("arxiv package is not installed. Install with: pip install arxiv") from exc

    if not query.strip():
        return []

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=1.0, num_retries=2)

    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            papers: List[Dict[str, Any]] = []
            for result in client.results(search):
                papers.append(
                    {
                        "paper_id": result.get_short_id(),
                        "title": _clean_text(result.title),
                        "abstract": _clean_text(result.summary),
                        "authors": [a.name for a in result.authors],
                        "published": result.published.isoformat() if result.published else None,
                        "updated": result.updated.isoformat() if result.updated else None,
                        "url": result.entry_id,
                        "categories": list(result.categories or []),
                    }
                )
            return papers
        except Exception as exc:  # arxiv lib raises generic transport errors
            msg = str(exc).lower()
            rate_limited = "429" in msg or "rate" in msg
            if attempt >= retries:
                logger.warning("arXiv fetch failed after %s attempts: %s", retries, exc)
                return []

            sleep_seconds = (2**attempt) if rate_limited else attempt
            logger.warning(
                "arXiv fetch attempt %s/%s failed (%s). Retrying in %ss.",
                attempt,
                retries,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    return []


def extract_abstracts(papers: List[Dict[str, Any]], max_chars: int = 1000) -> List[str]:
    """Extract bounded abstracts to keep prompt context small."""
    out: List[str] = []
    for paper in papers:
        title = str(paper.get("title", "Untitled")).strip()
        abstract = _clean_text(str(paper.get("abstract", "")))
        if not abstract:
            continue
        out.append(f"{title}: {abstract[:max_chars]}")
    return out


def build_literature_digest(
    papers: List[Dict[str, Any]],
    max_papers: int = 5,
    max_abstract_chars: int = 450,
) -> str:
    """Format papers into compact context for agents."""
    if not papers:
        return "No paper context available."

    lines = ["Recent papers:"]
    for idx, paper in enumerate(papers[:max_papers], start=1):
        title = str(paper.get("title", "Untitled")).strip()
        authors = ", ".join(paper.get("authors", [])[:3])
        if len(paper.get("authors", [])) > 3:
            authors += ", et al."
        abstract = _clean_text(str(paper.get("abstract", "")))[:max_abstract_chars]
        lines.append(f"{idx}. {title}")
        lines.append(f"   Authors: {authors or 'Unknown'}")
        lines.append(f"   Abstract: {abstract or 'N/A'}")
    return "\n".join(lines)


def _clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())
