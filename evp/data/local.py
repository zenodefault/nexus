from __future__ import annotations

import os
from typing import Any, Dict, List


def load_local_papers(papers_dir: str, max_chars: int = 4000) -> List[Dict[str, Any]]:
    """Load local paper summaries from .txt/.md files."""
    if not os.path.isdir(papers_dir):
        return []

    entries: List[Dict[str, Any]] = []
    candidates = []
    for name in os.listdir(papers_dir):
        if name.lower().endswith((".txt", ".md")):
            candidates.append(name)
    for name in sorted(candidates):
        path = os.path.join(papers_dir, name)
        paper = _parse_paper_file(path, max_chars=max_chars)
        if paper:
            entries.append(paper)
    return entries


def _parse_paper_file(path: str, max_chars: int) -> Dict[str, Any] | None:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return None

    lines = raw.splitlines()
    title = _title_from_filename(path)
    authors: List[str] = []
    body_start = 0

    if lines and lines[0].lower().startswith("title:"):
        title = lines[0].split(":", 1)[1].strip() or title
        body_start = 1
        if len(lines) > 1 and lines[1].lower().startswith("authors:"):
            authors_line = lines[1].split(":", 1)[1]
            authors = [a.strip() for a in authors_line.split(",") if a.strip()]
            body_start = 2

    abstract = "\n".join(lines[body_start:]).strip()
    if not abstract:
        return None

    return {
        "paper_id": os.path.splitext(os.path.basename(path))[0],
        "title": title,
        "abstract": abstract[:max_chars],
        "authors": authors,
        "published": None,
        "updated": None,
        "url": None,
        "categories": ["local"],
        "source": "local",
        "path": path,
    }


def _title_from_filename(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace("_", " ").replace("-", " ").strip() or "Untitled"
