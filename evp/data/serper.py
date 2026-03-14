from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

from evp.utils.logging_utils import get_logger


_SERPER_URL = "https://google.serper.dev/scholar"


def fetch_serper_scholar(
    query: str,
    max_results: int = 8,
    retries: int = 3,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch Google Scholar results via Serper API and return normalized dicts."""
    logger = get_logger("serper")

    if not query.strip():
        return []

    key = api_key or os.getenv("SERPER_API_KEY")
    if not key:
        logger.info("SERPER_API_KEY not set; skipping Scholar fetch.")
        return []

    payload = {"q": query, "num": max_results}

    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            req = urllib.request.Request(
                _SERPER_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            return _normalize_results(data)
        except Exception as exc:
            if attempt >= retries:
                logger.warning("Serper Scholar fetch failed after %s attempts: %s", retries, exc)
                return []
            time.sleep(attempt)

    return []


def _normalize_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in payload.get("organic", []) or []:
        title = str(item.get("title", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        link = str(item.get("link", "")).strip() or None
        if not (title or snippet):
            continue
        out.append(
            {
                "paper_id": link or title,
                "title": title or "Untitled",
                "abstract": snippet,
                "authors": [],
                "published": item.get("year"),
                "updated": None,
                "url": link,
                "categories": ["scholar"],
                "source": "serper_scholar",
            }
        )
    return out
