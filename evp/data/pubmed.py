from __future__ import annotations

import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from evp.utils.logging_utils import get_logger


_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def fetch_pubmed_papers(
    query: str,
    max_results: int = 8,
    retries: int = 3,
    email: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch papers from PubMed (NCBI E-utilities) and return normalized dicts."""
    logger = get_logger("pubmed")

    if not query.strip():
        return []

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "xml",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    ids = _retry_request(_BASE + "esearch.fcgi", params, retries=retries, logger=logger)
    if not ids:
        return []

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
    }
    if email:
        fetch_params["email"] = email
    if api_key:
        fetch_params["api_key"] = api_key

    xml_text = _retry_request_raw(
        _BASE + "efetch.fcgi",
        fetch_params,
        retries=retries,
        logger=logger,
    )
    if not xml_text:
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("PubMed XML parse failed")
        return []

    papers: List[Dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue
        pmid = _text(medline.find("PMID"))
        art = medline.find("Article")
        if art is None:
            continue
        title = _text(art.find("ArticleTitle"))
        abstract = " ".join(
            [_text(a) for a in art.findall("Abstract/AbstractText") if _text(a)]
        ).strip()
        authors = []
        for author in art.findall("AuthorList/Author"):
            last = _text(author.find("LastName"))
            fore = _text(author.find("ForeName"))
            if last or fore:
                authors.append(" ".join([fore, last]).strip())
        pub_date = _text(art.find("Journal/JournalIssue/PubDate/Year"))

        if not (title or abstract):
            continue

        papers.append(
            {
                "paper_id": pmid or f"pubmed:{len(papers)}",
                "title": title or "Untitled",
                "abstract": abstract,
                "authors": authors,
                "published": pub_date or None,
                "updated": None,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                "categories": ["pubmed"],
                "source": "pubmed",
            }
        )

    return papers


def _retry_request(
    url: str,
    params: Dict[str, str],
    retries: int,
    logger,
) -> List[str]:
    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            xml_text = _request(url, params)
            root = ET.fromstring(xml_text)
            ids = [node.text for node in root.findall(".//Id") if node.text]
            return ids
        except Exception as exc:
            if attempt >= retries:
                logger.warning("PubMed fetch failed after %s attempts: %s", retries, exc)
                return []
            time.sleep(attempt)
    return []


def _retry_request_raw(
    url: str,
    params: Dict[str, str],
    retries: int,
    logger,
) -> str:
    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            return _request(url, params)
        except Exception as exc:
            if attempt >= retries:
                logger.warning("PubMed fetch failed after %s attempts: %s", retries, exc)
                return ""
            time.sleep(attempt)
    return ""


def _request(url: str, params: Dict[str, str]) -> str:
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{query}") as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _text(node: Optional[ET.Element]) -> str:
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())
