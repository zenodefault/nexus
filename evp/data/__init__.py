"""Data ingestion modules for EVP."""

from evp.data.arxiv import build_literature_digest, extract_abstracts, fetch_papers
from evp.data.pubmed import fetch_pubmed_papers
from evp.data.serper import fetch_serper_scholar

__all__ = ["fetch_papers", "extract_abstracts", "build_literature_digest"]
