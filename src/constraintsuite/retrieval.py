"""
BM25 retrieval utilities for ConstraintSuite.

This module provides functions for:
- Loading/building BM25 indexes via Pyserini
- Retrieving candidate document pools
- Batch retrieval for efficiency

Primary tool: Pyserini with prebuilt MS MARCO index.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pyserini.search.lucene import LuceneSearcher

logger = logging.getLogger("constraintsuite")


class BM25Retriever:
    """
    BM25 retriever using Pyserini.

    This class wraps Pyserini's LuceneSearcher for convenient
    BM25 retrieval from prebuilt or custom indexes.

    Attributes:
        index_name: Name of the Pyserini prebuilt index.
        searcher: Underlying LuceneSearcher instance.

    Example:
        >>> retriever = BM25Retriever("msmarco-v1-passage")
        >>> candidates = retriever.retrieve("python web scraping", k=200)
        >>> print(len(candidates))
        200
    """

    def __init__(
        self,
        index_name: str = "msmarco-v1-passage",
        k1: float = 0.9,
        b: float = 0.4
    ):
        """
        Initialize BM25 retriever.

        Args:
            index_name: Pyserini prebuilt index name.
                Available indexes: https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
            k1: BM25 k1 parameter (term frequency saturation).
            b: BM25 b parameter (document length normalization).

        Example:
            >>> retriever = BM25Retriever("msmarco-v1-passage")

        Note:
            First-time initialization will download the index (~2GB for MS MARCO).
        """
        self.index_name = index_name
        self.k1 = k1
        self.b = b

        logger.info(f"Loading BM25 index: {index_name}")

        # Load prebuilt or custom index
        if Path(index_name).exists():
            # Custom index path
            self.searcher = LuceneSearcher(index_name)
        else:
            # Prebuilt index
            self.searcher = LuceneSearcher.from_prebuilt_index(index_name)

        # Set BM25 parameters
        self.searcher.set_bm25(k1, b)

        logger.info(f"BM25 retriever initialized with k1={k1}, b={b}")

    def retrieve(
        self,
        query: str,
        k: int = 200
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Query text.
            k: Number of documents to retrieve.

        Returns:
            List of candidate documents, each with:
            - doc_id: str
            - text: str
            - title: str | None
            - bm25_rank: int (0-indexed)
            - bm25_score: float

        Example:
            >>> candidates = retriever.retrieve("python without selenium", k=200)
            >>> print(candidates[0]["doc_id"])
            'msmarco:1234567'
        """
        hits = self.searcher.search(query, k=k)

        candidates = []
        for rank, hit in enumerate(hits):
            doc = self._parse_document(hit)
            doc["bm25_rank"] = rank
            doc["bm25_score"] = hit.score
            candidates.append(doc)

        return candidates

    def _parse_document(self, hit) -> dict[str, Any]:
        """Parse a Pyserini hit into a document dict."""
        doc_id = hit.docid

        # Try to get the raw document content
        raw = hit.lucene_document.get("raw")
        if raw:
            try:
                parsed = json.loads(raw)
                return {
                    "doc_id": doc_id,
                    "text": parsed.get("contents", parsed.get("text", "")),
                    "title": parsed.get("title"),
                }
            except json.JSONDecodeError:
                # Raw field is plain text
                return {
                    "doc_id": doc_id,
                    "text": raw,
                    "title": None,
                }

        # Fall back to contents field
        contents = hit.lucene_document.get("contents")
        if contents:
            return {
                "doc_id": doc_id,
                "text": contents,
                "title": None,
            }

        # Last resort: fetch document directly
        fetched = self.searcher.doc(doc_id)
        if fetched:
            raw = fetched.raw()
            try:
                parsed = json.loads(raw)
                return {
                    "doc_id": doc_id,
                    "text": parsed.get("contents", parsed.get("text", "")),
                    "title": parsed.get("title"),
                }
            except json.JSONDecodeError:
                return {
                    "doc_id": doc_id,
                    "text": raw,
                    "title": None,
                }

        return {
            "doc_id": doc_id,
            "text": "",
            "title": None,
        }

    def retrieve_batch(
        self,
        queries: list[tuple[str, str]],
        k: int = 200,
        threads: int = 8
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Batch retrieve for multiple queries.

        Args:
            queries: List of (query_id, query_text) tuples.
            k: Number of documents per query.
            threads: Number of parallel threads.

        Returns:
            Dictionary mapping query_id -> list of candidates.

        Example:
            >>> queries = [("q1", "python scraping"), ("q2", "web automation")]
            >>> results = retriever.retrieve_batch(queries, k=200)
            >>> print(len(results["q1"]))
            200
        """
        results = {}

        # Use batch search for efficiency
        query_ids = [qid for qid, _ in queries]
        query_texts = [qtext for _, qtext in queries]

        # Pyserini's batch search
        batch_hits = self.searcher.batch_search(
            queries=query_texts,
            qids=query_ids,
            k=k,
            threads=threads
        )

        # Parse results
        for qid in query_ids:
            hits = batch_hits.get(qid, [])
            candidates = []
            for rank, hit in enumerate(hits):
                doc = self._parse_document(hit)
                doc["bm25_rank"] = rank
                doc["bm25_score"] = hit.score
                candidates.append(doc)
            results[qid] = candidates

        logger.info(f"Retrieved candidates for {len(results)} queries")
        return results

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single document by ID.

        Args:
            doc_id: Document identifier.

        Returns:
            Document dict with text and title, or None if not found.

        Example:
            >>> doc = retriever.get_document("msmarco:1234567")
            >>> print(doc["text"][:100])
        """
        doc = self.searcher.doc(doc_id)
        if doc is None:
            return None

        raw = doc.raw()
        try:
            parsed = json.loads(raw)
            return {
                "doc_id": doc_id,
                "text": parsed.get("contents", parsed.get("text", "")),
                "title": parsed.get("title"),
            }
        except json.JSONDecodeError:
            return {
                "doc_id": doc_id,
                "text": raw,
                "title": None,
            }

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return self.searcher.num_docs


def build_custom_index(
    corpus_path: str | Path,
    index_path: str | Path,
    threads: int = 4
) -> None:
    """
    Build a custom BM25 index from a corpus file.

    Args:
        corpus_path: Path to corpus JSONL file.
            Each line: {"id": str, "contents": str}
        index_path: Path to output index directory.
        threads: Number of indexing threads.

    Example:
        >>> build_custom_index("data/raw/corpus.jsonl", "data/indexes/custom")

    Note:
        For MS MARCO, prefer using prebuilt indexes.
        Custom indexing is useful for BEIR domains or custom corpora.
    """
    import subprocess
    import shutil

    corpus_path = Path(corpus_path)
    index_path = Path(index_path)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    # Create output directory
    index_path.mkdir(parents=True, exist_ok=True)

    # Build index using Pyserini CLI
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(corpus_path.parent),
        "--index", str(index_path),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]

    logger.info(f"Building index at {index_path}")
    subprocess.run(cmd, check=True)
    logger.info("Index build complete")


def verify_index(index_name: str = "msmarco-v1-passage") -> bool:
    """
    Verify that an index is accessible and working.

    Args:
        index_name: Pyserini prebuilt index name or path.

    Returns:
        True if index is working, False otherwise.
    """
    try:
        retriever = BM25Retriever(index_name)
        # Test query
        results = retriever.retrieve("test query", k=1)
        logger.info(f"Index verification passed: {len(retriever)} documents")
        return True
    except Exception as e:
        logger.error(f"Index verification failed: {e}")
        return False
