"""
BM25 retrieval utilities for ConstraintSuite.

This module provides functions for:
- Loading/building BM25 indexes via Pyserini
- Retrieving candidate document pools
- Batch retrieval for efficiency

Primary tool: Pyserini with prebuilt MS MARCO index.
"""

from pathlib import Path
from typing import Any


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
        # TODO: Implementation
        # Hint: from pyserini.search.lucene import LuceneSearcher
        # self.searcher = LuceneSearcher.from_prebuilt_index(index_name)
        raise NotImplementedError("BM25Retriever.__init__ not yet implemented")

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
        # TODO: Implementation
        raise NotImplementedError("BM25Retriever.retrieve not yet implemented")

    def retrieve_batch(
        self,
        queries: list[tuple[str, str]],
        k: int = 200,
        threads: int = 4
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
        # TODO: Implementation
        raise NotImplementedError("BM25Retriever.retrieve_batch not yet implemented")

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
        # TODO: Implementation
        raise NotImplementedError("BM25Retriever.get_document not yet implemented")


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
    # TODO: Implementation
    raise NotImplementedError("build_custom_index not yet implemented")
