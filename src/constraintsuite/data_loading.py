"""
Data loading utilities for ConstraintSuite.

This module provides functions for loading IR corpora and queries from:
- MS MARCO (via Pyserini or ir_datasets)
- BEIR datasets

Primary data source: MS MARCO passages with prebuilt Pyserini index.
"""

from pathlib import Path
from typing import Iterator


def load_msmarco_queries(
    split: str = "train",
    limit: int | None = None
) -> dict[str, str]:
    """
    Load MS MARCO queries.

    Args:
        split: Dataset split ("train", "dev", "eval").
        limit: Optional limit on number of queries to load.

    Returns:
        Dictionary mapping query_id -> query_text.

    Example:
        >>> queries = load_msmarco_queries("train", limit=1000)
        >>> print(queries["123456"])
        'what is python web scraping'

    Note:
        Uses ir_datasets for standardized access:
        https://ir-datasets.com/msmarco-passage.html
    """
    # TODO: Implementation
    # Hint: Use ir_datasets.load("msmarco-passage/train")
    raise NotImplementedError("load_msmarco_queries not yet implemented")


def load_msmarco_corpus(
    limit: int | None = None
) -> dict[str, dict]:
    """
    Load MS MARCO passage corpus.

    Args:
        limit: Optional limit on number of passages to load.

    Returns:
        Dictionary mapping doc_id -> {"text": str, "title": str | None}.

    Example:
        >>> corpus = load_msmarco_corpus(limit=10000)
        >>> print(corpus["1234567"]["text"][:100])
        'This is a passage about...'

    Note:
        For large-scale use, prefer using Pyserini's index directly
        via the retrieval module.
    """
    # TODO: Implementation
    raise NotImplementedError("load_msmarco_corpus not yet implemented")


def load_beir_dataset(
    dataset_name: str,
    split: str = "test"
) -> tuple[dict, dict, dict]:
    """
    Load a BEIR dataset.

    Args:
        dataset_name: Name of BEIR dataset (e.g., "scifact", "nfcorpus").
        split: Dataset split ("train", "dev", "test").

    Returns:
        Tuple of (corpus, queries, qrels):
        - corpus: dict[doc_id, {"title": str, "text": str}]
        - queries: dict[query_id, str]
        - qrels: dict[query_id, dict[doc_id, relevance_score]]

    Example:
        >>> corpus, queries, qrels = load_beir_dataset("scifact", "test")
        >>> print(len(queries))
        300

    Note:
        Uses BEIR library for downloading and loading:
        https://github.com/beir-cellar/beir
    """
    # TODO: Implementation
    # Hint: Use beir.util.download_and_unzip and GenericDataLoader
    raise NotImplementedError("load_beir_dataset not yet implemented")


def iter_queries(
    corpus: str,
    split: str = "train",
    batch_size: int = 100
) -> Iterator[list[tuple[str, str]]]:
    """
    Iterate over queries in batches (memory efficient).

    Args:
        corpus: Corpus name ("msmarco-passage", "beir/scifact", etc.).
        split: Dataset split.
        batch_size: Number of queries per batch.

    Yields:
        List of (query_id, query_text) tuples.

    Example:
        >>> for batch in iter_queries("msmarco-passage", batch_size=100):
        ...     process_batch(batch)
    """
    # TODO: Implementation
    raise NotImplementedError("iter_queries not yet implemented")


def get_query_entities(
    query: str,
    method: str = "simple"
) -> list[str]:
    """
    Extract candidate entities/terms from a query for negation.

    Args:
        query: Query text.
        method: Extraction method:
            - "simple": Split on whitespace, filter stopwords
            - "noun_phrases": Extract noun phrases (requires spaCy)
            - "ner": Named entity recognition (requires spaCy)

    Returns:
        List of candidate entities that could be negated.

    Example:
        >>> entities = get_query_entities("python web scraping selenium")
        >>> print(entities)
        ['python', 'selenium', 'scraping']

    Note:
        These candidates will be used to generate negated queries
        like "python web scraping without selenium".
    """
    # TODO: Implementation
    raise NotImplementedError("get_query_entities not yet implemented")
