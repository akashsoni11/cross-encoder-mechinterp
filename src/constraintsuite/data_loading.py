"""
Data loading utilities for ConstraintSuite.

This module provides functions for loading IR corpora and queries from:
- MS MARCO (via Pyserini or ir_datasets)
- BEIR datasets

Primary data source: MS MARCO passages with prebuilt Pyserini index.
"""

import logging
from typing import Iterator

import ir_datasets

logger = logging.getLogger("constraintsuite")

# Stopwords for filtering entity candidates
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "from", "in", "into", "of", "on", "to", "with",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "can", "this", "that", "these", "those", "what", "which", "who",
    "how", "why", "where", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "also", "now", "about", "after", "before", "between", "during",
    "through", "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "further", "once", "here", "there", "any", "many", "much",
}


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
    # Map split names to ir_datasets names
    split_map = {
        "train": "msmarco-passage/train/queries",
        "dev": "msmarco-passage/dev/small",
        "eval": "msmarco-passage/eval/small",
    }

    dataset_name = split_map.get(split, f"msmarco-passage/{split}")

    # Try to load with queries
    try:
        dataset = ir_datasets.load(dataset_name)
    except KeyError:
        # Fall back to just the train split
        dataset = ir_datasets.load("msmarco-passage/train")

    queries = {}
    for i, query in enumerate(dataset.queries_iter()):
        if limit is not None and i >= limit:
            break
        queries[query.query_id] = query.text

    logger.info(f"Loaded {len(queries)} queries from MS MARCO {split}")
    return queries


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
    dataset = ir_datasets.load("msmarco-passage")

    corpus = {}
    for i, doc in enumerate(dataset.docs_iter()):
        if limit is not None and i >= limit:
            break
        corpus[doc.doc_id] = {
            "text": doc.text,
            "title": None,  # MS MARCO passages don't have titles
        }

    logger.info(f"Loaded {len(corpus)} documents from MS MARCO corpus")
    return corpus


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
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    # Download dataset if not present
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, "data/beir")

    # Load with GenericDataLoader
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    logger.info(
        f"Loaded BEIR {dataset_name}/{split}: "
        f"{len(corpus)} docs, {len(queries)} queries"
    )
    return corpus, queries, qrels


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
    if corpus == "msmarco-passage":
        dataset = ir_datasets.load(f"msmarco-passage/{split}")
        batch = []
        for query in dataset.queries_iter():
            batch.append((query.query_id, query.text))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # Yield remaining
            yield batch
    elif corpus.startswith("beir/"):
        # Load BEIR queries
        dataset_name = corpus.split("/")[1]
        _, queries, _ = load_beir_dataset(dataset_name, split)
        items = list(queries.items())
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    else:
        raise ValueError(f"Unknown corpus: {corpus}")


def get_query_entities(
    query: str,
    method: str = "noun_phrases"
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
    if method == "simple":
        return _get_entities_simple(query)
    elif method == "noun_phrases":
        return _get_entities_noun_phrases(query)
    elif method == "ner":
        return _get_entities_ner(query)
    else:
        raise ValueError(f"Unknown method: {method}")


def _get_entities_simple(query: str) -> list[str]:
    """Simple tokenization with stopword filtering."""
    words = query.lower().split()
    entities = [
        w for w in words
        if w not in STOPWORDS
        and len(w) >= 2
        and w.isalpha()
    ]
    return entities


# Global spaCy model cache
_nlp = None


def _get_spacy_model():
    """Load spaCy model (cached)."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed, download it
            logger.info("Downloading spaCy en_core_web_sm model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_entities_noun_phrases(query: str) -> list[str]:
    """Extract noun phrases using spaCy."""
    nlp = _get_spacy_model()
    doc = nlp(query)

    entities = []

    # Extract noun chunks
    for chunk in doc.noun_chunks:
        # Get the root noun and full chunk
        text = chunk.text.lower().strip()
        root = chunk.root.text.lower()

        # Filter out stopwords and short terms
        if text not in STOPWORDS and len(text) >= 2:
            entities.append(text)
        if root not in STOPWORDS and len(root) >= 2 and root != text:
            entities.append(root)

    # Also add individual nouns not in chunks
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            text = token.text.lower()
            if text not in STOPWORDS and len(text) >= 2 and text not in entities:
                entities.append(text)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique.append(e)

    return unique


def _get_entities_ner(query: str) -> list[str]:
    """Extract named entities using spaCy NER."""
    nlp = _get_spacy_model()
    doc = nlp(query)

    entities = []
    for ent in doc.ents:
        text = ent.text.lower()
        if text not in STOPWORDS and len(text) >= 2:
            entities.append(text)

    # Fall back to noun phrases if no named entities found
    if not entities:
        entities = _get_entities_noun_phrases(query)

    return entities


def batch_get_last_nouns(queries: list[str], batch_size: int = 1000) -> list[str | None]:
    """
    Extract last noun from each query using batched spaCy processing.

    Args:
        queries: List of query texts.
        batch_size: spaCy pipe batch size.

    Returns:
        List of last nouns (or None) aligned with input queries.
    """
    from tqdm import tqdm

    nlp = _get_spacy_model()
    results: list[str | None] = [None] * len(queries)

    docs = nlp.pipe(queries, batch_size=batch_size, disable=["ner", "lemmatizer"])
    for i, doc in enumerate(tqdm(docs, total=len(queries), desc="Extracting entities")):
        # Get last noun chunk
        chunks = list(doc.noun_chunks)
        if chunks:
            text = chunks[-1].text.lower().strip()
            if text not in STOPWORDS and len(text) >= 2:
                results[i] = text
                continue

        # Fall back to last NOUN token
        for token in reversed(doc):
            if token.pos_ in ("NOUN", "PROPN"):
                text = token.text.lower()
                if text not in STOPWORDS and len(text) >= 2:
                    results[i] = text
                    break

    return results


def get_last_noun(query: str) -> str | None:
    """
    Get the last noun/noun phrase from a query.

    This is often a good candidate for negation as queries tend to have
    modifiers at the end (e.g., "recipes without peanuts").

    Args:
        query: Query text.

    Returns:
        Last noun phrase or None if not found.
    """
    nlp = _get_spacy_model()
    doc = nlp(query)

    # Get last noun chunk
    chunks = list(doc.noun_chunks)
    if chunks:
        last_chunk = chunks[-1]
        text = last_chunk.text.lower().strip()
        if text not in STOPWORDS and len(text) >= 2:
            return text

    # Fall back to last NOUN token
    for token in reversed(doc):
        if token.pos_ in ("NOUN", "PROPN"):
            text = token.text.lower()
            if text not in STOPWORDS and len(text) >= 2:
                return text

    return None
