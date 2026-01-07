"""
Document pair mining for ConstraintSuite.

This module provides the core logic for mining (doc_pos, doc_neg) pairs
from a candidate pool, where:
- doc_neg: violates the constraint (contains Y affirmatively)
- doc_pos: satisfies the constraint (omits Y or negates Y)

Key functions:
- contains_y: Check if text contains forbidden term
- y_is_negated_nearby: Check if Y appears in negated context
- mine_pair: Extract best (doc_pos, doc_neg) pair from candidates
"""

import re
from dataclasses import dataclass
from typing import Any


# Default negation markers (regex patterns)
DEFAULT_NEG_MARKERS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bfree of\b",
    r"\bexclude[sd]?\b",
    r"\blocks?\b",
    r"\babsent\b",
    r"\b-free\b",
]


@dataclass
class MinedPair:
    """
    A mined document pair with metadata.

    Attributes:
        doc_pos: Document satisfying the constraint.
        doc_neg: Document violating the constraint.
        slice_type: Type of pair ("minpairs", "explicit", "omission").
        metadata: Additional mining metadata.
    """
    doc_pos: dict[str, Any]
    doc_neg: dict[str, Any]
    slice_type: str
    metadata: dict[str, Any]


def contains_y(
    text: str,
    y_forms: list[str],
    case_sensitive: bool = False
) -> bool:
    """
    Check if text contains any surface form of Y.

    Args:
        text: Document text to search.
        y_forms: List of surface forms for Y.
        case_sensitive: Whether to match case-sensitively.

    Returns:
        True if any form of Y is found in text.

    Example:
        >>> contains_y("Install Selenium WebDriver", ["selenium", "webdriver"])
        True
        >>> contains_y("Use requests library", ["selenium"])
        False
    """
    # TODO: Implementation
    raise NotImplementedError("contains_y not yet implemented")


def y_is_negated_nearby(
    text: str,
    y: str,
    window: int = 40,
    neg_markers: list[str] | None = None
) -> bool:
    """
    Check if Y appears near a negation marker.

    Args:
        text: Document text to search.
        y: The term to check.
        window: Character window to check around Y.
        neg_markers: Regex patterns for negation markers.
            Defaults to DEFAULT_NEG_MARKERS.

    Returns:
        True if Y appears within `window` characters of a negation marker.

    Example:
        >>> y_is_negated_nearby("This recipe contains no peanuts", "peanuts")
        True
        >>> y_is_negated_nearby("Thai peanut noodles recipe", "peanuts")
        False
        >>> y_is_negated_nearby("peanut-free stir fry", "peanut")
        True

    Note:
        This is a heuristic. It may have false positives (e.g., "not only peanuts")
        and false negatives (e.g., "avoids using peanuts").
    """
    # TODO: Implementation
    raise NotImplementedError("y_is_negated_nearby not yet implemented")


def mine_pair(
    candidates: list[dict[str, Any]],
    y_forms: list[str],
    prefer_explicit: bool = True
) -> MinedPair | None:
    """
    Mine a (doc_pos, doc_neg) pair from candidates.

    This is the core mining function. It finds:
    - doc_neg: Contains Y affirmatively (not negated)
    - doc_pos: Either omits Y entirely OR contains Y in negated form

    Args:
        candidates: List of candidate documents from retrieval.
            Each dict has: doc_id, text, title, bm25_rank, bm25_score
        y_forms: Surface forms of the forbidden term Y.
        prefer_explicit: If True, prefer doc_pos that explicitly negates Y
            over doc_pos that simply omits Y.

    Returns:
        MinedPair if a valid pair is found, None otherwise.

    Example:
        >>> candidates = retriever.retrieve("python scraping without selenium", k=200)
        >>> pair = mine_pair(candidates, ["selenium", "webdriver"])
        >>> if pair:
        ...     print(f"doc_pos: {pair.doc_pos['doc_id']}")
        ...     print(f"doc_neg: {pair.doc_neg['doc_id']}")
        ...     print(f"slice: {pair.slice_type}")

    Mining strategy:
        1. Split candidates into violators (contain Y, not negated) and satisfiers
        2. Pick best violator as doc_neg (highest BM25 rank)
        3. Pick satisfier with closest BM25 rank to doc_neg (topical similarity)
        4. Determine slice type based on whether doc_pos mentions Y
    """
    # TODO: Implementation
    raise NotImplementedError("mine_pair not yet implemented")


def mine_minpair(
    doc: dict[str, Any],
    y: str,
    edit_type: str = "insert_no"
) -> MinedPair | None:
    """
    Create a minimal pair by surgically editing a document.

    For the MinPairs slice, we create doc_pos and doc_neg that differ
    only by a single edit (NevIR-style).

    Args:
        doc: Source document containing Y.
        y: The term to edit.
        edit_type: Type of edit:
            - "insert_no": Insert "no" before Y
            - "add_free": Change "Y" to "Y-free"
            - "replace_contains": Change "contains Y" to "contains no Y"

    Returns:
        MinedPair with surgically edited doc_pos and original doc_neg,
        or None if edit is not applicable.

    Example:
        >>> doc = {"text": "This recipe contains peanuts and almonds."}
        >>> pair = mine_minpair(doc, "peanuts", "replace_contains")
        >>> print(pair.doc_pos["text"])
        'This recipe contains no peanuts and almonds.'
        >>> print(pair.doc_neg["text"])
        'This recipe contains peanuts and almonds.'

    Note:
        MinPairs give the cleanest signal for mechanistic analysis
        because the only difference is the negation.
    """
    # TODO: Implementation
    raise NotImplementedError("mine_minpair not yet implemented")


def classify_pair_slice(
    doc_pos: dict[str, Any],
    doc_neg: dict[str, Any],
    y_forms: list[str]
) -> str:
    """
    Classify a pair into one of the three slices.

    Args:
        doc_pos: Document satisfying constraint.
        doc_neg: Document violating constraint.
        y_forms: Surface forms of Y.

    Returns:
        Slice type: "minpairs", "explicit", or "omission".

    Classification rules:
        - "minpairs": doc_pos and doc_neg are near-identical (high similarity)
        - "explicit": doc_pos mentions Y in negated form
        - "omission": doc_pos does not mention Y at all
    """
    # TODO: Implementation
    raise NotImplementedError("classify_pair_slice not yet implemented")


def batch_mine_pairs(
    query_candidates: dict[str, tuple[list[str], list[dict[str, Any]]]],
    config: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Mine pairs in batch for multiple queries.

    Args:
        query_candidates: Dict mapping query_id -> (y_forms, candidates).
        config: Mining configuration.

    Returns:
        List of mined examples ready for JSONL export.

    Example:
        >>> query_candidates = {
        ...     "q1": (["selenium"], candidates_q1),
        ...     "q2": (["peanuts"], candidates_q2),
        ... }
        >>> examples = batch_mine_pairs(query_candidates, config)
    """
    # TODO: Implementation
    raise NotImplementedError("batch_mine_pairs not yet implemented")
