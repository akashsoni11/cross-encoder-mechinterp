"""
Filtering and quality assurance for ConstraintSuite.

This module provides functions for:
- Filtering mined pairs by quality criteria
- Ensuring topical similarity between doc_pos and doc_neg
- Deduplication
- Constraint validity checks

Quality is critical: mechanistic analysis requires clean examples
where the constraint is the only deciding factor.
"""

import logging
from dataclasses import dataclass
from typing import Any

from constraintsuite.pair_mining import contains_y, y_is_negated_nearby

logger = logging.getLogger("constraintsuite")


@dataclass
class FilterResult:
    """
    Result of filtering a candidate pair.

    Attributes:
        passed: Whether the pair passed all filters.
        reason: If failed, the reason for rejection.
        scores: Dictionary of filter scores for debugging.
    """
    passed: bool
    reason: str | None
    scores: dict[str, float]


def tokenize(text: str) -> set[str]:
    """Simple whitespace tokenization with lowercasing."""
    return set(text.lower().split())


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def passes_length_filters(
    doc_pos: dict[str, Any],
    doc_neg: dict[str, Any],
    min_length: int = 50,
    max_length: int = 2000,
    max_ratio: float = 2.0
) -> FilterResult:
    """
    Check if documents pass length filters.

    Args:
        doc_pos: Positive document.
        doc_neg: Negative document.
        min_length: Minimum document length (characters).
        max_length: Maximum document length (characters).
        max_ratio: Maximum length ratio between documents.

    Returns:
        FilterResult indicating pass/fail and reason.

    Example:
        >>> result = passes_length_filters(doc_pos, doc_neg, min_length=50)
        >>> if not result.passed:
        ...     print(f"Rejected: {result.reason}")
    """
    text_pos = doc_pos.get("text", "")
    text_neg = doc_neg.get("text", "")

    len_pos = len(text_pos)
    len_neg = len(text_neg)

    scores = {
        "len_pos": len_pos,
        "len_neg": len_neg,
        "len_ratio": max(len_pos, len_neg) / max(min(len_pos, len_neg), 1),
    }

    # Check minimum length
    if len_pos < min_length:
        return FilterResult(False, f"doc_pos too short ({len_pos} < {min_length})", scores)
    if len_neg < min_length:
        return FilterResult(False, f"doc_neg too short ({len_neg} < {min_length})", scores)

    # Check maximum length
    if len_pos > max_length:
        return FilterResult(False, f"doc_pos too long ({len_pos} > {max_length})", scores)
    if len_neg > max_length:
        return FilterResult(False, f"doc_neg too long ({len_neg} > {max_length})", scores)

    # Check length ratio
    if scores["len_ratio"] > max_ratio:
        return FilterResult(
            False,
            f"length ratio too high ({scores['len_ratio']:.2f} > {max_ratio})",
            scores
        )

    return FilterResult(True, None, scores)


def compute_lexical_overlap(
    doc_pos: dict[str, Any],
    doc_neg: dict[str, Any],
    query: str | None = None
) -> dict[str, float]:
    """
    Compute lexical overlap metrics.

    Args:
        doc_pos: Positive document.
        doc_neg: Negative document.
        query: Optional query for query-document overlap.

    Returns:
        Dictionary with:
        - doc_doc_jaccard: Jaccard similarity between documents
        - doc_doc_overlap: Term overlap ratio
        - query_pos_overlap: Query term overlap with doc_pos
        - query_neg_overlap: Query term overlap with doc_neg

    Example:
        >>> scores = compute_lexical_overlap(doc_pos, doc_neg, query)
        >>> print(f"Doc-doc Jaccard: {scores['doc_doc_jaccard']:.2f}")
    """
    text_pos = doc_pos.get("text", "")
    text_neg = doc_neg.get("text", "")

    tokens_pos = tokenize(text_pos)
    tokens_neg = tokenize(text_neg)

    # Document-document similarity
    doc_doc_jaccard = jaccard_similarity(tokens_pos, tokens_neg)

    # Term overlap ratio (how many terms are shared)
    shared = len(tokens_pos & tokens_neg)
    total_unique = len(tokens_pos | tokens_neg)
    doc_doc_overlap = shared / total_unique if total_unique > 0 else 0.0

    scores = {
        "doc_doc_jaccard": doc_doc_jaccard,
        "doc_doc_overlap": doc_doc_overlap,
    }

    # Query-document overlap
    if query:
        tokens_query = tokenize(query)
        if tokens_query:
            scores["query_pos_overlap"] = len(tokens_pos & tokens_query) / len(tokens_query)
            scores["query_neg_overlap"] = len(tokens_neg & tokens_query) / len(tokens_query)
        else:
            scores["query_pos_overlap"] = 0.0
            scores["query_neg_overlap"] = 0.0

    return scores


def passes_similarity_filters(
    doc_pos: dict[str, Any],
    doc_neg: dict[str, Any],
    query: str,
    min_query_overlap: float = 0.3,
    min_doc_similarity: float = 0.2
) -> FilterResult:
    """
    Check if documents have sufficient topical similarity.

    Args:
        doc_pos: Positive document.
        doc_neg: Negative document.
        query: Query text.
        min_query_overlap: Minimum overlap with query terms.
        min_doc_similarity: Minimum document-document similarity.

    Returns:
        FilterResult with pass/fail and similarity scores.

    Purpose:
        Ensure both documents are on-topic and similar enough
        that the constraint (not topic) is the distinguishing factor.
    """
    scores = compute_lexical_overlap(doc_pos, doc_neg, query)

    # Check query overlap
    query_pos = scores.get("query_pos_overlap", 0.0)
    query_neg = scores.get("query_neg_overlap", 0.0)

    if query_pos < min_query_overlap:
        return FilterResult(
            False,
            f"doc_pos query overlap too low ({query_pos:.2f} < {min_query_overlap})",
            scores
        )
    if query_neg < min_query_overlap:
        return FilterResult(
            False,
            f"doc_neg query overlap too low ({query_neg:.2f} < {min_query_overlap})",
            scores
        )

    # Check document similarity
    doc_sim = scores.get("doc_doc_jaccard", 0.0)
    if doc_sim < min_doc_similarity:
        return FilterResult(
            False,
            f"doc similarity too low ({doc_sim:.2f} < {min_doc_similarity})",
            scores
        )

    return FilterResult(True, None, scores)


def validate_constraint(
    doc_pos: dict[str, Any],
    doc_neg: dict[str, Any],
    y_forms: list[str],
    slice_type: str
) -> FilterResult:
    """
    Validate that the constraint is correctly satisfied/violated.

    Args:
        doc_pos: Document that should satisfy constraint.
        doc_neg: Document that should violate constraint.
        y_forms: Surface forms of forbidden term Y.
        slice_type: Type of slice ("minpairs", "explicit", "omission").

    Returns:
        FilterResult indicating validity.

    Validation rules by slice:
        - minpairs: doc_pos and doc_neg differ only by negation edit
        - explicit: doc_pos mentions Y but negates it; doc_neg affirms Y
        - omission: doc_pos doesn't mention Y; doc_neg contains Y

    Example:
        >>> result = validate_constraint(doc_pos, doc_neg, ["selenium"], "omission")
        >>> if not result.passed:
        ...     print(f"Invalid: {result.reason}")
    """
    text_pos = doc_pos.get("text", "")
    text_neg = doc_neg.get("text", "")
    y = y_forms[0] if y_forms else ""

    scores = {
        "pos_contains_y": contains_y(text_pos, y_forms),
        "neg_contains_y": contains_y(text_neg, y_forms),
        "pos_y_negated": y_is_negated_nearby(text_pos, y) if contains_y(text_pos, y_forms) else False,
        "neg_y_negated": y_is_negated_nearby(text_neg, y) if contains_y(text_neg, y_forms) else False,
    }

    # doc_neg MUST contain Y affirmatively (not negated)
    if not scores["neg_contains_y"]:
        return FilterResult(False, "doc_neg doesn't contain Y", scores)
    if scores["neg_y_negated"]:
        return FilterResult(False, "doc_neg has Y negated (should be affirmative)", scores)

    # Validate by slice type
    if slice_type == "minpairs":
        # For minpairs, we check if it's been edited
        if doc_pos.get("is_edited"):
            # Edited minpair - should have negation in doc_pos
            if not scores["pos_y_negated"]:
                return FilterResult(False, "minpair doc_pos should have Y negated", scores)
        # Otherwise, assume it was correctly constructed
        pass

    elif slice_type == "explicit":
        # doc_pos should contain Y in negated form
        if not scores["pos_contains_y"]:
            return FilterResult(False, "explicit doc_pos should contain Y", scores)
        if not scores["pos_y_negated"]:
            return FilterResult(False, "explicit doc_pos should have Y negated", scores)

    elif slice_type == "omission":
        # doc_pos should NOT contain Y
        if scores["pos_contains_y"]:
            return FilterResult(False, "omission doc_pos should not contain Y", scores)

    else:
        logger.warning(f"Unknown slice type: {slice_type}")

    return FilterResult(True, None, scores)


def filter_pair(
    doc_pos: dict[str, Any],
    doc_neg: dict[str, Any],
    query: str,
    y_forms: list[str],
    slice_type: str,
    config: dict[str, Any]
) -> FilterResult:
    """
    Apply all filters to a candidate pair.

    Args:
        doc_pos: Positive document.
        doc_neg: Negative document.
        query: Query text.
        y_forms: Surface forms of Y.
        slice_type: Type of slice.
        config: Filter configuration.

    Returns:
        FilterResult with combined pass/fail decision.

    Filters applied (in order):
        1. Length filters
        2. Similarity filters
        3. Constraint validity
        4. Deduplication check (if enabled)
    """
    filter_config = config.get("filtering", {})

    all_scores = {}

    # 1. Length filters
    length_result = passes_length_filters(
        doc_pos, doc_neg,
        min_length=filter_config.get("min_doc_length", 50),
        max_length=filter_config.get("max_doc_length", 2000),
        max_ratio=filter_config.get("max_doc_length_ratio", 2.0)
    )
    all_scores.update(length_result.scores)
    if not length_result.passed:
        return FilterResult(False, length_result.reason, all_scores)

    # 2. Similarity filters
    similarity_result = passes_similarity_filters(
        doc_pos, doc_neg, query,
        min_query_overlap=filter_config.get("min_lexical_overlap", 0.3),
        min_doc_similarity=filter_config.get("min_doc_similarity", 0.2)
    )
    all_scores.update(similarity_result.scores)
    if not similarity_result.passed:
        return FilterResult(False, similarity_result.reason, all_scores)

    # 3. Constraint validity
    constraint_result = validate_constraint(doc_pos, doc_neg, y_forms, slice_type)
    all_scores.update({f"constraint_{k}": v for k, v in constraint_result.scores.items()})
    if not constraint_result.passed:
        return FilterResult(False, constraint_result.reason, all_scores)

    return FilterResult(True, None, all_scores)


def deduplicate_pairs(
    pairs: list[dict[str, Any]],
    method: str = "doc_pair"
) -> list[dict[str, Any]]:
    """
    Remove duplicate pairs.

    Args:
        pairs: List of mined pairs.
        method: Deduplication method:
            - "doc_pair": Remove if same (doc_pos_id, doc_neg_id)
            - "doc_neg": Remove if same doc_neg across queries
            - "doc_pos": Remove if same doc_pos across queries

    Returns:
        Deduplicated list of pairs.

    Example:
        >>> deduped = deduplicate_pairs(pairs, method="doc_pair")
        >>> print(f"Removed {len(pairs) - len(deduped)} duplicates")
    """
    seen = set()
    result = []

    for pair in pairs:
        doc_pos = pair.get("doc_pos", {})
        doc_neg = pair.get("doc_neg", {})
        pos_id = doc_pos.get("doc_id", "")
        neg_id = doc_neg.get("doc_id", "")

        if method == "doc_pair":
            key = (pos_id, neg_id)
        elif method == "doc_neg":
            key = neg_id
        elif method == "doc_pos":
            key = pos_id
        else:
            raise ValueError(f"Unknown deduplication method: {method}")

        if key not in seen:
            seen.add(key)
            result.append(pair)

    removed = len(pairs) - len(result)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate pairs (method: {method})")

    return result


def batch_filter(
    pairs: list[dict[str, Any]],
    config: dict[str, Any],
    return_rejected: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """
    Filter pairs in batch with statistics.

    Args:
        pairs: List of candidate pairs.
        config: Filter configuration.
        return_rejected: Whether to return rejected pairs.

    Returns:
        Tuple of (accepted_pairs, rejected_pairs or None).

    Example:
        >>> accepted, rejected = batch_filter(pairs, config, return_rejected=True)
        >>> print(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")
    """
    accepted = []
    rejected = [] if return_rejected else None

    rejection_reasons = {}

    for pair in pairs:
        doc_pos = pair.get("doc_pos", {})
        doc_neg = pair.get("doc_neg", {})
        raw_query = pair.get("query", "")
        # query can be a dict {"base": ..., "neg": ...} or a plain string
        query = raw_query.get("neg", raw_query.get("base", "")) if isinstance(raw_query, dict) else raw_query
        y_forms = pair.get("y_forms", [])
        slice_type = pair.get("slice_type", "")

        result = filter_pair(doc_pos, doc_neg, query, y_forms, slice_type, config)

        if result.passed:
            # Add filter scores to pair metadata
            if "filter_scores" not in pair:
                pair["filter_scores"] = {}
            pair["filter_scores"].update(result.scores)
            accepted.append(pair)
        else:
            if return_rejected:
                pair["rejection_reason"] = result.reason
                pair["filter_scores"] = result.scores
                rejected.append(pair)

            # Track rejection reasons
            reason = result.reason or "unknown"
            reason_key = reason.split("(")[0].strip()  # Simplify reason
            rejection_reasons[reason_key] = rejection_reasons.get(reason_key, 0) + 1

    # Log statistics
    total = len(pairs)
    accepted_count = len(accepted)
    logger.info(
        f"Filtering: {accepted_count}/{total} pairs accepted "
        f"({100 * accepted_count / total if total > 0 else 0:.1f}%)"
    )

    if rejection_reasons:
        logger.info(f"Rejection reasons: {rejection_reasons}")

    # Deduplicate accepted pairs
    accepted = deduplicate_pairs(accepted, method="doc_pair")

    return accepted, rejected
