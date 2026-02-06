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

import copy
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger("constraintsuite")


# Default negation markers (regex patterns)
DEFAULT_NEG_MARKERS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bfree of\b",
    r"\bexclud(?:e[sd]?|ing)\b",
    r"\blacks?\b",
    r"\babsent\b",
    r"\b-free\b",
    r"\bfree\b",
    r"\bavoid",
    r"\bdon'?t\b",
    r"\bdoesn'?t\b",
    r"\bwon'?t\b",
    r"\bcan'?t\b",
    r"\bnever\b",
    r"\bnone\b",
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
    metadata: dict[str, Any] = field(default_factory=dict)


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
    if not case_sensitive:
        text = text.lower()
        y_forms = [y.lower() for y in y_forms]

    for y in y_forms:
        escaped = re.escape(y)
        # Use word boundaries when the term starts/ends with word chars,
        # otherwise fall back to plain match (handles "C++", ".NET", etc.)
        prefix = r"\b" if re.match(r"\w", y) else ""
        suffix = r"\b" if re.search(r"\w$", y) else ""
        pattern = f"{prefix}{escaped}{suffix}"
        if re.search(pattern, text, re.IGNORECASE if not case_sensitive else 0):
            return True

    return False


def find_y_positions(text: str, y: str, case_sensitive: bool = False) -> list[tuple[int, int]]:
    """
    Find all positions of Y in text.

    Args:
        text: Text to search.
        y: Term to find.
        case_sensitive: Whether to match case-sensitively.

    Returns:
        List of (start, end) positions.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = rf"\b{re.escape(y)}\b"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags)]


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
    if neg_markers is None:
        neg_markers = DEFAULT_NEG_MARKERS

    text_lower = text.lower()
    y_lower = y.lower()

    # Find all positions of Y
    y_positions = find_y_positions(text_lower, y_lower)

    if not y_positions:
        return False

    # Check each negation marker
    for marker_pattern in neg_markers:
        for match in re.finditer(marker_pattern, text_lower, re.IGNORECASE):
            marker_start = match.start()
            marker_end = match.end()

            # Check if any Y occurrence is within window of this marker
            for y_start, y_end in y_positions:
                # Check if marker is before Y (within window)
                if marker_end <= y_start and (y_start - marker_end) <= window:
                    return True
                # Check if marker is after Y (within window) - less common but possible
                if y_end <= marker_start and (marker_start - y_end) <= window:
                    return True
                # Check for compound patterns like "Y-free"
                if marker_pattern == r"\b-free\b" or marker_pattern == r"\bfree\b":
                    # Check for pattern like "peanut-free"
                    if y_end <= marker_start and text_lower[y_end:marker_start].strip() in ("", "-"):
                        return True

    return False


def mine_pair(
    candidates: list[dict[str, Any]],
    y_forms: list[str],
    prefer_explicit: bool = True,
    neg_window: int = 40
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
        neg_window: Window size for negation detection.

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
    if not candidates:
        return None

    y = y_forms[0] if y_forms else ""

    # Split candidates into violators and satisfiers
    violators = []  # Contains Y affirmatively
    satisfiers_explicit = []  # Contains Y but negated
    satisfiers_omission = []  # Doesn't contain Y

    for doc in candidates:
        text = doc.get("text", "")

        if contains_y(text, y_forms):
            # Document mentions Y
            if y_is_negated_nearby(text, y, window=neg_window):
                # Y is negated -> satisfies constraint
                satisfiers_explicit.append(doc)
            else:
                # Y is not negated -> violates constraint
                violators.append(doc)
        else:
            # Document doesn't mention Y -> satisfies constraint
            satisfiers_omission.append(doc)

    # Need at least one violator
    if not violators:
        return None

    # Need at least one satisfier
    if not satisfiers_explicit and not satisfiers_omission:
        return None

    # Pick doc_neg: highest-ranked violator (lowest bm25_rank value)
    doc_neg = min(violators, key=lambda d: d.get("bm25_rank", float("inf")))
    neg_rank = doc_neg.get("bm25_rank", 0)

    # Pick doc_pos: prefer explicit, then pick closest rank
    if prefer_explicit and satisfiers_explicit:
        # Prefer explicit negation
        doc_pos = min(
            satisfiers_explicit,
            key=lambda d: abs(d.get("bm25_rank", float("inf")) - neg_rank)
        )
        slice_type = "explicit"
    elif satisfiers_explicit:
        doc_pos = min(
            satisfiers_explicit,
            key=lambda d: abs(d.get("bm25_rank", float("inf")) - neg_rank)
        )
        slice_type = "explicit"
    else:
        doc_pos = min(
            satisfiers_omission,
            key=lambda d: abs(d.get("bm25_rank", float("inf")) - neg_rank)
        )
        slice_type = "omission"

    # Compute metadata
    pos_rank = doc_pos.get("bm25_rank", 0)
    rank_diff = abs(pos_rank - neg_rank)

    metadata = {
        "neg_rank": neg_rank,
        "pos_rank": pos_rank,
        "rank_diff": rank_diff,
        "num_violators": len(violators),
        "num_satisfiers_explicit": len(satisfiers_explicit),
        "num_satisfiers_omission": len(satisfiers_omission),
    }

    return MinedPair(
        doc_pos=doc_pos,
        doc_neg=doc_neg,
        slice_type=slice_type,
        metadata=metadata
    )


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
    text = doc.get("text", "")
    y_lower = y.lower()

    # Check if Y is in the text
    if not contains_y(text, [y]):
        return None

    # Already negated? Skip
    if y_is_negated_nearby(text, y):
        return None

    edited_text = None

    if edit_type == "insert_no":
        # Insert "no" before Y
        # Find Y in text (case-insensitive) and insert "no " before it
        pattern = rf"(\b)({re.escape(y)})(\b)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start, end = match.start(2), match.end(2)
            edited_text = text[:start] + "no " + text[start:]

    elif edit_type == "add_free":
        # Change "Y" to "Y-free"
        pattern = rf"\b({re.escape(y)})\b"
        edited_text = re.sub(
            pattern,
            r"\1-free",
            text,
            count=1,
            flags=re.IGNORECASE
        )

    elif edit_type == "replace_contains":
        # Change "contains Y" or "has Y" or "includes Y" to "contains no Y"
        patterns = [
            (rf"(contains?\s+)({re.escape(y)})", r"\1no \2"),
            (rf"(has\s+)({re.escape(y)})", r"\1no \2"),
            (rf"(includes?\s+)({re.escape(y)})", r"\1no \2"),
            (rf"(with\s+)({re.escape(y)})", r"without \2"),
            (rf"(using\s+)({re.escape(y)})", r"not using \2"),
        ]
        for pattern, replacement in patterns:
            new_text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
            if new_text != text:
                edited_text = new_text
                break

    elif edit_type == "without":
        # Change "with Y" to "without Y"
        pattern = rf"\bwith\s+({re.escape(y)})\b"
        new_text = re.sub(pattern, r"without \1", text, count=1, flags=re.IGNORECASE)
        if new_text != text:
            edited_text = new_text

    else:
        logger.warning(f"Unknown edit type: {edit_type}")
        return None

    if edited_text is None or edited_text == text:
        return None

    # Create doc_pos (edited) and doc_neg (original)
    doc_pos = copy.deepcopy(doc)
    doc_pos["text"] = edited_text
    doc_pos["doc_id"] = f"{doc['doc_id']}_edited"
    doc_pos["is_edited"] = True
    doc_pos["edit_type"] = edit_type

    doc_neg = copy.deepcopy(doc)
    doc_neg["is_edited"] = False

    return MinedPair(
        doc_pos=doc_pos,
        doc_neg=doc_neg,
        slice_type="minpairs",
        metadata={
            "edit_type": edit_type,
            "original_text": text,
            "edited_text": edited_text,
        }
    )


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
    text_pos = doc_pos.get("text", "")
    text_neg = doc_neg.get("text", "")

    # Check for minpairs (high similarity)
    similarity = SequenceMatcher(None, text_pos, text_neg).ratio()
    if similarity > 0.9:
        return "minpairs"

    # Check if marked as edited
    if doc_pos.get("is_edited"):
        return "minpairs"

    # Check if doc_pos contains Y
    y = y_forms[0] if y_forms else ""
    if contains_y(text_pos, y_forms):
        # Check if Y is negated
        if y_is_negated_nearby(text_pos, y):
            return "explicit"
        else:
            # This shouldn't happen if mining is correct
            logger.warning(f"doc_pos contains Y but not negated: {doc_pos.get('doc_id')}")
            return "explicit"
    else:
        return "omission"


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
    neg_window = config.get("filtering", {}).get("negation_window", 40)
    prefer_explicit = config.get("prefer_explicit", True)
    slice_targets = config.get("slice_distribution", {})

    results = []
    stats = {"total": 0, "success": 0, "minpairs": 0, "explicit": 0, "omission": 0}

    for query_id, (y_forms, candidates) in query_candidates.items():
        stats["total"] += 1

        # Mine regular pair
        pair = mine_pair(
            candidates,
            y_forms,
            prefer_explicit=prefer_explicit,
            neg_window=neg_window
        )

        if pair:
            stats["success"] += 1
            stats[pair.slice_type] += 1

            example = {
                "query_id": query_id,
                "y": y_forms[0] if y_forms else "",
                "y_forms": y_forms,
                "doc_pos": pair.doc_pos,
                "doc_neg": pair.doc_neg,
                "slice_type": pair.slice_type,
                "metadata": pair.metadata,
            }
            results.append(example)

        # Try to create minpairs from violators
        if candidates:
            # Find a good document for minpair editing
            for doc in candidates:
                if not contains_y(doc.get("text", ""), y_forms):
                    continue
                if y_is_negated_nearby(doc.get("text", ""), y_forms[0] if y_forms else ""):
                    continue

                # Try different edit types
                for edit_type in ["insert_no", "replace_contains", "add_free"]:
                    minpair = mine_minpair(doc, y_forms[0] if y_forms else "", edit_type)
                    if minpair:
                        stats["minpairs"] += 1
                        example = {
                            "query_id": f"{query_id}_minpair",
                            "y": y_forms[0] if y_forms else "",
                            "y_forms": y_forms,
                            "doc_pos": minpair.doc_pos,
                            "doc_neg": minpair.doc_neg,
                            "slice_type": "minpairs",
                            "metadata": minpair.metadata,
                        }
                        results.append(example)
                        break  # One minpair per query is enough
                break  # Only try first suitable document

    logger.info(
        f"Mined {stats['success']}/{stats['total']} pairs "
        f"(minpairs: {stats['minpairs']}, explicit: {stats['explicit']}, omission: {stats['omission']})"
    )

    return results


def mine_all_edit_types(
    doc: dict[str, Any],
    y: str
) -> list[MinedPair]:
    """
    Try all edit types on a document and return successful minpairs.

    Args:
        doc: Source document.
        y: Term to negate.

    Returns:
        List of successful MinedPair objects.
    """
    edit_types = ["insert_no", "add_free", "replace_contains", "without"]
    pairs = []

    for edit_type in edit_types:
        pair = mine_minpair(doc, y, edit_type)
        if pair:
            pairs.append(pair)

    return pairs
