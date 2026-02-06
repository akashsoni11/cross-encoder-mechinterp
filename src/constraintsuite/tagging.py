"""
Difficulty tagging and stratification for ConstraintSuite.

This module provides functions for:
- Computing difficulty tags for pairs
- Stratifying dataset by difficulty
- Sampling gold sets with appropriate distributions

Tags are used for:
- Analysis (understanding model behavior by difficulty)
- Gold set sampling (oversampling hard examples)
- Mechanistic analysis (different slices for different purposes)
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from constraintsuite.filtering import compute_lexical_overlap, tokenize
from constraintsuite.pair_mining import contains_y, y_is_negated_nearby

logger = logging.getLogger("constraintsuite")


@dataclass
class PairTags:
    """
    Tags for a mined pair.

    Attributes:
        difficulty: Overall difficulty ("easy", "medium", "hard").
        lexical_overlap_bin: Document overlap level ("low", "medium", "high").
        doc_length_bin: Average document length ("short", "medium", "long").
        doc_pos_mentions_y: Whether doc_pos mentions Y at all.
        y_negated_in_doc_pos: Whether Y is explicitly negated in doc_pos.
        negation_explicitness: How explicit the negation is.
    """
    difficulty: str
    lexical_overlap_bin: str
    doc_length_bin: str
    doc_pos_mentions_y: bool
    y_negated_in_doc_pos: bool
    negation_explicitness: str


def compute_lexical_overlap_bin(
    doc_pos: dict[str, Any],
    doc_neg: dict[str, Any],
    bins: list[float]
) -> str:
    """
    Compute lexical overlap bin between documents.

    Args:
        doc_pos: Positive document.
        doc_neg: Negative document.
        bins: Bin thresholds [low_max, medium_max].
            E.g., [0.3, 0.6] means:
            - < 0.3: "low"
            - 0.3-0.6: "medium"
            - >= 0.6: "high"

    Returns:
        Bin label ("low", "medium", "high").

    Example:
        >>> bin_label = compute_lexical_overlap_bin(doc_pos, doc_neg, [0.3, 0.6])
        >>> print(bin_label)
        'high'
    """
    scores = compute_lexical_overlap(doc_pos, doc_neg)
    jaccard = scores.get("doc_doc_jaccard", 0.0)

    if len(bins) >= 2:
        low_threshold, high_threshold = bins[0], bins[1]
    else:
        low_threshold, high_threshold = 0.3, 0.6

    if jaccard < low_threshold:
        return "low"
    elif jaccard < high_threshold:
        return "medium"
    else:
        return "high"


def compute_doc_length_bin(
    doc_pos: dict[str, Any],
    doc_neg: dict[str, Any],
    bins: list[int]
) -> str:
    """
    Compute document length bin.

    Args:
        doc_pos: Positive document.
        doc_neg: Negative document.
        bins: Bin thresholds in tokens [short_max, medium_max].
            E.g., [128, 256] means:
            - < 128 tokens: "short"
            - 128-256 tokens: "medium"
            - >= 256 tokens: "long"

    Returns:
        Bin label ("short", "medium", "long").
    """
    text_pos = doc_pos.get("text", "")
    text_neg = doc_neg.get("text", "")

    # Count tokens (simple whitespace split)
    tokens_pos = len(text_pos.split())
    tokens_neg = len(text_neg.split())
    avg_tokens = (tokens_pos + tokens_neg) / 2

    if len(bins) >= 2:
        short_max, medium_max = bins[0], bins[1]
    else:
        short_max, medium_max = 128, 256

    if avg_tokens < short_max:
        return "short"
    elif avg_tokens < medium_max:
        return "medium"
    else:
        return "long"


def compute_difficulty(
    doc_pos: dict[str, Any],
    doc_neg: dict[str, Any],
    y_forms: list[str],
    lexical_overlap_bin: str
) -> str:
    """
    Compute overall difficulty of a pair.

    Args:
        doc_pos: Positive document.
        doc_neg: Negative document.
        y_forms: Surface forms of Y.
        lexical_overlap_bin: Pre-computed overlap bin.

    Returns:
        Difficulty label ("easy", "medium", "hard").

    Heuristics:
        - "easy": Low overlap (topic difference helps distinguish)
        - "medium": Medium overlap, clear constraint signal
        - "hard": High overlap AND doc_pos doesn't mention Y
            (model must infer absence)

    Example:
        >>> difficulty = compute_difficulty(doc_pos, doc_neg, ["selenium"], "high")
        >>> print(difficulty)
        'hard'
    """
    text_pos = doc_pos.get("text", "")
    y = y_forms[0] if y_forms else ""

    # Check if doc_pos contains Y
    pos_contains_y = contains_y(text_pos, y_forms)
    pos_y_negated = y_is_negated_nearby(text_pos, y) if pos_contains_y else False

    if lexical_overlap_bin == "low":
        # Low overlap = easy (topic difference helps)
        return "easy"
    elif lexical_overlap_bin == "high":
        if not pos_contains_y:
            # High overlap + omission = hardest
            # Model must infer that absence of Y is significant
            return "hard"
        elif pos_y_negated:
            # High overlap + explicit negation = medium-hard
            return "medium"
        else:
            # High overlap but Y not properly negated (shouldn't happen)
            return "hard"
    else:  # medium overlap
        if pos_contains_y and pos_y_negated:
            # Clear explicit negation signal
            return "medium"
        elif not pos_contains_y:
            # Must infer from omission
            return "hard"
        else:
            return "medium"


def compute_negation_explicitness(
    doc_pos: dict[str, Any],
    y_forms: list[str]
) -> str:
    """
    Compute how explicitly Y is negated in doc_pos.

    Returns:
        - "explicit": doc_pos contains Y with clear negation marker
        - "implicit": doc_pos doesn't contain Y (constraint by omission)
        - "none": doc_pos contains Y without negation (error case)
    """
    text_pos = doc_pos.get("text", "")
    y = y_forms[0] if y_forms else ""

    if not contains_y(text_pos, y_forms):
        return "implicit"
    elif y_is_negated_nearby(text_pos, y):
        return "explicit"
    else:
        return "none"


def tag_pair(
    doc_pos: dict[str, Any],
    doc_neg: dict[str, Any],
    y_forms: list[str],
    config: dict[str, Any]
) -> PairTags:
    """
    Compute all tags for a pair.

    Args:
        doc_pos: Positive document.
        doc_neg: Negative document.
        y_forms: Surface forms of Y.
        config: Tagging configuration with bin thresholds.

    Returns:
        PairTags with all computed tags.

    Example:
        >>> tags = tag_pair(doc_pos, doc_neg, ["selenium"], config)
        >>> print(f"Difficulty: {tags.difficulty}")
        >>> print(f"Overlap: {tags.lexical_overlap_bin}")
    """
    tagging_config = config.get("tagging", {})

    # Get bin thresholds from config
    overlap_bins = tagging_config.get("lexical_overlap_bins", [0.3, 0.6])
    length_bins = tagging_config.get("doc_length_bins", [128, 256])

    # Compute bins
    lexical_overlap_bin = compute_lexical_overlap_bin(doc_pos, doc_neg, overlap_bins)
    doc_length_bin = compute_doc_length_bin(doc_pos, doc_neg, length_bins)

    # Compute Y-related features
    text_pos = doc_pos.get("text", "")
    y = y_forms[0] if y_forms else ""
    doc_pos_mentions_y = contains_y(text_pos, y_forms)
    y_negated_in_doc_pos = y_is_negated_nearby(text_pos, y) if doc_pos_mentions_y else False

    # Compute difficulty
    difficulty = compute_difficulty(doc_pos, doc_neg, y_forms, lexical_overlap_bin)

    # Compute negation explicitness
    negation_explicitness = compute_negation_explicitness(doc_pos, y_forms)

    return PairTags(
        difficulty=difficulty,
        lexical_overlap_bin=lexical_overlap_bin,
        doc_length_bin=doc_length_bin,
        doc_pos_mentions_y=doc_pos_mentions_y,
        y_negated_in_doc_pos=y_negated_in_doc_pos,
        negation_explicitness=negation_explicitness,
    )


def batch_tag(
    pairs: list[dict[str, Any]],
    config: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Tag pairs in batch.

    Args:
        pairs: List of pairs to tag.
        config: Tagging configuration.

    Returns:
        Pairs with tags added to each.

    Example:
        >>> tagged = batch_tag(pairs, config)
        >>> print(tagged[0]["tags"]["difficulty"])
    """
    for pair in pairs:
        doc_pos = pair.get("doc_pos", {})
        doc_neg = pair.get("doc_neg", {})
        y_forms = pair.get("y_forms", [])

        tags = tag_pair(doc_pos, doc_neg, y_forms, config)

        # Add tags to pair
        pair["tags"] = {
            "difficulty": tags.difficulty,
            "lexical_overlap_bin": tags.lexical_overlap_bin,
            "doc_length_bin": tags.doc_length_bin,
            "doc_pos_mentions_y": tags.doc_pos_mentions_y,
            "y_negated_in_doc_pos": tags.y_negated_in_doc_pos,
            "negation_explicitness": tags.negation_explicitness,
        }

    # Log distribution
    stats = compute_distribution_stats(pairs)
    logger.info(f"Tagged {len(pairs)} pairs")
    logger.info(f"  Difficulty: {stats['difficulty_counts']}")
    logger.info(f"  Overlap: {stats['overlap_counts']}")

    return pairs


def stratify_by_difficulty(
    pairs: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """
    Group pairs by difficulty.

    Args:
        pairs: List of tagged pairs.

    Returns:
        Dictionary mapping difficulty -> list of pairs.

    Example:
        >>> stratified = stratify_by_difficulty(pairs)
        >>> print(f"Easy: {len(stratified['easy'])}")
        >>> print(f"Hard: {len(stratified['hard'])}")
    """
    stratified = defaultdict(list)

    for pair in pairs:
        tags = pair.get("tags", {})
        difficulty = tags.get("difficulty", "unknown")
        stratified[difficulty].append(pair)

    return dict(stratified)


def stratify_by_slice(
    pairs: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """
    Group pairs by slice type.

    Args:
        pairs: List of pairs.

    Returns:
        Dictionary mapping slice_type -> list of pairs.
    """
    stratified = defaultdict(list)

    for pair in pairs:
        slice_type = pair.get("slice_type", "unknown")
        stratified[slice_type].append(pair)

    return dict(stratified)


def sample_gold_set(
    pairs: list[dict[str, Any]],
    target_size: int = 50,
    difficulty_distribution: list[float] | None = None,
    oversample_hard: bool = True,
    seed: int | None = None
) -> list[dict[str, Any]]:
    """
    Sample a gold set with appropriate difficulty distribution.

    Args:
        pairs: Pool of pairs to sample from.
        target_size: Target number of examples.
        difficulty_distribution: Distribution [easy%, medium%, hard%].
            Defaults to [0.15, 0.35, 0.50] if oversample_hard.
        oversample_hard: Whether to oversample hard examples.
        seed: Random seed for reproducibility.

    Returns:
        Sampled gold set.

    Example:
        >>> gold = sample_gold_set(pairs, target_size=50, oversample_hard=True)
        >>> print(len(gold))
        50

    Purpose:
        Gold set is used for:
        - Manual verification
        - Qualitative analysis
        - Patching/ablation demos
        Rich in hard cases helps surface interesting failure modes.
    """
    if seed is not None:
        random.seed(seed)

    # Default distribution
    if difficulty_distribution is None:
        if oversample_hard:
            difficulty_distribution = [0.15, 0.35, 0.50]  # easy, medium, hard
        else:
            difficulty_distribution = [0.33, 0.34, 0.33]

    # Stratify by difficulty
    stratified = stratify_by_difficulty(pairs)

    # Calculate targets per difficulty
    targets = {
        "easy": int(target_size * difficulty_distribution[0]),
        "medium": int(target_size * difficulty_distribution[1]),
        "hard": int(target_size * difficulty_distribution[2]),
    }

    # Adjust to hit exact target
    total_target = sum(targets.values())
    if total_target < target_size:
        targets["hard"] += target_size - total_target

    # Sample from each difficulty
    gold = []
    for difficulty, target in targets.items():
        pool = stratified.get(difficulty, [])
        if len(pool) <= target:
            gold.extend(pool)
        else:
            gold.extend(random.sample(pool, target))

    # If we still need more, sample from remaining
    if len(gold) < target_size:
        remaining = [p for p in pairs if p not in gold]
        needed = target_size - len(gold)
        if len(remaining) >= needed:
            gold.extend(random.sample(remaining, needed))
        else:
            gold.extend(remaining)

    # Shuffle final set
    random.shuffle(gold)

    # Log distribution
    stats = compute_distribution_stats(gold)
    logger.info(
        f"Sampled gold set: {len(gold)} examples "
        f"(difficulty: {stats['difficulty_counts']})"
    )

    return gold


def sample_by_slice(
    pairs: list[dict[str, Any]],
    target_size: int,
    slice_distribution: dict[str, float],
    seed: int | None = None
) -> list[dict[str, Any]]:
    """
    Sample pairs according to slice distribution.

    Args:
        pairs: Pool of pairs.
        target_size: Target total size.
        slice_distribution: Dict mapping slice -> target fraction.
        seed: Random seed.

    Returns:
        Sampled pairs matching slice distribution.
    """
    if seed is not None:
        random.seed(seed)

    stratified = stratify_by_slice(pairs)

    # Calculate targets
    targets = {}
    for slice_type, fraction in slice_distribution.items():
        targets[slice_type] = int(target_size * fraction)

    # Adjust to hit exact target
    total = sum(targets.values())
    if total < target_size:
        # Add remaining to largest slice
        max_slice = max(slice_distribution, key=slice_distribution.get)
        targets[max_slice] += target_size - total

    # Sample
    result = []
    for slice_type, target in targets.items():
        pool = stratified.get(slice_type, [])
        if len(pool) <= target:
            result.extend(pool)
        else:
            result.extend(random.sample(pool, target))

    random.shuffle(result)
    return result


def compute_distribution_stats(
    pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Compute distribution statistics for a dataset.

    Args:
        pairs: List of tagged pairs.

    Returns:
        Dictionary with distribution statistics:
        - difficulty_counts: {easy: N, medium: N, hard: N}
        - overlap_counts: {low: N, medium: N, high: N}
        - length_counts: {short: N, medium: N, long: N}
        - slice_counts: {minpairs: N, explicit: N, omission: N}

    Example:
        >>> stats = compute_distribution_stats(pairs)
        >>> print(f"Hard examples: {stats['difficulty_counts']['hard']}")
    """
    difficulty_counts = defaultdict(int)
    overlap_counts = defaultdict(int)
    length_counts = defaultdict(int)
    slice_counts = defaultdict(int)
    explicitness_counts = defaultdict(int)

    for pair in pairs:
        tags = pair.get("tags", {})
        slice_type = pair.get("slice_type", "unknown")

        difficulty_counts[tags.get("difficulty", "unknown")] += 1
        overlap_counts[tags.get("lexical_overlap_bin", "unknown")] += 1
        length_counts[tags.get("doc_length_bin", "unknown")] += 1
        slice_counts[slice_type] += 1
        explicitness_counts[tags.get("negation_explicitness", "unknown")] += 1

    return {
        "total": len(pairs),
        "difficulty_counts": dict(difficulty_counts),
        "overlap_counts": dict(overlap_counts),
        "length_counts": dict(length_counts),
        "slice_counts": dict(slice_counts),
        "explicitness_counts": dict(explicitness_counts),
    }


def format_stats_report(stats: dict[str, Any]) -> str:
    """Format distribution stats as a readable report."""
    lines = [
        f"Dataset Statistics (n={stats['total']})",
        "=" * 40,
        "",
        "Difficulty Distribution:",
    ]

    for difficulty in ["easy", "medium", "hard"]:
        count = stats["difficulty_counts"].get(difficulty, 0)
        pct = 100 * count / stats["total"] if stats["total"] > 0 else 0
        lines.append(f"  {difficulty}: {count} ({pct:.1f}%)")

    lines.extend(["", "Slice Distribution:"])
    for slice_type in ["minpairs", "explicit", "omission"]:
        count = stats["slice_counts"].get(slice_type, 0)
        pct = 100 * count / stats["total"] if stats["total"] > 0 else 0
        lines.append(f"  {slice_type}: {count} ({pct:.1f}%)")

    lines.extend(["", "Lexical Overlap:"])
    for overlap in ["low", "medium", "high"]:
        count = stats["overlap_counts"].get(overlap, 0)
        pct = 100 * count / stats["total"] if stats["total"] > 0 else 0
        lines.append(f"  {overlap}: {count} ({pct:.1f}%)")

    return "\n".join(lines)
