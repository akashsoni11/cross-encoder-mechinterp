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

from dataclasses import dataclass
from typing import Any


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
    # TODO: Implementation
    raise NotImplementedError("compute_lexical_overlap_bin not yet implemented")


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
    # TODO: Implementation
    raise NotImplementedError("compute_doc_length_bin not yet implemented")


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
    # TODO: Implementation
    raise NotImplementedError("compute_difficulty not yet implemented")


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
    # TODO: Implementation
    raise NotImplementedError("tag_pair not yet implemented")


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
    # TODO: Implementation
    raise NotImplementedError("batch_tag not yet implemented")


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
    # TODO: Implementation
    raise NotImplementedError("stratify_by_difficulty not yet implemented")


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
    # TODO: Implementation
    raise NotImplementedError("sample_gold_set not yet implemented")


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
    # TODO: Implementation
    raise NotImplementedError("compute_distribution_stats not yet implemented")
