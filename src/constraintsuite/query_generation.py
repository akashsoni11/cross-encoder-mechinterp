"""
Query generation utilities for ConstraintSuite.

This module provides functions for:
- Generating negated queries from base queries
- Applying negation templates
- Selecting appropriate entities for negation

Templates supported:
- WITHOUT_Y: "{topic} without {y}"
- EXCLUDING_Y: "{topic} excluding {y}"
- NOT_ABOUT_Y: "{topic} not about {y}"
"""

from dataclasses import dataclass
from typing import Any


# Default negation templates
TEMPLATES = {
    "WITHOUT_Y": "{topic} without {y}",
    "EXCLUDING_Y": "{topic} excluding {y}",
    "NOT_ABOUT_Y": "{topic} not about {y}",
}


@dataclass
class GeneratedQuery:
    """
    A generated negated query with metadata.

    Attributes:
        base: Original query without negation.
        negated: Query with negation added.
        template: Template used (e.g., "WITHOUT_Y").
        y: The forbidden entity/term.
        y_surface_forms: All surface forms to check for Y.
    """
    base: str
    negated: str
    template: str
    y: str
    y_surface_forms: list[str]


def generate_negated_query(
    base_query: str,
    y: str,
    template: str = "WITHOUT_Y",
    y_surface_forms: list[str] | None = None
) -> GeneratedQuery:
    """
    Generate a negated query from a base query.

    Args:
        base_query: Original query text.
        y: Entity/term to exclude.
        template: Negation template to use.
        y_surface_forms: Alternative forms of Y to check in documents.
            If None, defaults to [y, y.lower(), y.upper(), y.capitalize()].

    Returns:
        GeneratedQuery with base, negated, and metadata.

    Example:
        >>> gq = generate_negated_query("python web scraping selenium", "selenium")
        >>> print(gq.negated)
        'python web scraping without selenium'
        >>> print(gq.y_surface_forms)
        ['selenium', 'Selenium', 'SELENIUM']

    Raises:
        ValueError: If template is not recognized.
    """
    # TODO: Implementation
    raise NotImplementedError("generate_negated_query not yet implemented")


def select_entity_for_negation(
    query: str,
    candidates: list[str] | None = None,
    method: str = "last_noun"
) -> str | None:
    """
    Select an entity from the query to negate.

    Args:
        query: Query text.
        candidates: Optional pre-extracted candidate entities.
        method: Selection method:
            - "last_noun": Select last noun/noun phrase
            - "random": Random selection from candidates
            - "salient": Select most salient entity (TF-IDF based)

    Returns:
        Selected entity, or None if no suitable candidate.

    Example:
        >>> y = select_entity_for_negation("python web scraping selenium")
        >>> print(y)
        'selenium'

    Note:
        "last_noun" often works well because queries tend to have
        modifiers at the end (e.g., "recipes without peanuts").
    """
    # TODO: Implementation
    raise NotImplementedError("select_entity_for_negation not yet implemented")


def batch_generate_queries(
    base_queries: list[tuple[str, str]],
    config: dict[str, Any]
) -> list[tuple[str, GeneratedQuery]]:
    """
    Generate negated queries in batch.

    Args:
        base_queries: List of (query_id, query_text) tuples.
        config: Configuration dict with:
            - query_templates: dict of template weights
            - Any other generation parameters

    Returns:
        List of (query_id, GeneratedQuery) tuples.

    Example:
        >>> queries = [("q1", "python scraping"), ("q2", "web automation")]
        >>> generated = batch_generate_queries(queries, config)
        >>> print(generated[0][1].negated)
        'python scraping without ...'
    """
    # TODO: Implementation
    raise NotImplementedError("batch_generate_queries not yet implemented")


def expand_surface_forms(y: str) -> list[str]:
    """
    Expand a term to its surface forms for matching.

    Args:
        y: Base term.

    Returns:
        List of surface forms including:
        - Original
        - Lowercase
        - Uppercase
        - Capitalized
        - Common variants (plural, etc.)

    Example:
        >>> forms = expand_surface_forms("selenium")
        >>> print(forms)
        ['selenium', 'Selenium', 'SELENIUM', 'seleniums']
    """
    # TODO: Implementation
    raise NotImplementedError("expand_surface_forms not yet implemented")


def is_valid_negation_target(y: str, query: str) -> bool:
    """
    Check if Y is a valid target for negation in the query.

    Args:
        y: Candidate entity to negate.
        query: Full query text.

    Returns:
        True if Y is a valid negation target.

    Validity criteria:
        - Y appears in the query
        - Y is not a stopword
        - Y is not the only content word
        - Y has reasonable length (2+ characters)

    Example:
        >>> is_valid_negation_target("selenium", "python web scraping selenium")
        True
        >>> is_valid_negation_target("the", "the best recipes")
        False
    """
    # TODO: Implementation
    raise NotImplementedError("is_valid_negation_target not yet implemented")
