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

import logging
import random
import re
from dataclasses import dataclass
from typing import Any

from constraintsuite.data_loading import get_query_entities, get_last_noun, STOPWORDS

logger = logging.getLogger("constraintsuite")


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
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template: {template}. Available: {list(TEMPLATES.keys())}")

    # Get template pattern
    pattern = TEMPLATES[template]

    # Create topic by removing Y from base query
    # Use case-insensitive replacement
    topic = re.sub(rf"\b{re.escape(y)}\b", "", base_query, flags=re.IGNORECASE)
    topic = " ".join(topic.split())  # Normalize whitespace

    # Generate negated query
    negated = pattern.format(topic=topic, y=y)

    # Generate surface forms
    if y_surface_forms is None:
        y_surface_forms = expand_surface_forms(y)

    return GeneratedQuery(
        base=base_query,
        negated=negated,
        template=template,
        y=y,
        y_surface_forms=y_surface_forms
    )


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
    if method == "last_noun":
        return get_last_noun(query)

    # Get candidates if not provided
    if candidates is None:
        candidates = get_query_entities(query)

    if not candidates:
        return None

    if method == "random":
        return random.choice(candidates)
    elif method == "salient":
        # Simple heuristic: prefer longer terms (often more specific)
        # In practice, you'd use TF-IDF or other saliency measures
        sorted_candidates = sorted(candidates, key=len, reverse=True)
        return sorted_candidates[0]
    else:
        raise ValueError(f"Unknown method: {method}")


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
    # Get template weights from config
    template_config = config.get("query_templates", {})
    templates = []
    weights = []

    for name, template_info in template_config.items():
        if isinstance(template_info, dict):
            templates.append(name)
            weights.append(template_info.get("weight", 1.0))
        else:
            # Simple weight value
            templates.append(name)
            weights.append(template_info)

    # Normalize weights
    if not templates:
        templates = list(TEMPLATES.keys())
        weights = [1.0 / len(templates)] * len(templates)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    # Get entity selection method from config
    entity_method = config.get("entity_selection_method", "last_noun")

    # Pre-extract entities in batch using spaCy pipe for speed
    from tqdm import tqdm
    from constraintsuite.data_loading import batch_get_last_nouns

    logger.info(f"Extracting entities from {len(base_queries)} queries (batched spaCy)...")
    query_texts = [text for _, text in base_queries]
    last_nouns = batch_get_last_nouns(query_texts)

    # Generate queries
    results = []
    skipped_no_entity = 0
    skipped_invalid = 0
    skipped_error = 0

    for i, (query_id, query_text) in enumerate(tqdm(base_queries, desc="Generating queries")):
        y = last_nouns[i] if entity_method == "last_noun" else select_entity_for_negation(query_text, method=entity_method)

        if y is None:
            skipped_no_entity += 1
            continue
        if not is_valid_negation_target(y, query_text):
            skipped_invalid += 1
            continue

        # Select template (weighted random)
        template = random.choices(templates, weights=weights)[0]

        # Generate negated query
        try:
            gen_query = generate_negated_query(query_text, y, template)
            results.append((query_id, gen_query))
        except ValueError as e:
            skipped_error += 1
            logger.debug(f"Skipping query {query_id}: {e}")
            continue

    logger.info(
        f"Generated {len(results)} negated queries from {len(base_queries)} base queries "
        f"(no_entity={skipped_no_entity}, invalid={skipped_invalid}, error={skipped_error})"
    )
    return results


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
    forms = set()

    # Basic case variants
    forms.add(y)
    forms.add(y.lower())
    forms.add(y.upper())
    forms.add(y.capitalize())
    forms.add(y.title())

    # Simple plural heuristics
    y_lower = y.lower()
    if not y_lower.endswith("s"):
        forms.add(y_lower + "s")
        forms.add(y.capitalize() + "s")
    if y_lower.endswith("y") and len(y_lower) > 1 and y_lower[-2] not in "aeiou":
        # e.g., "query" -> "queries"
        forms.add(y_lower[:-1] + "ies")
    if y_lower.endswith(("s", "x", "z", "ch", "sh")):
        forms.add(y_lower + "es")

    # Remove the original if it got duplicated
    forms_list = list(forms)

    # Put original first, then sort rest
    result = [y]
    for f in sorted(forms_list):
        if f != y and f not in result:
            result.append(f)

    return result


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
    y_lower = y.lower()

    # Check length
    if len(y) < 2:
        return False

    # Check if stopword
    if y_lower in STOPWORDS:
        return False

    # Check if Y appears in query (case-insensitive)
    if y_lower not in query.lower():
        return False

    # Check if Y is the only content word
    query_words = [
        w.lower() for w in query.split()
        if w.lower() not in STOPWORDS and len(w) >= 2
    ]
    if len(query_words) <= 1:
        return False

    # Check if Y is too long (likely a phrase that needs special handling)
    if len(y.split()) > 3:
        return False

    return True


def remove_entity_from_query(query: str, y: str) -> str:
    """
    Remove an entity from a query while preserving grammar.

    Args:
        query: Original query.
        y: Entity to remove.

    Returns:
        Query with entity removed and whitespace normalized.
    """
    # Case-insensitive removal
    result = re.sub(rf"\b{re.escape(y)}\b", "", query, flags=re.IGNORECASE)

    # Clean up whitespace
    result = " ".join(result.split())

    return result


def generate_adversarial_queries(
    base_queries: list[tuple[str, str]],
    adversarial_phrases: list[str],
    limit: int | None = None
) -> list[tuple[str, dict]]:
    """
    Generate adversarial control queries with phrases that look like negation
    but are not (e.g., "without further ado").

    Args:
        base_queries: List of (query_id, query_text) tuples.
        adversarial_phrases: Phrases to append (e.g., "without doubt").
        limit: Maximum number of queries to generate.

    Returns:
        List of (query_id, metadata) tuples for adversarial examples.
    """
    results = []

    for query_id, query_text in base_queries:
        if limit is not None and len(results) >= limit:
            break

        # Pick a random adversarial phrase
        phrase = random.choice(adversarial_phrases)

        # Append to query
        adversarial_query = f"{query_text} {phrase}"

        results.append((
            f"adv_{query_id}",
            {
                "base_query": query_text,
                "adversarial_query": adversarial_query,
                "phrase": phrase,
                "type": "adversarial_control"
            }
        ))

    return results
