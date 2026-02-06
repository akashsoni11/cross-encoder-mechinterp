"""
LLM-assisted utilities for ConstraintSuite using Codex CLI.

This module provides functions for:
- MinPairs grammar fixing (fix surgical edits that break grammar)
- Gold set validation (verify example quality)
- Surface form expansion (generate synonyms/variants)
- Ambiguous case classification

Uses Codex CLI with GPT 5.2 model for LLM calls.
Results are cached to avoid duplicate calls.
"""

import hashlib
import json
import logging
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("constraintsuite")

# Cache directory for LLM responses
CACHE_DIR = Path("data/cache/llm_responses")

# Codex agents directory
CODEX_AGENTS_DIR = Path("./codex-agents/dataset")

# Default model for Codex
DEFAULT_MODEL = "gpt-5.2"


def get_cache_path(prompt_hash: str) -> Path:
    """Get cache path for a given prompt hash."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{prompt_hash}.json"


def hash_prompt(prompt: str) -> str:
    """Hash a prompt for caching."""
    return hashlib.md5(prompt.encode()).hexdigest()


def run_codex_agent(
    prompt: str,
    agent_id: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
    use_cache: bool = True,
) -> str:
    """
    Run a Codex agent with the given prompt.

    Args:
        prompt: The prompt to send to Codex.
        agent_id: Optional agent ID (auto-generated if not provided).
        model: Model to use (default: gpt-5.2).
        timeout: Timeout in seconds.
        use_cache: Whether to use caching.

    Returns:
        The agent's response text.
    """
    # Check cache first
    if use_cache:
        prompt_hash = hash_prompt(prompt)
        cache_path = get_cache_path(prompt_hash)

        if cache_path.exists():
            with open(cache_path, "r") as f:
                cached = json.load(f)
                logger.debug(f"Cache hit for prompt {prompt_hash[:8]}")
                return cached["response"]

    # Ensure agents directory exists
    CODEX_AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate agent ID if not provided
    if agent_id is None:
        agent_id = f"llm-{uuid.uuid4().hex[:8]}"

    final_file = CODEX_AGENTS_DIR / f"{agent_id}-final.txt"
    log_file = CODEX_AGENTS_DIR / f"{agent_id}-log.txt"

    # Clean up any existing files
    final_file.unlink(missing_ok=True)
    log_file.unlink(missing_ok=True)

    # Build Codex command
    cmd = [
        "codex", "exec",
        "--full-auto",
        "-m", model,
        "--output-last-message", str(final_file),
        prompt,
    ]

    try:
        # Run Codex agent
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        # Wait for completion with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            if final_file.exists() and final_file.stat().st_size > 0:
                break
            if process.poll() is not None:
                # Process finished
                break
            time.sleep(1)

        # Read result
        if final_file.exists() and final_file.stat().st_size > 0:
            response = final_file.read_text().strip()
        else:
            # Check if process is still running
            if process.poll() is None:
                process.terminate()
                logger.warning(f"Agent {agent_id} timed out after {timeout}s")
            response = ""

        # Cache response
        if use_cache and response:
            prompt_hash = hash_prompt(prompt)
            cache_path = get_cache_path(prompt_hash)
            with open(cache_path, "w") as f:
                json.dump({
                    "prompt": prompt,
                    "response": response,
                    "agent_id": agent_id,
                }, f)

        return response

    except FileNotFoundError:
        logger.warning("Codex CLI not found. Install with: pip install codex-cli")
        return ""
    except Exception as e:
        logger.warning(f"Codex agent failed: {e}")
        return ""


def fix_minpair_grammar(
    original_text: str,
    edited_text: str,
    edit_type: str,
    y: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Fix grammar issues in surgically edited text using Codex.

    Args:
        original_text: Original document text.
        edited_text: Text after surgical edit.
        edit_type: Type of edit applied.
        y: The negated term.
        model: Model to use.

    Returns:
        Grammar-corrected text.
    """
    prompt = f"""You are a precise text editor. Fix grammatical issues in edited text.

INPUT:
Original: "{original_text}"
Edited: "{edited_text}"
Edit type: {edit_type}
Negated term: {y}

TASK:
1. Identify grammatical issues caused by the edit
2. Fix minimally while preserving the negation of '{y}'
3. The result must still clearly negate '{y}'

OUTPUT FORMAT (JSON):
{{
  "corrected_text": "<fixed text>",
  "changes_made": ["<change 1>", "<change 2>"],
  "negation_preserved": true
}}

Provide ONLY the JSON output, nothing else."""

    response = run_codex_agent(prompt, model=model)

    if not response:
        return edited_text

    try:
        # Parse JSON response
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        result = json.loads(response.strip())
        return result.get("corrected_text", edited_text)
    except json.JSONDecodeError:
        logger.warning("Could not parse grammar fix response")
        return edited_text


def validate_gold_example(
    example: dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Validate a gold set example using Codex.

    Args:
        example: Dataset example to validate.
        model: Model to use.

    Returns:
        Validation result with:
        - valid: bool
        - confidence: float
        - issues: list of issues found
        - reasoning: explanation
    """
    query = example.get("query", {})
    q_neg = query.get("neg", "") if isinstance(query, dict) else str(query)
    y = example.get("constraint", {}).get("y", example.get("y", ""))

    doc_pos = example.get("doc_pos", {})
    doc_neg = example.get("doc_neg", {})
    text_pos = doc_pos.get("text", "")[:500] if isinstance(doc_pos, dict) else str(doc_pos)[:500]
    text_neg = doc_neg.get("text", "")[:500] if isinstance(doc_neg, dict) else str(doc_neg)[:500]

    prompt = f"""You are evaluating a constraint-based IR dataset example.

EXAMPLE:
Query (with negation): {q_neg}
Forbidden term (Y): {y}

Document A (should SATISFY constraint - avoid/negate Y):
"{text_pos}"

Document B (should VIOLATE constraint - affirm Y):
"{text_neg}"

EVALUATE:
1. Does Document B clearly mention '{y}' affirmatively?
2. Does Document A either omit '{y}' or negate it?
3. Are both documents topically relevant to the query?
4. Is the constraint the main distinguishing factor?

OUTPUT FORMAT (JSON):
{{
    "valid": true/false,
    "confidence": 0.0-1.0,
    "doc_a_status": "omits_y" | "negates_y" | "affirms_y",
    "doc_b_status": "affirms_y" | "negates_y" | "omits_y",
    "issues": [],
    "reasoning": "brief explanation"
}}

Provide ONLY the JSON output."""

    response = run_codex_agent(prompt, model=model)

    default_result = {
        "valid": True,
        "confidence": 0.5,
        "issues": ["Could not validate"],
        "reasoning": "LLM call failed or returned empty",
    }

    if not response:
        return default_result

    try:
        # Parse JSON response
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        return json.loads(response.strip())
    except json.JSONDecodeError:
        return {
            "valid": True,
            "confidence": 0.5,
            "issues": ["Could not parse LLM response"],
            "reasoning": response[:200] if response else "Empty response",
        }


def expand_surface_forms_llm(
    y: str,
    context: str | None = None,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """
    Generate surface form variations using Codex.

    Args:
        y: Base term.
        context: Optional query/document context.
        model: Model to use.

    Returns:
        List of surface form variations.
    """
    context_str = f"\nCONTEXT: {context}" if context else ""

    prompt = f"""Generate surface form variations for entity matching.

TERM: {y}{context_str}

GENERATE:
- Case variations (lowercase, uppercase, title case)
- Plural/singular forms
- Common abbreviations
- Synonyms referring to the same concept
- Common misspellings (if any)

OUTPUT FORMAT (JSON):
{{
  "term": "{y}",
  "variations": ["variation1", "variation2", ...],
  "synonyms": ["synonym1", "synonym2", ...],
  "abbreviations": ["abbr1", ...]
}}

Provide ONLY the JSON output."""

    response = run_codex_agent(prompt, model=model)

    # Default fallback
    default_forms = [y, y.lower(), y.upper(), y.capitalize()]

    if not response:
        return default_forms

    try:
        # Parse JSON response
        if "```" in response:
            response = response.split("```")[1] if "```json" not in response else response.split("```json")[1]
            response = response.split("```")[0]

        result = json.loads(response.strip())

        # Combine all forms
        forms = set()
        forms.add(y)
        forms.update(result.get("variations", []))
        forms.update(result.get("synonyms", []))
        forms.update(result.get("abbreviations", []))

        return list(forms) if forms else default_forms

    except json.JSONDecodeError:
        return default_forms


def classify_ambiguous_pair(
    doc_pos: dict[str, Any],
    doc_neg: dict[str, Any],
    y: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Classify an ambiguous pair where automated classification is uncertain.

    Args:
        doc_pos: Positive document.
        doc_neg: Negative document.
        y: Forbidden term.
        model: Model to use.

    Returns:
        Classification result with slice_type and confidence.
    """
    text_pos = doc_pos.get("text", "")[:500]
    text_neg = doc_neg.get("text", "")[:500]

    prompt = f"""Classify this document pair for a negation constraint dataset.

Forbidden term: {y}

Document A:
"{text_pos}"

Document B:
"{text_neg}"

CLASSIFY into one of:
- "minpairs": Documents are near-identical except for negation edit
- "explicit": Doc A mentions the term but in negated context (e.g., 'no {y}', 'without {y}')
- "omission": Doc A doesn't mention the term at all

OUTPUT FORMAT (JSON):
{{
    "slice_type": "minpairs" | "explicit" | "omission",
    "confidence": 0.0-1.0,
    "doc_a_mentions_term": true/false,
    "doc_a_negates_term": true/false,
    "reasoning": "brief explanation"
}}

Provide ONLY the JSON output."""

    response = run_codex_agent(prompt, model=model)

    default_result = {
        "slice_type": "omission",
        "confidence": 0.5,
        "reasoning": "LLM not available",
    }

    if not response:
        return default_result

    try:
        if "```" in response:
            response = response.split("```")[1].split("```")[0]
            if response.startswith("json"):
                response = response[4:]
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return default_result


def batch_validate_gold_set(
    examples: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """
    Validate a batch of gold set examples.

    Args:
        examples: List of examples to validate.
        model: Model to use.

    Returns:
        List of validation results.
    """
    results = []
    for example in examples:
        result = validate_gold_example(example, model=model)
        result["example_id"] = example.get("id", example.get("query_id", ""))
        results.append(result)

    # Summary
    valid_count = sum(1 for r in results if r.get("valid", False))
    avg_confidence = sum(r.get("confidence", 0) for r in results) / len(results) if results else 0

    logger.info(
        f"Gold set validation: {valid_count}/{len(results)} valid "
        f"(avg confidence: {avg_confidence:.2f})"
    )

    return results


def batch_fix_grammar(
    pairs: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    batch_size: int = 20,
) -> list[dict[str, Any]]:
    """
    Fix grammar for a batch of MinPairs.

    Args:
        pairs: List of MinPair examples.
        model: Model to use.
        batch_size: Number of examples to process per Codex agent.

    Returns:
        List of pairs with corrected doc_pos text.
    """
    results = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]

        # Build batch prompt
        batch_data = []
        for p in batch:
            batch_data.append({
                "id": p.get("id", ""),
                "original": p.get("metadata", {}).get("original_text", ""),
                "edited": p.get("doc_pos", {}).get("text", ""),
                "y": p.get("constraint", {}).get("y", ""),
            })

        prompt = f"""Fix grammar issues in these MinPairs. Preserve negation semantics.

INPUT BATCH:
{json.dumps(batch_data, indent=2)}

TASK:
For each example, fix grammar while preserving negation of the 'y' term.

OUTPUT FORMAT (JSON array):
[
  {{"id": "...", "corrected_text": "...", "valid": true}},
  ...
]

Provide ONLY the JSON array."""

        response = run_codex_agent(
            prompt,
            agent_id=f"grammar-batch-{i // batch_size:03d}",
            model=model,
        )

        if response:
            try:
                if "```" in response:
                    response = response.split("```")[1].split("```")[0]
                    if response.startswith("json"):
                        response = response[4:]
                fixes = json.loads(response.strip())

                # Apply fixes
                fixes_by_id = {f["id"]: f for f in fixes}
                for p in batch:
                    pid = p.get("id", "")
                    if pid in fixes_by_id and fixes_by_id[pid].get("valid"):
                        p["doc_pos"]["text"] = fixes_by_id[pid]["corrected_text"]
                        p["grammar_fixed"] = True

            except json.JSONDecodeError:
                logger.warning(f"Could not parse batch {i // batch_size} response")

        results.extend(batch)

    fixed_count = sum(1 for p in results if p.get("grammar_fixed"))
    logger.info(f"Grammar fix: {fixed_count}/{len(results)} pairs updated")

    return results
