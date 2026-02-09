"""
Reranker evaluation for ConstraintSuite.

This module provides functions for:
- Scoring query-document pairs with cross-encoder rerankers
- Computing evaluation metrics (pairwise accuracy, score gap)
- Baseline evaluation with multiple models

Primary models:
- cross-encoder/ms-marco-MiniLM-L6-v2 (fast baseline)
- BAAI/bge-reranker-base (stronger)

Metrics:
- Pairwise accuracy: % where score(doc_pos) > score(doc_neg)
- Score gap (Δ): score(doc_pos) - score(doc_neg)
- Query sensitivity (Δ_sens): Δ(q_neg) - Δ(q_base)
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

logger = logging.getLogger("constraintsuite")


@dataclass
class EvaluationResult:
    """
    Evaluation results for a model on a dataset.

    Attributes:
        model_name: Name/path of the evaluated model.
        pairwise_accuracy: % of pairs correctly ranked.
        mean_score_gap: Average score(doc_pos) - score(doc_neg).
        score_gap_std: Standard deviation of score gaps.
        mean_query_sensitivity: Average Δ_sens (if base queries available).
        per_example_results: Detailed results per example.
        slice_results: Results broken down by slice type.
        difficulty_results: Results broken down by difficulty.
    """

    model_name: str
    pairwise_accuracy: float
    mean_score_gap: float
    score_gap_std: float
    mean_query_sensitivity: float | None
    per_example_results: list[dict[str, Any]] = field(default_factory=list)
    slice_results: dict[str, dict[str, float]] = field(default_factory=dict)
    difficulty_results: dict[str, dict[str, float]] = field(default_factory=dict)


class CrossEncoderScorer:
    """
    Wrapper for cross-encoder reranker scoring.

    Uses sentence-transformers CrossEncoder for inference.

    Example:
        >>> scorer = CrossEncoderScorer("cross-encoder/ms-marco-MiniLM-L6-v2")
        >>> score = scorer.score("python without selenium", "BeautifulSoup tutorial")
        >>> print(f"Score: {score:.4f}")
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        cpu_threads: int | None = None,
    ):
        """
        Initialize cross-encoder scorer.

        Args:
            model_name: HuggingFace model name or path.
            device: Device to use ("auto", "cuda", "cpu", "mps").
            batch_size: Batch size for inference.

        Example:
            >>> scorer = CrossEncoderScorer("cross-encoder/ms-marco-MiniLM-L6-v2")
        """
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size

        # Determine device
        import torch

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if device == "cpu":
            if cpu_threads is None:
                detected = os.cpu_count() or 1
                cpu_threads = max(1, detected - 1)
            torch.set_num_threads(cpu_threads)
            if hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(max(1, cpu_threads // 2))
                except RuntimeError:
                    # PyTorch only allows this before parallel work starts in a process.
                    logger.debug(
                        "Could not reset PyTorch interop threads; keeping existing setting"
                    )
            logger.info(f"Configured PyTorch CPU threads: {cpu_threads}")
        elif device == "mps":
            # Prefer higher-throughput kernels on Apple Silicon.
            torch.set_float32_matmul_precision("high")

        logger.info(f"Loading CrossEncoder: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)
        self.device = device

    def score(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.

        Args:
            query: Query text.
            document: Document text.

        Returns:
            Relevance score (higher = more relevant).

        Example:
            >>> score = scorer.score("python scraping", "BeautifulSoup tutorial")
            >>> print(f"Score: {score:.4f}")
        """
        scores = self.model.predict([(query, document)])
        return float(scores[0])

    def score_pairs(
        self,
        pairs: list[tuple[str, str]],
        show_progress_bar: bool | None = None,
    ) -> list[float]:
        """
        Score multiple query-document pairs.

        Args:
            pairs: List of (query, document) tuples.

        Returns:
            List of relevance scores.

        Example:
            >>> pairs = [("query1", "doc1"), ("query1", "doc2")]
            >>> scores = scorer.score_pairs(pairs)
            >>> print(f"Scores: {scores}")
        """
        if not pairs:
            return []

        if show_progress_bar is None:
            show_progress_bar = len(pairs) > 100

        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=show_progress_bar
        )
        return [float(s) for s in scores]


def _extract_example_fields(example: dict[str, Any]) -> dict[str, Any]:
    """Normalize an example into fields used by evaluation."""
    query_info = example.get("query", {})
    if isinstance(query_info, dict):
        q_neg = query_info.get("neg", query_info.get("negated", ""))
        q_base = query_info.get("base", "")
    else:
        q_neg = str(query_info)
        q_base = ""

    doc_pos = example.get("doc_pos", example.get("docs", {}).get("pos", {}))
    doc_neg = example.get("doc_neg", example.get("docs", {}).get("neg", {}))

    text_pos = doc_pos.get("text", "") if isinstance(doc_pos, dict) else str(doc_pos)
    text_neg = doc_neg.get("text", "") if isinstance(doc_neg, dict) else str(doc_neg)

    return {
        "q_neg": q_neg,
        "q_base": q_base,
        "text_pos": text_pos,
        "text_neg": text_neg,
        "id": example.get("id", example.get("query_id", "")),
        "slice_type": example.get("slice_type", "unknown"),
        "difficulty": example.get("tags", {}).get("difficulty", "unknown"),
    }


def evaluate_example(
    scorer: CrossEncoderScorer, example: dict[str, Any], include_base_query: bool = True
) -> dict[str, Any]:
    """
    Evaluate a single example.

    Args:
        scorer: CrossEncoderScorer instance.
        example: Dataset example with query and docs.
        include_base_query: Whether to also score with base query.

    Returns:
        Dictionary with:
        - score_pos: Score for (q_neg, doc_pos)
        - score_neg: Score for (q_neg, doc_neg)
        - score_gap: score_pos - score_neg
        - correct: Whether doc_pos scored higher
        - score_pos_base: Score for (q_base, doc_pos) [if included]
        - score_neg_base: Score for (q_base, doc_neg) [if included]
        - query_sensitivity: Δ(q_neg) - Δ(q_base) [if included]

    Example:
        >>> result = evaluate_example(scorer, example)
        >>> print(f"Correct: {result['correct']}")
        >>> print(f"Score gap: {result['score_gap']:.4f}")
    """
    data = _extract_example_fields(example)
    q_neg = data["q_neg"]
    q_base = data["q_base"]
    text_pos = data["text_pos"]
    text_neg = data["text_neg"]

    # Score with negated query
    neg_scores = scorer.score_pairs([(q_neg, text_pos), (q_neg, text_neg)], show_progress_bar=False)
    score_pos = neg_scores[0]
    score_neg = neg_scores[1]
    score_gap = score_pos - score_neg
    correct = score_pos > score_neg

    result = {
        "score_pos": score_pos,
        "score_neg": score_neg,
        "score_gap": score_gap,
        "correct": correct,
    }

    # Optionally score with base query
    if include_base_query and q_base:
        base_scores = scorer.score_pairs(
            [(q_base, text_pos), (q_base, text_neg)], show_progress_bar=False
        )
        score_pos_base = base_scores[0]
        score_neg_base = base_scores[1]
        score_gap_base = score_pos_base - score_neg_base

        # Query sensitivity: how much does adding negation change the gap?
        query_sensitivity = score_gap - score_gap_base

        result.update(
            {
                "score_pos_base": score_pos_base,
                "score_neg_base": score_neg_base,
                "score_gap_base": score_gap_base,
                "query_sensitivity": query_sensitivity,
            }
        )

    return result


def evaluate_dataset(
    model_name: str,
    examples: list[dict[str, Any]],
    device: str = "auto",
    batch_size: int = 32,
    cpu_threads: int | None = None,
    include_base_query: bool = True,
) -> EvaluationResult:
    """
    Evaluate a model on the full dataset.

    Args:
        model_name: HuggingFace model name.
        examples: List of dataset examples.
        device: Device for inference.
        batch_size: Batch size.
        cpu_threads: CPU threads for PyTorch CPU execution.
        include_base_query: Whether to compute query sensitivity.

    Returns:
        EvaluationResult with all metrics.

    Example:
        >>> results = evaluate_dataset(
        ...     "cross-encoder/ms-marco-MiniLM-L6-v2",
        ...     examples
        ... )
        >>> print(f"Accuracy: {results.pairwise_accuracy:.2%}")
        >>> print(f"Mean Δ: {results.mean_score_gap:.4f}")
    """
    scorer = CrossEncoderScorer(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        cpu_threads=cpu_threads,
    )

    # Prepare all pairs once, then score in large batches to maximize accelerator throughput.
    extracted = [_extract_example_fields(example) for example in examples]

    neg_pairs: list[tuple[str, str]] = []
    for item in extracted:
        neg_pairs.append((item["q_neg"], item["text_pos"]))
        neg_pairs.append((item["q_neg"], item["text_neg"]))
    neg_scores = scorer.score_pairs(neg_pairs, show_progress_bar=len(neg_pairs) > 1000)

    base_scores: list[float] = []
    base_offsets: dict[int, int] = {}
    if include_base_query:
        base_pairs: list[tuple[str, str]] = []
        for idx, item in enumerate(extracted):
            if item["q_base"]:
                base_offsets[idx] = len(base_pairs)
                base_pairs.append((item["q_base"], item["text_pos"]))
                base_pairs.append((item["q_base"], item["text_neg"]))
        if base_pairs:
            base_scores = scorer.score_pairs(base_pairs, show_progress_bar=len(base_pairs) > 1000)

    per_example_results = []
    for idx, item in enumerate(tqdm(extracted, desc="Assembling metrics")):
        neg_offset = idx * 2
        score_pos = neg_scores[neg_offset]
        score_neg = neg_scores[neg_offset + 1]
        score_gap = score_pos - score_neg
        correct = score_pos > score_neg

        result = {
            "id": item["id"],
            "slice_type": item["slice_type"],
            "difficulty": item["difficulty"],
            "score_pos": score_pos,
            "score_neg": score_neg,
            "score_gap": score_gap,
            "correct": correct,
        }

        base_offset = base_offsets.get(idx)
        if include_base_query and base_offset is not None:
            score_pos_base = base_scores[base_offset]
            score_neg_base = base_scores[base_offset + 1]
            score_gap_base = score_pos_base - score_neg_base
            result.update(
                {
                    "score_pos_base": score_pos_base,
                    "score_neg_base": score_neg_base,
                    "score_gap_base": score_gap_base,
                    "query_sensitivity": score_gap - score_gap_base,
                }
            )

        per_example_results.append(result)

    # Compute aggregate metrics
    correct_count = sum(1 for r in per_example_results if r["correct"])
    pairwise_accuracy = correct_count / len(per_example_results) if per_example_results else 0.0

    score_gaps = [r["score_gap"] for r in per_example_results]
    mean_score_gap = np.mean(score_gaps) if score_gaps else 0.0
    score_gap_std = np.std(score_gaps) if score_gaps else 0.0

    # Query sensitivity
    sensitivities = [
        r.get("query_sensitivity")
        for r in per_example_results
        if r.get("query_sensitivity") is not None
    ]
    mean_query_sensitivity = np.mean(sensitivities) if sensitivities else None

    # Compute slice results
    slice_results = evaluate_by_slice_from_results(per_example_results)
    difficulty_results = evaluate_by_difficulty_from_results(per_example_results)

    logger.info(
        f"Evaluation complete: {model_name}\n"
        f"  Pairwise Accuracy: {pairwise_accuracy:.2%}\n"
        f"  Mean Score Gap: {mean_score_gap:.4f} (std: {score_gap_std:.4f})\n"
        f"  Query Sensitivity: {f'{mean_query_sensitivity:.4f}' if mean_query_sensitivity is not None else 'N/A'}"
    )

    return EvaluationResult(
        model_name=model_name,
        pairwise_accuracy=pairwise_accuracy,
        mean_score_gap=mean_score_gap,
        score_gap_std=score_gap_std,
        mean_query_sensitivity=mean_query_sensitivity,
        per_example_results=per_example_results,
        slice_results=slice_results,
        difficulty_results=difficulty_results,
    )


def evaluate_by_slice_from_results(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by slice type."""
    slice_results = defaultdict(lambda: {"correct": 0, "total": 0, "gaps": []})

    for r in results:
        slice_type = r.get("slice_type", "unknown")
        slice_results[slice_type]["total"] += 1
        if r["correct"]:
            slice_results[slice_type]["correct"] += 1
        slice_results[slice_type]["gaps"].append(r["score_gap"])

    output = {}
    for slice_type, data in slice_results.items():
        output[slice_type] = {
            "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0.0,
            "mean_gap": np.mean(data["gaps"]) if data["gaps"] else 0.0,
            "count": data["total"],
        }

    return output


def evaluate_by_difficulty_from_results(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by difficulty."""
    diff_results = defaultdict(lambda: {"correct": 0, "total": 0, "gaps": []})

    for r in results:
        difficulty = r.get("difficulty", "unknown")
        diff_results[difficulty]["total"] += 1
        if r["correct"]:
            diff_results[difficulty]["correct"] += 1
        diff_results[difficulty]["gaps"].append(r["score_gap"])

    output = {}
    for difficulty, data in diff_results.items():
        output[difficulty] = {
            "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0.0,
            "mean_gap": np.mean(data["gaps"]) if data["gaps"] else 0.0,
            "count": data["total"],
        }

    return output


def evaluate_by_slice(results: EvaluationResult) -> dict[str, dict[str, float]]:
    """
    Break down results by slice type.

    Args:
        results: EvaluationResult with per_example_results.

    Returns:
        Dictionary mapping slice -> metrics.

    Example:
        >>> slice_results = evaluate_by_slice(results)
        >>> print(f"MinPairs accuracy: {slice_results['minpairs']['accuracy']:.2%}")
        >>> print(f"Omission accuracy: {slice_results['omission']['accuracy']:.2%}")
    """
    return evaluate_by_slice_from_results(results.per_example_results)


def evaluate_by_difficulty(results: EvaluationResult) -> dict[str, dict[str, float]]:
    """
    Break down results by difficulty.

    Args:
        results: EvaluationResult with per_example_results.

    Returns:
        Dictionary mapping difficulty -> metrics.

    Example:
        >>> diff_results = evaluate_by_difficulty(results)
        >>> print(f"Easy accuracy: {diff_results['easy']['accuracy']:.2%}")
        >>> print(f"Hard accuracy: {diff_results['hard']['accuracy']:.2%}")
    """
    return evaluate_by_difficulty_from_results(results.per_example_results)


def compare_models(
    model_names: list[str], examples: list[dict[str, Any]], **kwargs
) -> dict[str, EvaluationResult]:
    """
    Compare multiple models on the same dataset.

    Args:
        model_names: List of model names to evaluate.
        examples: Dataset examples.
        **kwargs: Additional arguments for evaluate_dataset.

    Returns:
        Dictionary mapping model_name -> EvaluationResult.

    Example:
        >>> models = [
        ...     "cross-encoder/ms-marco-MiniLM-L6-v2",
        ...     "BAAI/bge-reranker-base"
        ... ]
        >>> comparison = compare_models(models, examples)
        >>> for name, result in comparison.items():
        ...     print(f"{name}: {result.pairwise_accuracy:.2%}")
    """
    results = {}

    for model_name in model_names:
        logger.info(f"Evaluating model: {model_name}")
        results[model_name] = evaluate_dataset(model_name, examples, **kwargs)

    # Print comparison summary
    logger.info("\n" + "=" * 60)
    logger.info("Model Comparison Summary")
    logger.info("=" * 60)
    for name, result in results.items():
        logger.info(f"{name}:")
        logger.info(f"  Accuracy: {result.pairwise_accuracy:.2%}")
        logger.info(f"  Mean Gap: {result.mean_score_gap:.4f}")

    return results


def export_results(results: EvaluationResult, output_path: str) -> None:
    """
    Export evaluation results to JSON.

    Args:
        results: EvaluationResult to export.
        output_path: Path to output JSON file.

    Example:
        >>> export_results(results, "results/minilm_eval.json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = {
        "model_name": results.model_name,
        "pairwise_accuracy": results.pairwise_accuracy,
        "mean_score_gap": results.mean_score_gap,
        "score_gap_std": results.score_gap_std,
        "mean_query_sensitivity": results.mean_query_sensitivity,
        "slice_results": results.slice_results,
        "difficulty_results": results.difficulty_results,
        "per_example_results": results.per_example_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results exported to {output_path}")


def format_results_report(results: EvaluationResult) -> str:
    """Format evaluation results as a readable report."""
    lines = [
        f"Evaluation Results: {results.model_name}",
        "=" * 60,
        "",
        "Overall Metrics:",
        f"  Pairwise Accuracy: {results.pairwise_accuracy:.2%}",
        f"  Mean Score Gap: {results.mean_score_gap:.4f} (std: {results.score_gap_std:.4f})",
    ]

    if results.mean_query_sensitivity is not None:
        lines.append(f"  Query Sensitivity: {results.mean_query_sensitivity:.4f}")

    lines.extend(["", "By Slice Type:"])
    for slice_type in ["minpairs", "explicit", "omission"]:
        if slice_type in results.slice_results:
            data = results.slice_results[slice_type]
            lines.append(
                f"  {slice_type}: {data['accuracy']:.2%} "
                f"(n={data['count']}, gap={data['mean_gap']:.4f})"
            )

    lines.extend(["", "By Difficulty:"])
    for difficulty in ["easy", "medium", "hard"]:
        if difficulty in results.difficulty_results:
            data = results.difficulty_results[difficulty]
            lines.append(
                f"  {difficulty}: {data['accuracy']:.2%} "
                f"(n={data['count']}, gap={data['mean_gap']:.4f})"
            )

    return "\n".join(lines)
