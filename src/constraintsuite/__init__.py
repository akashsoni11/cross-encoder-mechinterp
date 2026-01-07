"""
ConstraintSuite: Dataset for evaluating constraint-sensitive relevance in cross-encoder rerankers.

This package provides tools for:
- Generating constraint-based query-document pairs
- Mining positive/negative document pairs
- Evaluating reranker performance on constraint handling
- Supporting mechanistic interpretability research

Example usage:
    from constraintsuite import load_config, generate_dataset, evaluate_reranker

    config = load_config("configs/negation_v0.yaml")
    dataset = generate_dataset(config)
    results = evaluate_reranker("cross-encoder/ms-marco-MiniLM-L6-v2", dataset)
"""

__version__ = "0.1.0"
__author__ = "Tanuj Sharma"

from constraintsuite.utils import load_config, save_jsonl, load_jsonl

__all__ = [
    "__version__",
    "load_config",
    "save_jsonl",
    "load_jsonl",
]
