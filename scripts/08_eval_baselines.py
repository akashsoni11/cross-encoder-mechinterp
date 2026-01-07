#!/usr/bin/env python3
"""
Script 08: Evaluate baseline rerankers.

Evaluates cross-encoder rerankers on the dataset:
- cross-encoder/ms-marco-MiniLM-L6-v2 (fast baseline)
- BAAI/bge-reranker-base (stronger)

Metrics:
- Pairwise accuracy
- Mean score gap
- Query sensitivity (if base queries available)
- Breakdown by slice and difficulty

Usage:
    python scripts/08_eval_baselines.py --config configs/negation_v0.yaml
    python scripts/08_eval_baselines.py --model cross-encoder/ms-marco-MiniLM-L6-v2
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline rerankers on ConstraintSuite"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/negation_v0.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/release/negation_v0/main.jsonl",
        help="Path to dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to evaluate (default: all from config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_eval.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ConstraintSuite - Baseline Evaluation")
    print("=" * 60)

    # TODO: Implementation
    # 1. Load dataset
    # 2. Initialize model(s)
    # 3. For each model:
    #    a. Score all pairs
    #    b. Compute metrics
    #    c. Breakdown by slice and difficulty
    # 4. Save results

    print("\n[Not yet implemented]")
    print("This script will:")
    print(f"  1. Load dataset from {args.dataset}")
    print("  2. Evaluate reranker(s):")
    if args.model:
        print(f"     - {args.model}")
    else:
        print("     - cross-encoder/ms-marco-MiniLM-L6-v2")
        print("     - BAAI/bge-reranker-base")
    print("  3. Compute metrics (accuracy, score gap)")
    print(f"  4. Save results to {args.output}")


if __name__ == "__main__":
    main()
