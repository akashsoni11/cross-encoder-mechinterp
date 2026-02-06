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
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.utils import load_config, setup_logging, load_jsonl, ensure_dir
from constraintsuite.evaluation import (
    evaluate_dataset,
    compare_models,
    export_results,
    format_results_report,
)


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
        default=None,
        help="Path to dataset (default: from config)"
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
        help="Device for inference (auto, cuda, cpu, mps)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (default: from config)"
    )
    parser.add_argument(
        "--include-base-query",
        action="store_true",
        default=True,
        help="Include base query scoring for sensitivity analysis"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))

    # Set paths
    dataset_path = args.dataset or Path(config["paths"]["release"]) / "main.jsonl"
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    # Get model config
    model_config = config.get("models", {})
    batch_size = args.batch_size or model_config.get("batch_size", 32)

    # Determine models to evaluate
    if args.model:
        models = [args.model]
    else:
        models = [
            model_config.get("baseline_reranker", "cross-encoder/ms-marco-MiniLM-L6-v2"),
            model_config.get("stronger_reranker", "BAAI/bge-reranker-base"),
        ]

    print("=" * 60)
    print("ConstraintSuite - Baseline Evaluation")
    print("=" * 60)

    # Load dataset
    print(f"\n[1/3] Loading dataset from {dataset_path}...")
    examples = load_jsonl(dataset_path)
    print(f"Loaded {len(examples)} examples")

    # Evaluate models
    print(f"\n[2/3] Evaluating {len(models)} model(s)...")
    all_results = {}

    for model_name in models:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_name}")
        print("=" * 60)

        result = evaluate_dataset(
            model_name=model_name,
            examples=examples,
            device=args.device,
            batch_size=batch_size,
            include_base_query=args.include_base_query
        )

        all_results[model_name] = result

        # Print results
        print("\n" + format_results_report(result))

    # Save results
    print(f"\n[3/3] Saving results to {output_path}...")

    # Export each model's results
    for model_name, result in all_results.items():
        # Sanitize model name for filename
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        model_output = output_path.parent / f"{output_path.stem}_{safe_name}.json"
        export_results(result, str(model_output))

    # Save summary
    summary = {
        "models": list(all_results.keys()),
        "dataset": str(dataset_path),
        "num_examples": len(examples),
        "results": {
            name: {
                "pairwise_accuracy": r.pairwise_accuracy,
                "mean_score_gap": r.mean_score_gap,
                "score_gap_std": r.score_gap_std,
                "mean_query_sensitivity": r.mean_query_sensitivity,
                "slice_results": r.slice_results,
                "difficulty_results": r.difficulty_results,
            }
            for name, r in all_results.items()
        }
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {output_path}")

    # Print comparison
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"\nDataset: {len(examples)} examples")
    print("\nModel Comparison:")
    print("-" * 60)
    print(f"{'Model':<45} {'Accuracy':>10}")
    print("-" * 60)
    for name, result in all_results.items():
        short_name = name.split("/")[-1] if "/" in name else name
        print(f"{short_name:<45} {result.pairwise_accuracy:>10.2%}")
    print("-" * 60)

    print("\n" + "=" * 60)
    print("Baseline evaluation complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
