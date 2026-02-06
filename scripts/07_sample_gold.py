#!/usr/bin/env python3
"""
Script 07: Sample gold set and create release dataset.

Creates:
- Main dataset (target size from config)
- Gold set (50 examples with oversampled hard cases)

The gold set is used for:
- Manual verification
- Qualitative analysis
- Patching/ablation demos

Usage:
    python scripts/07_sample_gold.py --config configs/negation_v0.yaml
    python scripts/07_sample_gold.py --gold-size 50 --oversample-hard
"""

import argparse
import random
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.utils import load_config, setup_logging, load_jsonl, save_jsonl, set_seed, ensure_dir
from constraintsuite.tagging import (
    sample_gold_set,
    sample_by_slice,
    compute_distribution_stats,
    format_stats_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Sample gold set and create release dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/negation_v0.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input filtered pairs (default: from config)"
    )
    parser.add_argument(
        "--main-output",
        type=str,
        default=None,
        help="Output main dataset (default: from config)"
    )
    parser.add_argument(
        "--gold-output",
        type=str,
        default=None,
        help="Output gold set (default: from config)"
    )
    parser.add_argument(
        "--main-size",
        type=int,
        default=None,
        help="Main dataset size (default: from config)"
    )
    parser.add_argument(
        "--gold-size",
        type=int,
        default=None,
        help="Gold set size (default: from config)"
    )
    parser.add_argument(
        "--oversample-hard",
        action="store_true",
        default=True,
        help="Oversample hard examples in gold set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))

    # Set seed
    seed = args.seed or config.get("random_seed", 42)
    set_seed(seed)

    # Set paths
    input_path = args.input or Path(config["paths"]["intermediate"]) / "filtered_pairs.jsonl"
    release_dir = Path(config["paths"]["release"])
    main_output = args.main_output or release_dir / "main.jsonl"
    gold_output = args.gold_output or release_dir / "gold.jsonl"
    ensure_dir(release_dir)

    # Get sizes from config
    targets = config.get("targets", {})
    main_size = args.main_size or targets.get("main_set_size", 2000)
    gold_size = args.gold_size or targets.get("gold_set_size", 50)

    # Get slice distribution
    slice_distribution = config.get("slice_distribution", {
        "minpairs": 0.2,
        "explicit": 0.4,
        "omission": 0.4,
    })

    print("=" * 60)
    print("ConstraintSuite - Dataset Sampling")
    print("=" * 60)

    # Load pairs
    print(f"\n[1/4] Loading pairs from {input_path}...")
    pairs = load_jsonl(input_path)
    print(f"Loaded {len(pairs)} pairs")

    # Sample main dataset
    print(f"\n[2/4] Sampling main dataset ({main_size} examples)...")
    if len(pairs) <= main_size:
        main_dataset = pairs
        print(f"Using all {len(pairs)} pairs (less than target)")
    else:
        main_dataset = sample_by_slice(pairs, main_size, slice_distribution, seed=seed)
    print(f"Main dataset: {len(main_dataset)} examples")

    # Sample gold set from main dataset
    print(f"\n[3/4] Sampling gold set ({gold_size} examples)...")
    difficulty_distribution = config.get("gold_sampling", {}).get(
        "difficulty_distribution", [0.15, 0.35, 0.50]
    )
    gold_set = sample_gold_set(
        main_dataset,
        target_size=gold_size,
        difficulty_distribution=difficulty_distribution,
        oversample_hard=args.oversample_hard,
        seed=seed
    )
    print(f"Gold set: {len(gold_set)} examples")

    # Save datasets
    print(f"\n[4/4] Saving datasets...")
    save_jsonl(main_dataset, main_output)
    print(f"Saved main dataset to {main_output}")
    save_jsonl(gold_set, gold_output)
    print(f"Saved gold set to {gold_output}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Main Dataset Statistics")
    print("=" * 60)
    main_stats = compute_distribution_stats(main_dataset)
    print(format_stats_report(main_stats))

    print("\n" + "=" * 60)
    print("Gold Set Statistics")
    print("=" * 60)
    gold_stats = compute_distribution_stats(gold_set)
    print(format_stats_report(gold_stats))

    print("\n" + "=" * 60)
    print("Dataset sampling complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Main dataset: {main_output}")
    print(f"  Gold set: {gold_output}")


if __name__ == "__main__":
    main()
