#!/usr/bin/env python3
"""
Script 07: Sample gold set for manual verification.

Samples 30-60 examples with appropriate difficulty distribution:
- Oversample hard examples (for interesting failure modes)
- Ensure coverage across slices

The gold set is used for:
- Manual verification
- Qualitative analysis
- Patching/ablation demos

Usage:
    python scripts/07_sample_gold.py --config configs/negation_v0.yaml
    python scripts/07_sample_gold.py --size 50 --oversample-hard
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Sample gold set for manual verification"
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
        default="data/intermediate/filtered_pairs.jsonl",
        help="Input filtered pairs"
    )
    parser.add_argument(
        "--main-output",
        type=str,
        default="data/release/negation_v0/main.jsonl",
        help="Output main dataset"
    )
    parser.add_argument(
        "--gold-output",
        type=str,
        default="data/release/negation_v0/gold.jsonl",
        help="Output gold set"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50,
        help="Gold set size"
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
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ConstraintSuite - Gold Set Sampling")
    print("=" * 60)

    # TODO: Implementation
    # 1. Load filtered pairs
    # 2. Stratify by difficulty
    # 3. Sample gold set with appropriate distribution
    # 4. Save main dataset and gold set separately

    print("\n[Not yet implemented]")
    print("This script will:")
    print(f"  1. Load pairs from {args.input}")
    print(f"  2. Sample {args.size} examples for gold set")
    print(f"  3. Save main dataset to {args.main_output}")
    print(f"  4. Save gold set to {args.gold_output}")


if __name__ == "__main__":
    main()
