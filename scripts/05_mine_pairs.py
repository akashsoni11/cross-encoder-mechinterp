#!/usr/bin/env python3
"""
Script 05: Mine document pairs from candidates.

For each query's candidate pool, mines (doc_pos, doc_neg) pairs where:
- doc_neg: contains the forbidden entity Y affirmatively
- doc_pos: either omits Y or mentions Y in negated form

Usage:
    python scripts/05_mine_pairs.py --config configs/negation_v0.yaml
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Mine document pairs from candidates"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/negation_v0.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="data/intermediate/candidates.jsonl",
        help="Input candidates"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/intermediate/raw_pairs.jsonl",
        help="Output path for mined pairs"
    )
    parser.add_argument(
        "--prefer-explicit",
        action="store_true",
        default=True,
        help="Prefer doc_pos that explicitly negates Y"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ConstraintSuite - Pair Mining")
    print("=" * 60)

    # TODO: Implementation
    # 1. Load candidates
    # 2. For each query's candidates:
    #    a. Find violators (contain Y affirmatively)
    #    b. Find satisfiers (omit Y or negate Y)
    #    c. Select best pair
    # 3. Classify into slices (minpairs, explicit, omission)
    # 4. Save mined pairs

    print("\n[Not yet implemented]")
    print("This script will:")
    print(f"  1. Load candidates from {args.candidates}")
    print("  2. Mine (doc_pos, doc_neg) pairs")
    print("  3. Classify into slices")
    print(f"  4. Save to {args.output}")


if __name__ == "__main__":
    main()
