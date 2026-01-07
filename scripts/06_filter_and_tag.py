#!/usr/bin/env python3
"""
Script 06: Filter pairs and add difficulty tags.

Applies quality filters:
- Length filters (min/max doc length, length ratio)
- Similarity filters (topical overlap)
- Constraint validity checks

Adds tags for stratification:
- difficulty: easy/medium/hard
- lexical_overlap_bin: low/medium/high
- doc_length_bin: short/medium/long

Usage:
    python scripts/06_filter_and_tag.py --config configs/negation_v0.yaml
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Filter pairs and add difficulty tags"
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
        default="data/intermediate/raw_pairs.jsonl",
        help="Input mined pairs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/intermediate/filtered_pairs.jsonl",
        help="Output filtered and tagged pairs"
    )
    parser.add_argument(
        "--rejected",
        type=str,
        default="data/intermediate/rejected_pairs.jsonl",
        help="Output rejected pairs (for debugging)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ConstraintSuite - Filter and Tag")
    print("=" * 60)

    # TODO: Implementation
    # 1. Load raw pairs
    # 2. Apply filters (length, similarity, validity)
    # 3. Tag passing pairs (difficulty, overlap, length)
    # 4. Save filtered pairs and rejected pairs

    print("\n[Not yet implemented]")
    print("This script will:")
    print(f"  1. Load pairs from {args.input}")
    print("  2. Apply quality filters")
    print("  3. Add difficulty tags")
    print(f"  4. Save filtered to {args.output}")
    print(f"  5. Save rejected to {args.rejected}")


if __name__ == "__main__":
    main()
