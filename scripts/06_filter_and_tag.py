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
import sys
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.utils import load_config, setup_logging, load_jsonl, save_jsonl, ensure_dir
from constraintsuite.filtering import batch_filter
from constraintsuite.tagging import batch_tag, compute_distribution_stats, format_stats_report


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
        default=None,
        help="Input mined pairs (default: from config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filtered and tagged pairs (default: from config)"
    )
    parser.add_argument(
        "--rejected",
        type=str,
        default=None,
        help="Output rejected pairs for debugging (default: from config)"
    )
    parser.add_argument(
        "--save-rejected",
        action="store_true",
        default=True,
        help="Save rejected pairs"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))

    # Set paths
    input_path = args.input or Path(config["paths"]["intermediate"]) / "raw_pairs.jsonl"
    output_path = args.output or Path(config["paths"]["intermediate"]) / "filtered_pairs.jsonl"
    rejected_path = args.rejected or Path(config["paths"]["intermediate"]) / "rejected_pairs.jsonl"
    ensure_dir(Path(output_path).parent)

    print("=" * 60)
    print("ConstraintSuite - Filter and Tag")
    print("=" * 60)

    # Load pairs
    print(f"\n[1/4] Loading pairs from {input_path}...")
    pairs = load_jsonl(input_path)
    print(f"Loaded {len(pairs)} pairs")

    # Filter pairs
    print("\n[2/4] Filtering pairs...")
    accepted, rejected = batch_filter(pairs, config, return_rejected=args.save_rejected)
    print(f"Accepted: {len(accepted)}, Rejected: {len(rejected) if rejected else 0}")

    # Tag accepted pairs
    print("\n[3/4] Tagging pairs with difficulty...")
    tagged = batch_tag(accepted, config)

    # Save results
    print(f"\n[4/4] Saving results...")
    save_jsonl(tagged, output_path)
    print(f"Saved {len(tagged)} filtered pairs to {output_path}")

    if args.save_rejected and rejected:
        save_jsonl(rejected, rejected_path)
        print(f"Saved {len(rejected)} rejected pairs to {rejected_path}")

    # Print statistics
    stats = compute_distribution_stats(tagged)
    print("\n" + "=" * 60)
    print(format_stats_report(stats))
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Filter and tag complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
