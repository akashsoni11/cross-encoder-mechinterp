#!/usr/bin/env python3
"""
Script 03: Generate negated queries.

Takes base queries and generates negated variants using templates:
- WITHOUT_Y: "{topic} without {y}"
- EXCLUDING_Y: "{topic} excluding {y}"
- NOT_ABOUT_Y: "{topic} not about {y}"

Usage:
    python scripts/03_generate_queries.py --config configs/negation_v0.yaml
    python scripts/03_generate_queries.py --limit 1000  # Limit queries
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate negated queries for ConstraintSuite"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/negation_v0.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/intermediate/negated_queries.jsonl",
        help="Output path for generated queries"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ConstraintSuite - Query Generation")
    print("=" * 60)

    # TODO: Implementation
    # 1. Load config
    # 2. Load base queries from MS MARCO
    # 3. For each query:
    #    a. Extract candidate entities
    #    b. Select entity to negate
    #    c. Generate negated query using template
    # 4. Save to output JSONL

    print("\n[Not yet implemented]")
    print("This script will:")
    print("  1. Load MS MARCO queries")
    print("  2. Extract entities from each query")
    print("  3. Generate negated variants using templates")
    print("  4. Save to:", args.output)


if __name__ == "__main__":
    main()
