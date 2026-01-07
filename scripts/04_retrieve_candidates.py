#!/usr/bin/env python3
"""
Script 04: Retrieve candidate documents using BM25.

For each query, retrieves top-k candidate documents from the index.
These candidates will be used to mine (doc_pos, doc_neg) pairs.

Usage:
    python scripts/04_retrieve_candidates.py --config configs/negation_v0.yaml
    python scripts/04_retrieve_candidates.py --k 200  # Top-k candidates
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve candidate documents using BM25"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/negation_v0.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="data/intermediate/negated_queries.jsonl",
        help="Input negated queries"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/intermediate/candidates.jsonl",
        help="Output path for candidates"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=200,
        help="Number of candidates per query"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of retrieval threads"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ConstraintSuite - Candidate Retrieval")
    print("=" * 60)

    # TODO: Implementation
    # 1. Load config and queries
    # 2. Initialize BM25Retriever
    # 3. For each query, retrieve k candidates
    # 4. Save candidates with query metadata

    print("\n[Not yet implemented]")
    print("This script will:")
    print(f"  1. Load queries from {args.queries}")
    print(f"  2. Retrieve top-{args.k} candidates per query")
    print(f"  3. Save to {args.output}")


if __name__ == "__main__":
    main()
