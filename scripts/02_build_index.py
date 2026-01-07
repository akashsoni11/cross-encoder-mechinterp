#!/usr/bin/env python3
"""
Script 02: Build or verify BM25 index.

For MS MARCO, we use Pyserini's prebuilt index (no building needed).
This script verifies the index is accessible and working.

For custom corpora, this script can build a new index.

Usage:
    python scripts/02_build_index.py --config configs/negation_v0.yaml
    python scripts/02_build_index.py --verify-only
"""

import argparse
from pathlib import Path


def verify_prebuilt_index(index_name: str) -> bool:
    """
    Verify that a prebuilt Pyserini index is accessible.

    Args:
        index_name: Pyserini index name (e.g., "msmarco-v1-passage").

    Returns:
        True if index is accessible.
    """
    # TODO: Implementation
    # Hint:
    # from pyserini.search.lucene import LuceneSearcher
    # try:
    #     searcher = LuceneSearcher.from_prebuilt_index(index_name)
    #     hits = searcher.search("test query", k=1)
    #     return len(hits) > 0
    # except Exception:
    #     return False
    raise NotImplementedError("verify_prebuilt_index not yet implemented")


def build_custom_index(
    corpus_path: Path,
    index_path: Path,
    threads: int = 4
) -> None:
    """
    Build a custom BM25 index from a JSONL corpus.

    Args:
        corpus_path: Path to corpus JSONL file.
        index_path: Path to output index directory.
        threads: Number of indexing threads.
    """
    # TODO: Implementation
    raise NotImplementedError("build_custom_index not yet implemented")


def main():
    parser = argparse.ArgumentParser(
        description="Build or verify BM25 index"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/negation_v0.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify index accessibility"
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="msmarco-v1-passage",
        help="Pyserini prebuilt index name"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ConstraintSuite - Index Setup")
    print("=" * 60)

    print(f"\nVerifying index: {args.index_name}")
    if verify_prebuilt_index(args.index_name):
        print("✓ Index is accessible and working")
    else:
        print("✗ Index verification failed")
        exit(1)


if __name__ == "__main__":
    main()
