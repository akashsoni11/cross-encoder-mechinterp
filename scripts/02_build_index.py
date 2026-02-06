#!/usr/bin/env python3
"""
Script 02: Verify BM25 index.

Verifies that the prebuilt MS MARCO index is accessible and working.

Usage:
    python scripts/02_build_index.py --config configs/negation_v0.yaml
    python scripts/02_build_index.py --verify-only
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.utils import load_config, setup_logging
from constraintsuite.retrieval import BM25Retriever, verify_index


def main():
    parser = argparse.ArgumentParser(
        description="Verify BM25 index for ConstraintSuite"
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
        help="Only verify the index (default behavior)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))

    print("=" * 60)
    print("ConstraintSuite - Index Verification")
    print("=" * 60)

    # Get index name from config
    index_name = config.get("retrieval", {}).get("index_name", "msmarco-v1-passage")

    print(f"\nVerifying index: {index_name}")

    # Verify index
    if verify_index(index_name):
        print("\n[OK] Index verification passed!")

        # Run a test query
        print("\nRunning test query...")
        retriever = BM25Retriever(index_name)
        results = retriever.retrieve("python web scraping", k=5)

        print(f"Top 5 results for 'python web scraping':")
        for i, doc in enumerate(results):
            text_preview = doc["text"][:100].replace("\n", " ")
            print(f"  {i+1}. {doc['doc_id']}: {text_preview}...")

    else:
        print("\n[FAIL] Index verification failed!")
        print("Please run: python scripts/01_download_data.py --index-only")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Index verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
