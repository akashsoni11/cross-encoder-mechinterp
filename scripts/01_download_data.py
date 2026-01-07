#!/usr/bin/env python3
"""
Script 01: Download data and indexes.

Downloads:
- MS MARCO passage corpus (via ir_datasets)
- Pyserini prebuilt BM25 index
- Optionally: BEIR datasets

Usage:
    python scripts/01_download_data.py --config configs/negation_v0.yaml
    python scripts/01_download_data.py --index-only  # Just download index
"""

import argparse
from pathlib import Path


def download_msmarco_index(output_dir: Path) -> None:
    """
    Download Pyserini prebuilt MS MARCO passage index.

    The index is ~2GB and will be cached by Pyserini.

    Args:
        output_dir: Directory for any local caching.
    """
    # TODO: Implementation
    # Hint:
    # from pyserini.search.lucene import LuceneSearcher
    # searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
    # This triggers download if not cached
    raise NotImplementedError("download_msmarco_index not yet implemented")


def download_msmarco_queries(output_dir: Path, split: str = "train") -> None:
    """
    Download MS MARCO queries.

    Args:
        output_dir: Directory to save queries.
        split: Dataset split (train, dev, eval).
    """
    # TODO: Implementation
    # Hint:
    # import ir_datasets
    # dataset = ir_datasets.load(f"msmarco-passage/{split}")
    raise NotImplementedError("download_msmarco_queries not yet implemented")


def download_beir_dataset(dataset_name: str, output_dir: Path) -> None:
    """
    Download a BEIR dataset.

    Args:
        dataset_name: BEIR dataset name (e.g., "scifact").
        output_dir: Directory to save dataset.
    """
    # TODO: Implementation
    # Hint:
    # from beir import util
    # url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    # util.download_and_unzip(url, output_dir)
    raise NotImplementedError("download_beir_dataset not yet implemented")


def main():
    parser = argparse.ArgumentParser(
        description="Download data and indexes for ConstraintSuite"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/negation_v0.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only download the BM25 index"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for downloaded data"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ConstraintSuite - Data Download")
    print("=" * 60)

    # Download index
    print("\n[1/3] Downloading MS MARCO BM25 index...")
    download_msmarco_index(output_dir)

    if not args.index_only:
        # Download queries
        print("\n[2/3] Downloading MS MARCO queries...")
        download_msmarco_queries(output_dir)

        # Optionally download BEIR
        print("\n[3/3] BEIR datasets (optional, skipping for now)")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
