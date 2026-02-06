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
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.utils import load_config, setup_logging, ensure_dir


def download_msmarco_index(output_dir: Path) -> None:
    """
    Download Pyserini prebuilt MS MARCO passage index.

    The index is ~2GB and will be cached by Pyserini.

    Args:
        output_dir: Directory for any local caching.
    """
    from pyserini.search.lucene import LuceneSearcher

    print("Downloading MS MARCO BM25 index (this may take a while)...")
    print("Index will be cached in ~/.cache/pyserini/")

    # This triggers download if not cached
    searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
    print(f"Index loaded successfully: {searcher.num_docs:,} documents")


def download_msmarco_queries(output_dir: Path, split: str = "train") -> None:
    """
    Download MS MARCO queries.

    Args:
        output_dir: Directory to save queries.
        split: Dataset split (train, dev, eval).
    """
    import ir_datasets

    print(f"Loading MS MARCO queries ({split})...")

    # This downloads the dataset if not cached
    dataset = ir_datasets.load(f"msmarco-passage/{split}")

    # Count queries
    count = sum(1 for _ in dataset.queries_iter())
    print(f"MS MARCO {split} queries available: {count:,}")


def download_beir_dataset(dataset_name: str, output_dir: Path) -> None:
    """
    Download a BEIR dataset.

    Args:
        dataset_name: BEIR dataset name (e.g., "scifact").
        output_dir: Directory to save dataset.
    """
    from beir import util

    print(f"Downloading BEIR dataset: {dataset_name}...")

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, str(output_dir / "beir"))

    print(f"BEIR {dataset_name} downloaded to {data_path}")


def verify_spacy_model() -> None:
    """Verify spaCy model is available."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model (en_core_web_sm) available")
    except OSError:
        print("Downloading spaCy model (en_core_web_sm)...")
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("spaCy model downloaded")


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
    ensure_dir(output_dir)

    print("=" * 60)
    print("ConstraintSuite - Data Download")
    print("=" * 60)

    # Download index
    print("\n[1/4] Downloading MS MARCO BM25 index...")
    download_msmarco_index(output_dir)

    if not args.index_only:
        # Download queries
        print("\n[2/4] Downloading MS MARCO queries...")
        download_msmarco_queries(output_dir)

        # Verify spaCy model
        print("\n[3/4] Verifying spaCy model...")
        verify_spacy_model()

        # BEIR datasets (optional)
        print("\n[4/4] BEIR datasets (optional, skipping for now)")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
