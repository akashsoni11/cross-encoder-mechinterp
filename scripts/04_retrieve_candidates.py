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
import sys
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.utils import load_config, setup_logging, load_jsonl, save_jsonl, ensure_dir
from constraintsuite.retrieval import BM25Retriever


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
        default=None,
        help="Input negated queries (default: from config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for candidates (default: from config)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of candidates per query (default: from config)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of retrieval threads"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for retrieval"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))

    # Set paths
    queries_path = args.queries or Path(config["paths"]["intermediate"]) / "negated_queries.jsonl"
    output_path = args.output or Path(config["paths"]["intermediate"]) / "candidates.jsonl"
    ensure_dir(Path(output_path).parent)

    # Get k from config
    k = args.k or config.get("retrieval", {}).get("k_pool", 200)

    print("=" * 60)
    print("ConstraintSuite - Candidate Retrieval")
    print("=" * 60)

    # Load queries
    print(f"\n[1/3] Loading queries from {queries_path}...")
    queries = load_jsonl(queries_path)
    print(f"Loaded {len(queries)} queries")

    # Initialize retriever
    print("\n[2/3] Initializing BM25 retriever...")
    index_name = config.get("retrieval", {}).get("index_name", "msmarco-v1-passage")
    k1 = config.get("retrieval", {}).get("bm25_k1", 0.9)
    b = config.get("retrieval", {}).get("bm25_b", 0.4)
    retriever = BM25Retriever(index_name, k1=k1, b=b)
    print(f"Index loaded: {len(retriever):,} documents")

    # Retrieve candidates
    print(f"\n[3/3] Retrieving top-{k} candidates per query...")

    # Process in batches
    results = []
    for i in tqdm(range(0, len(queries), args.batch_size), desc="Batches"):
        batch = queries[i:i + args.batch_size]

        # Prepare batch for retrieval
        batch_queries = [
            (q["query_id"], q["query"]["neg"])
            for q in batch
        ]

        # Batch retrieve
        batch_results = retriever.retrieve_batch(batch_queries, k=k, threads=args.threads)

        # Combine with original query data
        for q in batch:
            qid = q["query_id"]
            candidates = batch_results.get(qid, [])

            results.append({
                "query_id": qid,
                "query": q["query"],
                "y": q["y"],
                "y_forms": q["y_forms"],
                "template": q["template"],
                "candidates": candidates,
            })

    # Save results
    save_jsonl(results, output_path)
    print(f"\nSaved {len(results)} query results to {output_path}")

    # Statistics
    total_candidates = sum(len(r["candidates"]) for r in results)
    avg_candidates = total_candidates / len(results) if results else 0

    print("\n" + "=" * 60)
    print("Retrieval Statistics")
    print("=" * 60)
    print(f"  Queries processed: {len(results)}")
    print(f"  Total candidates: {total_candidates:,}")
    print(f"  Avg candidates/query: {avg_candidates:.1f}")

    print("\n" + "=" * 60)
    print("Candidate retrieval complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
