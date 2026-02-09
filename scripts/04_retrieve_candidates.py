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
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.utils import ensure_dir, load_config, load_jsonl, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Retrieve candidate documents using BM25")
    parser.add_argument(
        "--config", type=str, default="configs/negation_v0.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--queries", type=str, default=None, help="Input negated queries (default: from config)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path for candidates (default: from config)"
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Number of candidates per query (default: from config)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of retrieval threads (default: from config or CPU count)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for retrieval (default: from config or 1000)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    # Import lazily so --help works even when Java/Pyserini is not installed yet.
    from constraintsuite.retrieval import BM25Retriever

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
    retrieval_threads = (
        args.threads
        or config.get("retrieval", {}).get("threads")
        or max(1, (os.cpu_count() or 1) - 1)
    )
    batch_size = args.batch_size or config.get("retrieval", {}).get("batch_size", 1000)

    print(
        f"\n[3/3] Retrieving top-{k} candidates/query with "
        f"{retrieval_threads} threads, batch_size={batch_size}..."
    )

    # Process in batches and stream to disk to keep memory bounded.
    total_results = 0
    total_candidates = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(queries), batch_size), desc="Batches"):
            batch = queries[i : i + batch_size]

            # Prepare batch for retrieval
            batch_queries = [(q["query_id"], q["query"]["neg"]) for q in batch]

            # Batch retrieve
            batch_results = retriever.retrieve_batch(batch_queries, k=k, threads=retrieval_threads)

            # Combine with original query data
            for q in batch:
                qid = q["query_id"]
                candidates = batch_results.get(qid, [])
                total_candidates += len(candidates)

                out_f.write(
                    json.dumps(
                        {
                            "query_id": qid,
                            "query": q["query"],
                            "y": q["y"],
                            "y_forms": q["y_forms"],
                            "template": q["template"],
                            "candidates": candidates,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_results += 1

    print(f"\nSaved {total_results} query results to {output_path}")

    # Statistics
    avg_candidates = total_candidates / total_results if total_results else 0

    print("\n" + "=" * 60)
    print("Retrieval Statistics")
    print("=" * 60)
    print(f"  Queries processed: {total_results}")
    print(f"  Total candidates: {total_candidates:,}")
    print(f"  Avg candidates/query: {avg_candidates:.1f}")

    print("\n" + "=" * 60)
    print("Candidate retrieval complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
