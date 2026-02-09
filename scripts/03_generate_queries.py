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
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.utils import load_config, setup_logging, set_seed, ensure_dir
from constraintsuite.data_loading import iter_queries
from constraintsuite.query_generation import batch_generate_queries


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
        default=None,
        help="Output path for generated queries (default: from config)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for query generation (default: from config or 50000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))

    # Set seed
    seed = args.seed or config.get("random_seed", 42)
    set_seed(seed)

    # Set output path
    output_path = args.output or Path(config["paths"]["intermediate"]) / "negated_queries.jsonl"
    ensure_dir(Path(output_path).parent)
    source_corpus = config.get("dataset", {}).get("source_corpus", "msmarco-passage")
    generation_cfg = config.get("generation", {})
    limit = args.limit if args.limit is not None else generation_cfg.get("limit")
    batch_size = args.batch_size or generation_cfg.get("batch_size", 50000)

    print("=" * 60)
    print("ConstraintSuite - Query Generation")
    print("=" * 60)

    # Stream base queries and generated outputs in batches to avoid high peak memory.
    print(f"\n[1/3] Streaming base queries from {source_corpus}...")
    if limit is not None:
        print(f"Limit: {limit:,} base queries")
    print(f"Batch size: {batch_size:,} base queries")

    print("\n[2/3] Generating and writing negated queries...")
    base_queries_seen = 0
    generated_count = 0
    template_counts = {}

    progress = tqdm(total=limit, desc="Base queries") if limit is not None else tqdm(
        desc="Base queries", unit="query"
    )

    with open(output_path, "w", encoding="utf-8") as out_f:
        for batch in iter_queries(source_corpus, split="train", batch_size=batch_size):
            if limit is not None and base_queries_seen >= limit:
                break

            if limit is not None:
                remaining = limit - base_queries_seen
                if remaining <= 0:
                    break
                if len(batch) > remaining:
                    batch = batch[:remaining]

            base_queries_seen += len(batch)
            progress.update(len(batch))

            generated = batch_generate_queries(batch, config)
            for query_id, gen_query in generated:
                out_f.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "query": {
                                "base": gen_query.base,
                                "neg": gen_query.negated,
                            },
                            "template": gen_query.template,
                            "y": gen_query.y,
                            "y_forms": gen_query.y_surface_forms,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                generated_count += 1
                template_counts[gen_query.template] = template_counts.get(gen_query.template, 0) + 1

    progress.close()
    print(f"Saved {generated_count} queries to {output_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Query Generation Statistics")
    print("=" * 60)
    print(f"  Base queries: {base_queries_seen}")
    print(f"  Generated queries: {generated_count}")
    success = (100 * generated_count / base_queries_seen) if base_queries_seen else 0
    print(f"  Success rate: {success:.1f}%")

    print("\nTemplate distribution:")
    for template, count in sorted(template_counts.items()):
        pct = (100 * count / generated_count) if generated_count else 0
        print(f"  {template}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("Query generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
