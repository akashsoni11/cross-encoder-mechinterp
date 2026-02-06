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
import sys
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.utils import load_config, setup_logging, save_jsonl, set_seed, ensure_dir
from constraintsuite.data_loading import load_msmarco_queries
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

    print("=" * 60)
    print("ConstraintSuite - Query Generation")
    print("=" * 60)

    # Load base queries
    print("\n[1/3] Loading MS MARCO queries...")
    base_queries = load_msmarco_queries("train", limit=args.limit)
    print(f"Loaded {len(base_queries)} queries")

    # Convert to list of tuples
    query_list = [(qid, text) for qid, text in base_queries.items()]

    # Generate negated queries
    print("\n[2/3] Generating negated queries...")
    generated = batch_generate_queries(query_list, config)
    print(f"Generated {len(generated)} negated queries")

    # Format for output
    print("\n[3/3] Saving to JSONL...")
    output_data = []
    for query_id, gen_query in generated:
        output_data.append({
            "query_id": query_id,
            "query": {
                "base": gen_query.base,
                "neg": gen_query.negated,
            },
            "template": gen_query.template,
            "y": gen_query.y,
            "y_forms": gen_query.y_surface_forms,
        })

    save_jsonl(output_data, output_path)
    print(f"Saved {len(output_data)} queries to {output_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Query Generation Statistics")
    print("=" * 60)
    print(f"  Base queries: {len(base_queries)}")
    print(f"  Generated queries: {len(generated)}")
    print(f"  Success rate: {100 * len(generated) / len(base_queries):.1f}%")

    # Template distribution
    template_counts = {}
    for _, gen_query in generated:
        template_counts[gen_query.template] = template_counts.get(gen_query.template, 0) + 1
    print("\nTemplate distribution:")
    for template, count in sorted(template_counts.items()):
        print(f"  {template}: {count} ({100 * count / len(generated):.1f}%)")

    print("\n" + "=" * 60)
    print("Query generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
