#!/usr/bin/env python3
"""
Script 05: Mine document pairs from candidates.

For each query's candidate pool, mines (doc_pos, doc_neg) pairs where:
- doc_neg: contains the forbidden entity Y affirmatively
- doc_pos: either omits Y or mentions Y in negated form

Usage:
    python scripts/05_mine_pairs.py --config configs/negation_v0.yaml
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.utils import load_config, setup_logging, load_jsonl, save_jsonl, ensure_dir
from constraintsuite.pair_mining import mine_pair, mine_minpair, contains_y, y_is_negated_nearby


def main():
    parser = argparse.ArgumentParser(
        description="Mine document pairs from candidates"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/negation_v0.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Input candidates (default: from config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for mined pairs (default: from config)"
    )
    parser.add_argument(
        "--prefer-explicit",
        action="store_true",
        default=True,
        help="Prefer doc_pos that explicitly negates Y"
    )
    parser.add_argument(
        "--include-minpairs",
        action="store_true",
        default=True,
        help="Also generate minpairs via surgical edits"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))

    # Set paths
    candidates_path = args.candidates or Path(config["paths"]["intermediate"]) / "candidates.jsonl"
    output_path = args.output or Path(config["paths"]["intermediate"]) / "raw_pairs.jsonl"
    ensure_dir(Path(output_path).parent)

    # Get mining parameters
    neg_window = config.get("filtering", {}).get("negation_window", 40)

    print("=" * 60)
    print("ConstraintSuite - Pair Mining")
    print("=" * 60)

    # Load candidates
    print(f"\n[1/3] Loading candidates from {candidates_path}...")
    candidates_data = load_jsonl(candidates_path)
    print(f"Loaded {len(candidates_data)} queries with candidates")

    # Mine pairs
    print("\n[2/3] Mining document pairs...")
    results = []
    stats = {
        "total": 0,
        "success": 0,
        "minpairs": 0,
        "explicit": 0,
        "omission": 0,
        "no_violators": 0,
        "no_satisfiers": 0,
    }

    for item in tqdm(candidates_data, desc="Mining"):
        stats["total"] += 1

        query_id = item["query_id"]
        query = item["query"]
        y = item["y"]
        y_forms = item["y_forms"]
        candidates = item["candidates"]

        # Mine regular pair
        pair = mine_pair(
            candidates,
            y_forms,
            prefer_explicit=args.prefer_explicit,
            neg_window=neg_window
        )

        if pair:
            stats["success"] += 1
            stats[pair.slice_type] += 1

            results.append({
                "id": f"negation_{pair.slice_type}_{query_id}",
                "suite": f"negation_{pair.slice_type}",
                "query_id": query_id,
                "query": query,
                "constraint": {
                    "type": "exclude",
                    "y": y,
                },
                "y_forms": y_forms,
                "doc_pos": pair.doc_pos,
                "doc_neg": pair.doc_neg,
                "slice_type": pair.slice_type,
                "metadata": pair.metadata,
            })
        else:
            # Track why we failed
            violators = [c for c in candidates if contains_y(c.get("text", ""), y_forms)
                        and not y_is_negated_nearby(c.get("text", ""), y)]
            if not violators:
                stats["no_violators"] += 1
            else:
                stats["no_satisfiers"] += 1

        # Try to create minpair
        if args.include_minpairs and candidates:
            # Find a good document for minpair editing
            for doc in candidates[:20]:  # Check first 20 candidates
                if not contains_y(doc.get("text", ""), y_forms):
                    continue
                if y_is_negated_nearby(doc.get("text", ""), y):
                    continue

                # Try different edit types
                for edit_type in ["replace_contains", "insert_no", "add_free"]:
                    minpair = mine_minpair(doc, y, edit_type)
                    if minpair:
                        stats["minpairs"] += 1
                        results.append({
                            "id": f"negation_minpairs_{query_id}_{edit_type}",
                            "suite": "negation_minpairs",
                            "query_id": f"{query_id}_minpair",
                            "query": query,
                            "constraint": {
                                "type": "exclude",
                                "y": y,
                            },
                            "y_forms": y_forms,
                            "doc_pos": minpair.doc_pos,
                            "doc_neg": minpair.doc_neg,
                            "slice_type": "minpairs",
                            "metadata": minpair.metadata,
                        })
                        break  # One minpair per edit type is enough
                break  # Only try first suitable document

    # Save results
    print(f"\n[3/3] Saving {len(results)} pairs to {output_path}...")
    save_jsonl(results, output_path)

    # Statistics
    print("\n" + "=" * 60)
    print("Mining Statistics")
    print("=" * 60)
    print(f"  Queries processed: {stats['total']}")
    print(f"  Pairs mined: {stats['success']} ({100 * stats['success'] / stats['total']:.1f}%)")
    print(f"\nSlice distribution:")
    print(f"  minpairs: {stats['minpairs']}")
    print(f"  explicit: {stats['explicit']}")
    print(f"  omission: {stats['omission']}")
    print(f"\nFailure reasons:")
    print(f"  No violators: {stats['no_violators']}")
    print(f"  No satisfiers: {stats['no_satisfiers']}")

    print("\n" + "=" * 60)
    print("Pair mining complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
