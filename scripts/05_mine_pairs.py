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
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.pair_mining import contains_y, mine_minpair, mine_pair, y_is_negated_nearby
from constraintsuite.utils import ensure_dir, iter_jsonl, load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Mine document pairs from candidates")
    parser.add_argument(
        "--config", type=str, default="configs/negation_v0.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--candidates", type=str, default=None, help="Input candidates (default: from config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for mined pairs (default: from config)",
    )
    parser.add_argument(
        "--prefer-explicit",
        action="store_true",
        default=True,
        help="Prefer doc_pos that explicitly negates Y",
    )
    parser.add_argument(
        "--include-minpairs",
        action="store_true",
        default=True,
        help="Also generate minpairs via surgical edits",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None, help="Stop early after writing this many mined pairs"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    # Set paths
    candidates_path = args.candidates or Path(config["paths"]["intermediate"]) / "candidates.jsonl"
    output_path = args.output or Path(config["paths"]["intermediate"]) / "raw_pairs.jsonl"
    ensure_dir(Path(output_path).parent)

    # Get mining parameters
    neg_window = config.get("filtering", {}).get("negation_window", 40)

    print("=" * 60)
    print("ConstraintSuite - Pair Mining")
    print("=" * 60)

    max_pairs = args.max_pairs or config.get("mining", {}).get("max_pairs")

    # Load candidates
    print(f"\n[1/3] Loading candidates from {candidates_path}...")
    print("Streaming candidates from disk...")

    # Mine pairs
    print("\n[2/3] Mining document pairs...")
    written_pairs = 0
    stats = {
        "total": 0,
        "success": 0,
        "minpairs": 0,
        "explicit": 0,
        "omission": 0,
        "no_violators": 0,
        "no_satisfiers": 0,
    }

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in tqdm(iter_jsonl(candidates_path), desc="Mining"):
            stats["total"] += 1

            query_id = item["query_id"]
            query = item["query"]
            y = item["y"]
            y_forms = item["y_forms"]
            candidates = item["candidates"]

            # Mine regular pair
            pair = mine_pair(
                candidates, y_forms, prefer_explicit=args.prefer_explicit, neg_window=neg_window
            )

            if pair:
                stats["success"] += 1
                stats[pair.slice_type] += 1

                out_f.write(
                    json.dumps(
                        {
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
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written_pairs += 1
            else:
                # Track why we failed
                violators = [
                    c
                    for c in candidates
                    if contains_y(c.get("text", ""), y_forms)
                    and not y_is_negated_nearby(c.get("text", ""), y)
                ]
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
                            out_f.write(
                                json.dumps(
                                    {
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
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            written_pairs += 1
                            break  # One minpair per edit type is enough
                    break  # Only try first suitable document

            if max_pairs is not None and written_pairs >= max_pairs:
                print(f"\nReached max_pairs={max_pairs}, stopping early.")
                break

    # Save results
    print(f"\n[3/3] Saved {written_pairs} pairs to {output_path}")

    # Statistics
    print("\n" + "=" * 60)
    print("Mining Statistics")
    print("=" * 60)
    print(f"  Queries processed: {stats['total']}")
    print(f"  Pairs mined: {stats['success']} ({100 * stats['success'] / stats['total']:.1f}%)")
    print("\nSlice distribution:")
    print(f"  minpairs: {stats['minpairs']}")
    print(f"  explicit: {stats['explicit']}")
    print(f"  omission: {stats['omission']}")
    print("\nFailure reasons:")
    print(f"  No violators: {stats['no_violators']}")
    print(f"  No satisfiers: {stats['no_satisfiers']}")

    print("\n" + "=" * 60)
    print("Pair mining complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
