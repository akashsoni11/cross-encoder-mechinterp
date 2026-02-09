#!/usr/bin/env python3
"""
Script 11: Activation patching scan.

Runs activation patching to localize negation processing in cross-encoder
rerankers. Supports component-level (25 sites) and head-level (72 sites) scans.

Usage:
    python scripts/11_patching_scan.py --config configs/patching_v0.yaml
    python scripts/11_patching_scan.py --scan-type head --max-examples 500
    python scripts/11_patching_scan.py --resume
    python scripts/11_patching_scan.py --dataset data/release/negation_v1/gold_curated.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.patching import (
    PatchableModel,
    get_hot_sites,
    plot_component_heatmap,
    plot_head_heatmap,
    run_component_scan,
    run_head_scan,
)
from constraintsuite.utils import ensure_dir, load_config, load_jsonl, set_seed, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Run activation patching scan on ConstraintSuite"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/patching_v0.yaml",
        help="Path to patching configuration file",
    )
    parser.add_argument(
        "--scan-type",
        type=str,
        choices=["component", "head", "both"],
        default=None,
        help="Type of scan (default: from config)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSONL (default: from config)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (auto, cuda, cpu, mps)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--head-layers",
        type=str,
        default=None,
        help="Comma-separated layer indices for head scan (e.g., '2,3,4')",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    patching_cfg = config.get("patching", {})
    dataset_cfg = config.get("dataset", {})
    setup_logging("INFO")
    set_seed(config.get("random_seed", 42))

    # Resolve parameters
    scan_type = args.scan_type or patching_cfg.get("scan_type", "component")
    device = args.device or patching_cfg.get("device", "mps")
    checkpoint_dir = Path(patching_cfg.get("checkpoint_dir", "results/patching"))
    checkpoint_every = patching_cfg.get("checkpoint_every", 100)
    min_total_effect = patching_cfg.get("min_total_effect", 0.5)
    hot_threshold = patching_cfg.get("hot_threshold", 0.1)

    # Load dataset
    dataset_path = args.dataset or dataset_cfg.get("main_path", "data/release/negation_v1/main.jsonl")
    print(f"Loading dataset from {dataset_path}...")
    examples = load_jsonl(dataset_path)
    if args.max_examples:
        examples = examples[: args.max_examples]
    print(f"Loaded {len(examples)} examples")

    # Load model
    model_name = patching_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L6-v2")
    max_length = patching_cfg.get("max_length", 256)
    print(f"\nLoading model: {model_name} on {device}...")
    model = PatchableModel(model_name=model_name, device=device, max_length=max_length)
    print("Model loaded.")

    ensure_dir(checkpoint_dir)

    # Run scans
    if scan_type in ("component", "both"):
        print("\n" + "=" * 60)
        print("Running component-level scan (25 sites)")
        print("=" * 60)

        component_scan = run_component_scan(
            model=model,
            examples=examples,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_every=checkpoint_every,
            min_total_effect=min_total_effect,
            resume=args.resume,
        )

        # Save results
        output_path = checkpoint_dir / "component_scan.json"
        with open(output_path, "w") as f:
            json.dump(component_scan.to_dict(), f, indent=2)
        print(f"\nResults saved to {output_path}")

        # Print summary
        print("\nComponent effects (sorted by |mean|):")
        print("-" * 50)
        sorted_sites = sorted(
            component_scan.site_stats.items(),
            key=lambda x: abs(x[1]["mean"]),
            reverse=True,
        )
        for name, stats in sorted_sites:
            print(
                f"  {name:<25} mean={stats['mean']:+.4f}  "
                f"|mean|={stats['abs_mean']:.4f}  std={stats['std']:.4f}"
            )

        # Plot heatmap
        try:
            heatmap_path = checkpoint_dir / "component_heatmap.png"
            plot_component_heatmap(component_scan, output_path=heatmap_path)
            print(f"Heatmap saved to {heatmap_path}")
        except Exception as e:
            print(f"Warning: Could not generate heatmap: {e}")

        # Hot sites
        hot = get_hot_sites(component_scan, threshold=hot_threshold)
        if hot:
            print(f"\nHot sites (threshold={hot_threshold}):")
            for name, effect in hot:
                print(f"  {name}: {effect:.4f}")
        else:
            print(f"\nNo hot sites found above threshold {hot_threshold}")

    if scan_type in ("head", "both"):
        print("\n" + "=" * 60)
        print("Running head-level scan")
        print("=" * 60)

        head_layers = None
        if args.head_layers:
            head_layers = [int(x.strip()) for x in args.head_layers.split(",")]
            print(f"Scanning layers: {head_layers}")

        head_scan = run_head_scan(
            model=model,
            examples=examples,
            layers=head_layers,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_every=checkpoint_every,
            min_total_effect=min_total_effect,
            resume=args.resume,
        )

        # Save results
        output_path = checkpoint_dir / "head_scan.json"
        with open(output_path, "w") as f:
            json.dump(head_scan.to_dict(), f, indent=2)
        print(f"\nResults saved to {output_path}")

        # Print top heads
        print("\nTop 10 heads by |mean effect|:")
        print("-" * 50)
        sorted_heads = sorted(
            head_scan.site_stats.items(),
            key=lambda x: abs(x[1]["mean"]),
            reverse=True,
        )
        for name, stats in sorted_heads[:10]:
            print(
                f"  {name:<25} mean={stats['mean']:+.4f}  "
                f"|mean|={stats['abs_mean']:.4f}  std={stats['std']:.4f}"
            )

        # Plot heatmap
        try:
            heatmap_path = checkpoint_dir / "head_heatmap.png"
            plot_head_heatmap(head_scan, output_path=heatmap_path)
            print(f"Heatmap saved to {heatmap_path}")
        except Exception as e:
            print(f"Warning: Could not generate heatmap: {e}")

    print("\n" + "=" * 60)
    print("Patching scan complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
