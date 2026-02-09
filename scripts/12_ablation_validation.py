#!/usr/bin/env python3
"""
Script 12: Ablation validation.

After identifying hot components from the patching scan, this script:
1. Ablates top-k sites individually → measures negation accuracy drop
2. Ablates random sites of same count → negative control
3. Compares effect sizes to validate causal role

Usage:
    python scripts/12_ablation_validation.py --config configs/patching_v0.yaml
    python scripts/12_ablation_validation.py --scan-results results/patching/component_scan.json
    python scripts/12_ablation_validation.py --sites "layer.3.ffn_out,layer.4.attn_out"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.patching import (
    AblationResult,
    PatchableModel,
    PatchingSite,
    get_component_sites,
    get_head_sites,
    get_hot_sites,
    run_ablation_study,
)
from constraintsuite.utils import ensure_dir, load_config, load_jsonl, set_seed, setup_logging


def _site_from_name(name: str) -> PatchingSite:
    """Reconstruct a PatchingSite from its name string."""
    all_sites = get_component_sites() + get_head_sites()
    for site in all_sites:
        if site.name == name:
            return site
    raise ValueError(f"Unknown site name: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation validation for hot components"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/patching_v0.yaml",
        help="Path to patching configuration file",
    )
    parser.add_argument(
        "--scan-results",
        type=str,
        default=None,
        help="Path to scan results JSON (to auto-detect hot sites)",
    )
    parser.add_argument(
        "--sites",
        type=str,
        default=None,
        help="Comma-separated site names to ablate (overrides auto-detection)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSONL",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["zero", "mean"],
        default=None,
        help="Ablation method (default: from config)",
    )
    parser.add_argument(
        "--n-controls",
        type=int,
        default=None,
        help="Number of random control ablations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    patching_cfg = config.get("patching", {})
    ablation_cfg = config.get("ablation", {})
    dataset_cfg = config.get("dataset", {})
    setup_logging("INFO")
    set_seed(config.get("random_seed", 42))

    device = args.device or patching_cfg.get("device", "mps")
    method = args.method or ablation_cfg.get("method", "zero")
    n_controls = args.n_controls if args.n_controls is not None else ablation_cfg.get("n_random_controls", 5)
    checkpoint_dir = Path(patching_cfg.get("checkpoint_dir", "results/patching"))
    hot_threshold = patching_cfg.get("hot_threshold", 0.1)

    # Determine target sites
    if args.sites:
        site_names = [s.strip() for s in args.sites.split(",")]
        target_sites = [_site_from_name(name) for name in site_names]
        print(f"Manually specified {len(target_sites)} sites: {site_names}")
    elif args.scan_results:
        print(f"Loading scan results from {args.scan_results}...")
        with open(args.scan_results) as f:
            scan_data = json.load(f)

        # Re-create a minimal ScanResult to use get_hot_sites
        from constraintsuite.patching import ScanResult
        scan = ScanResult(
            scan_type=scan_data.get("scan_type", "component"),
            model_name=scan_data.get("model_name", ""),
            num_examples=scan_data.get("num_examples", 0),
            sites=scan_data.get("sites", []),
            site_stats=scan_data.get("site_stats", {}),
        )
        hot = get_hot_sites(scan, threshold=hot_threshold)
        if not hot:
            print(f"No hot sites found above threshold {hot_threshold}. Exiting.")
            return
        target_sites = [_site_from_name(name) for name, _ in hot]
        print(f"Auto-detected {len(target_sites)} hot sites:")
        for name, effect in hot:
            print(f"  {name}: |mean|={effect:.4f}")
    else:
        # Try default scan results path
        default_scan = checkpoint_dir / "component_scan.json"
        if default_scan.exists():
            print(f"Loading scan results from {default_scan}...")
            with open(default_scan) as f:
                scan_data = json.load(f)
            from constraintsuite.patching import ScanResult
            scan = ScanResult(
                scan_type=scan_data.get("scan_type", "component"),
                model_name=scan_data.get("model_name", ""),
                num_examples=scan_data.get("num_examples", 0),
                sites=scan_data.get("sites", []),
                site_stats=scan_data.get("site_stats", {}),
            )
            hot = get_hot_sites(scan, threshold=hot_threshold)
            if not hot:
                print(f"No hot sites above threshold {hot_threshold}. Exiting.")
                return
            target_sites = [_site_from_name(name) for name, _ in hot]
            print(f"Auto-detected {len(target_sites)} hot sites from {default_scan}")
        else:
            print("No scan results found. Run 11_patching_scan.py first, or specify --sites.")
            return

    # Load dataset
    dataset_path = args.dataset or dataset_cfg.get("main_path", "data/release/negation_v1/main.jsonl")
    print(f"\nLoading dataset from {dataset_path}...")
    examples = load_jsonl(dataset_path)
    if args.max_examples:
        examples = examples[: args.max_examples]
    print(f"Loaded {len(examples)} examples")

    # Load model
    model_name = patching_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L6-v2")
    max_length = patching_cfg.get("max_length", 256)
    print(f"\nLoading model: {model_name} on {device}...")
    model = PatchableModel(model_name=model_name, device=device, max_length=max_length)

    # Run ablation study
    print(f"\nRunning ablation study (method={method}, {n_controls} random controls)...")
    result = run_ablation_study(
        model=model,
        examples=examples,
        target_sites=target_sites,
        method=method,
        n_random_controls=n_controls,
        random_seed=config.get("random_seed", 42),
    )

    # Save results
    ensure_dir(checkpoint_dir)
    output_path = checkpoint_dir / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Ablation Results Summary")
    print("=" * 60)
    print(f"\nBaseline accuracy: {result.baseline_accuracy:.2%}")

    print("\nTargeted ablation:")
    print("-" * 50)
    for site_name, stats in result.targeted_results.items():
        print(
            f"  {site_name:<25} acc={stats['accuracy']:.2%}  "
            f"drop={stats['accuracy_drop']:+.2%}  gap={stats['mean_score_gap']:.4f}"
        )

    print("\nRandom controls:")
    print("-" * 50)
    for ctrl in result.random_controls:
        sites_str = ", ".join(ctrl["sites"][:3])
        if len(ctrl["sites"]) > 3:
            sites_str += f", ... ({len(ctrl['sites'])} total)"
        print(
            f"  Control {ctrl['control_idx']}: acc={ctrl['accuracy']:.2%}  "
            f"drop={ctrl['accuracy_drop']:+.2%}"
        )

    # Compute mean random control drop
    if result.random_controls:
        mean_random_drop = sum(c["accuracy_drop"] for c in result.random_controls) / len(
            result.random_controls
        )
        mean_targeted_drop = sum(
            s["accuracy_drop"] for s in result.targeted_results.values()
        ) / len(result.targeted_results) if result.targeted_results else 0

        print(f"\nMean targeted drop:  {mean_targeted_drop:+.2%}")
        print(f"Mean random drop:    {mean_random_drop:+.2%}")
        if mean_random_drop != 0:
            ratio = mean_targeted_drop / mean_random_drop
            print(f"Targeted/Random ratio: {ratio:.2f}x")

    print(f"\nElapsed: {result.elapsed_seconds:.1f}s")
    print("\n" + "=" * 60)
    print("Ablation validation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
