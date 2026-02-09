#!/usr/bin/env python3
"""
Script 10: Curate gold set based on LLM validation report.

Reads gold.jsonl and gold_validation_report.json, separates valid from invalid,
writes curated gold set and issue summary.

Usage:
    python scripts/10_curate_gold.py --config configs/negation_v1.yaml
    python scripts/10_curate_gold.py --release-dir data/release/negation_v1
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constraintsuite.utils import load_config


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Curate gold set from validation report")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--release-dir", type=str, default=None, help="Path to release dir")
    parser.add_argument("--intermediate-dir", type=str, default=None, help="Path to intermediate dir")
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        release_dir = Path(args.release_dir or config.get("paths", {}).get("release", "data/release/negation_v1"))
        intermediate_dir = Path(args.intermediate_dir or config.get("paths", {}).get("intermediate", "data/intermediate/negation_v1"))
    else:
        release_dir = Path(args.release_dir or "data/release/negation_v1")
        intermediate_dir = Path(args.intermediate_dir or "data/intermediate/negation_v1")

    gold_path = release_dir / "gold.jsonl"
    report_path = intermediate_dir / "gold_validation_report.json"
    curated_path = release_dir / "gold_curated.jsonl"
    summary_path = intermediate_dir / "gold_issues_summary.json"

    # Load gold set
    if not gold_path.exists():
        print(f"ERROR: Gold set not found: {gold_path}")
        sys.exit(1)
    gold_records = load_jsonl(gold_path)
    gold_by_id = {r["id"]: r for r in gold_records}
    print(f"Loaded {len(gold_records)} gold records from {gold_path}")

    # Load validation report
    if not report_path.exists():
        print(f"ERROR: Validation report not found: {report_path}")
        sys.exit(1)
    with open(report_path) as f:
        report = json.load(f)
    print(f"Loaded validation report: {report['valid']} valid, {report['invalid']} invalid")

    # Separate valid from invalid
    valid_ids = set()
    invalid_details = []
    issue_counter = Counter()

    for detail in report["details"]:
        rec_id = detail["id"]
        if detail["valid"]:
            valid_ids.add(rec_id)
        else:
            issues = detail.get("issues", [])
            for issue in issues:
                issue_counter[issue] += 1
            invalid_details.append({
                "id": rec_id,
                "confidence": detail.get("confidence"),
                "issues": issues,
                "reasoning": detail.get("reasoning", ""),
                "doc_pos_status": detail.get("doc_pos_status"),
                "doc_neg_status": detail.get("doc_neg_status"),
            })

    # Write curated gold set (valid only)
    curated = [gold_by_id[rid] for rid in valid_ids if rid in gold_by_id]
    write_jsonl(curated_path, curated)
    print(f"\nWrote {len(curated)} curated gold records to {curated_path}")

    # Build issue summary
    # Categorize by slice
    slice_breakdown = Counter()
    for detail in invalid_details:
        rec = gold_by_id.get(detail["id"], {})
        slice_type = rec.get("slice_type", rec.get("suite", "unknown"))
        if "minpairs" in str(slice_type):
            slice_breakdown["minpairs"] += 1
        elif "explicit" in str(slice_type):
            slice_breakdown["explicit"] += 1
        elif "omission" in str(slice_type):
            slice_breakdown["omission"] += 1
        else:
            slice_breakdown["unknown"] += 1

    summary = {
        "total_gold": len(gold_records),
        "valid": len(curated),
        "invalid": len(invalid_details),
        "validity_rate": len(curated) / len(gold_records) if gold_records else 0,
        "issue_counts": dict(issue_counter.most_common()),
        "invalid_by_slice": dict(slice_breakdown),
        "invalid_details": invalid_details,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote issue summary to {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Gold Set Curation Summary")
    print(f"{'='*60}")
    print(f"Total gold records:   {len(gold_records)}")
    print(f"Valid (curated):      {len(curated)}")
    print(f"Invalid (removed):    {len(invalid_details)}")
    print(f"Validity rate:        {len(curated)/len(gold_records):.0%}")

    print(f"\nIssue breakdown (invalid records):")
    for issue, count in issue_counter.most_common():
        print(f"  {issue}: {count}")

    print(f"\nInvalid by slice:")
    for slice_type, count in slice_breakdown.most_common():
        print(f"  {slice_type}: {count}")


if __name__ == "__main__":
    main()
