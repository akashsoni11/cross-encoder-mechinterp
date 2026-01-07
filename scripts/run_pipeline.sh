#!/bin/bash
# ConstraintSuite Dataset Construction Pipeline
# ==============================================
#
# Runs the full pipeline from data download to baseline evaluation.
#
# Usage:
#   ./scripts/run_pipeline.sh                    # Use default config
#   ./scripts/run_pipeline.sh configs/custom.yaml  # Use custom config
#
# Prerequisites:
#   - Python 3.10+
#   - pip install -e .  (install constraintsuite package)
#   - ~2GB disk space for MS MARCO index

set -e  # Exit on error

# Configuration
CONFIG="${1:-configs/negation_v0.yaml}"
SCRIPTS_DIR="$(dirname "$0")"

echo "============================================================"
echo "ConstraintSuite - Full Pipeline"
echo "============================================================"
echo "Config: $CONFIG"
echo "============================================================"

# Step 1: Download data
echo ""
echo "[Step 1/8] Downloading data..."
python "$SCRIPTS_DIR/01_download_data.py" --config "$CONFIG"

# Step 2: Verify index
echo ""
echo "[Step 2/8] Verifying BM25 index..."
python "$SCRIPTS_DIR/02_build_index.py" --config "$CONFIG" --verify-only

# Step 3: Generate queries
echo ""
echo "[Step 3/8] Generating negated queries..."
python "$SCRIPTS_DIR/03_generate_queries.py" --config "$CONFIG"

# Step 4: Retrieve candidates
echo ""
echo "[Step 4/8] Retrieving candidate documents..."
python "$SCRIPTS_DIR/04_retrieve_candidates.py" --config "$CONFIG"

# Step 5: Mine pairs
echo ""
echo "[Step 5/8] Mining document pairs..."
python "$SCRIPTS_DIR/05_mine_pairs.py" --config "$CONFIG"

# Step 6: Filter and tag
echo ""
echo "[Step 6/8] Filtering and tagging pairs..."
python "$SCRIPTS_DIR/06_filter_and_tag.py" --config "$CONFIG"

# Step 7: Sample gold set
echo ""
echo "[Step 7/8] Sampling gold set..."
python "$SCRIPTS_DIR/07_sample_gold.py" --config "$CONFIG"

# Step 8: Evaluate baselines
echo ""
echo "[Step 8/8] Evaluating baseline rerankers..."
python "$SCRIPTS_DIR/08_eval_baselines.py" --config "$CONFIG"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  - Main dataset:  data/release/negation_v0/main.jsonl"
echo "  - Gold set:      data/release/negation_v0/gold.jsonl"
echo "  - Results:       results/baseline_eval.json"
echo ""
echo "Next steps:"
echo "  1. Manually verify gold set examples"
echo "  2. Run mechanistic analysis (patching/ablation)"
echo "============================================================"
