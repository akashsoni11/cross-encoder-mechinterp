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

read_cfg() {
  local key="$1"
  python - "$CONFIG" "$key" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
key_path = sys.argv[2].split(".")
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
value = cfg
for part in key_path:
    if not isinstance(value, dict) or part not in value:
        print("")
        raise SystemExit(0)
    value = value[part]
if value is None:
    print("")
else:
    print(value)
PY
}

GEN_LIMIT="$(read_cfg generation.limit)"
GEN_BATCH_SIZE="$(read_cfg generation.batch_size)"
RET_BATCH_SIZE="$(read_cfg retrieval.batch_size)"
RET_THREADS="$(read_cfg retrieval.threads)"
MODEL_DEVICE="$(read_cfg models.device)"
CPU_THREADS="$(read_cfg models.cpu_threads)"
RELEASE_DIR="$(read_cfg paths.release)"

echo "============================================================"
echo "ConstraintSuite - Full Pipeline"
echo "============================================================"
echo "Config: $CONFIG"
echo "============================================================"

if [ -n "$CPU_THREADS" ]; then
  export OMP_NUM_THREADS="$CPU_THREADS"
  export MKL_NUM_THREADS="$CPU_THREADS"
  export OPENBLAS_NUM_THREADS="$CPU_THREADS"
fi
export TOKENIZERS_PARALLELISM=true

if [ "$MODEL_DEVICE" = "mps" ]; then
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.95}"
fi

# Step 1: Download data
echo ""
echo "[Step 1/9] Downloading data..."
python "$SCRIPTS_DIR/01_download_data.py" --config "$CONFIG"

# Step 2: Verify index
echo ""
echo "[Step 2/9] Verifying BM25 index..."
python "$SCRIPTS_DIR/02_build_index.py" --config "$CONFIG" --verify-only

# Step 3: Generate queries
echo ""
echo "[Step 3/9] Generating negated queries..."
STEP3_CMD=(python "$SCRIPTS_DIR/03_generate_queries.py" --config "$CONFIG")
if [ -n "$GEN_LIMIT" ]; then
  STEP3_CMD+=(--limit "$GEN_LIMIT")
fi
if [ -n "$GEN_BATCH_SIZE" ]; then
  STEP3_CMD+=(--batch-size "$GEN_BATCH_SIZE")
fi
"${STEP3_CMD[@]}"

# Step 4: Retrieve candidates
echo ""
echo "[Step 4/9] Retrieving candidate documents..."
STEP4_CMD=(python "$SCRIPTS_DIR/04_retrieve_candidates.py" --config "$CONFIG")
if [ -n "$RET_THREADS" ]; then
  STEP4_CMD+=(--threads "$RET_THREADS")
fi
if [ -n "$RET_BATCH_SIZE" ]; then
  STEP4_CMD+=(--batch-size "$RET_BATCH_SIZE")
fi
"${STEP4_CMD[@]}"

# Step 5: Mine pairs
echo ""
echo "[Step 5/9] Mining document pairs..."
python "$SCRIPTS_DIR/05_mine_pairs.py" --config "$CONFIG"

# Step 6: Filter and tag
echo ""
echo "[Step 6/9] Filtering and tagging pairs..."
python "$SCRIPTS_DIR/06_filter_and_tag.py" --config "$CONFIG"

# Step 7: Sample gold set
echo ""
echo "[Step 7/9] Sampling gold set..."
python "$SCRIPTS_DIR/07_sample_gold.py" --config "$CONFIG"

# Step 8: Evaluate baselines
echo ""
echo "[Step 8/9] Evaluating baseline rerankers..."
python "$SCRIPTS_DIR/08_eval_baselines.py" --config "$CONFIG"

# Step 9: LLM-assisted dataset enhancement (optional, requires Gemini or Codex CLI)
echo ""
echo "[Step 9/9] LLM-assisted enhancement..."
if command -v gemini &>/dev/null; then
  python "$SCRIPTS_DIR/09_codex_enhance.py" --config "$CONFIG" --task all --backend gemini
elif command -v codex &>/dev/null; then
  echo "  Gemini CLI not found, falling back to Codex CLI..."
  python "$SCRIPTS_DIR/09_codex_enhance.py" --config "$CONFIG" --task all --backend codex
else
  echo "  Neither Gemini nor Codex CLI found — skipping LLM enhancement."
  echo "  Install Gemini CLI: npm install -g @google/gemini-cli"
  echo "  Or Codex CLI: pip install codex-cli && codex login"
fi

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  - Main dataset:  ${RELEASE_DIR:-data/release/negation_v0}/main.jsonl"
echo "  - Gold set:      ${RELEASE_DIR:-data/release/negation_v0}/gold.jsonl"
echo "  - Results:       results/baseline_eval.json"
echo ""
echo "Next steps:"
echo "  1. Manually verify gold set examples"
echo "  2. Run mechanistic analysis (patching/ablation)"
echo "  3. To re-run enhancement: python scripts/09_codex_enhance.py --config $CONFIG --backend gemini"
echo "============================================================"
