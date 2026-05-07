#!/usr/bin/env bash
# PersonaBEAM: One-command reproduction of all paper figures and tables.
#
# Usage:
#   bash reproduce_all.sh
#
# This script:
#   1. Creates a virtual environment and installs dependencies
#   2. Downloads the PersonaBEAM dataset from Hugging Face
#   3. Runs the main analysis (figures 1-4, tables, statistics;
#      plus ablation figures 5-6 if ablation data is present)
#   4. Runs the semantic analysis (figure 7, tables)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_DIR="./personabeam"
OUTPUT_DIR="./outputs"

# --- 1. Set up Python environment ---
if [ ! -d ".venv" ]; then
    echo ">>> Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo ">>> Installing dependencies..."
pip install -q -r requirements.txt

# --- 2. Download dataset from Hugging Face ---
if [ ! -f "$DATASET_DIR/data/responses.parquet" ]; then
    echo ">>> Downloading PersonaBEAM dataset..."
    pip install -q huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='qzkiyoshi/personabeam', repo_type='dataset', local_dir='$DATASET_DIR')
"
    echo ">>> Dataset downloaded to $DATASET_DIR/"
else
    echo ">>> Dataset already present at $DATASET_DIR/"
fi

mkdir -p "$OUTPUT_DIR"

# --- 3. Main analysis ---
echo ""
echo ">>> Running main analysis (figures 1-4, tables, statistics)..."
ABLATION_FLAG=""
# Check if ablation CSVs exist in the dataset
if ls "$DATASET_DIR"/ablation/results_*_nopersona.csv 1>/dev/null 2>&1; then
    ABLATION_FLAG="--ablation_dir $DATASET_DIR/ablation"
    echo "    (ablation data found — will generate figures 5-6 and ablation table)"
fi
python3 run_analysis.py \
    --data "$DATASET_DIR/data/responses.parquet" \
    --output_dir "$OUTPUT_DIR" \
    $ABLATION_FLAG

# --- 4. Semantic analysis ---
echo ""
echo ">>> Running semantic analysis (TF-IDF vocabulary, similarity)..."
python3 run_semantic_analysis.py \
    --parquet "$DATASET_DIR/data/responses.parquet" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "All outputs saved to $OUTPUT_DIR/"
echo "=========================================="
ls -1 "$OUTPUT_DIR"/*.pdf "$OUTPUT_DIR"/*.tex 2>/dev/null
