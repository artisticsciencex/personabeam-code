#!/usr/bin/env bash
# PersonaBEAM: One-command reproduction of all paper figures and tables.
# Usage: bash reproduce_all.sh
set -e

echo "=== PersonaBEAM Reproduction ==="

# 1. Install dependencies
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt

# 2. Download dataset from Hugging Face
echo "[2/3] Downloading dataset from Hugging Face..."
if [ ! -f ./personabeam/data/responses.parquet ]; then
    huggingface-cli download qzkiyoshi/personabeam \
        --repo-type=dataset \
        --local-dir ./personabeam
else
    echo "  Dataset already downloaded, skipping."
fi

# 3. Run analysis
echo "[3/3] Running analysis..."
python run_analysis.py \
    --data ./personabeam/data/responses.parquet \
    --output_dir ./outputs

echo ""
echo "=== Done ==="
echo "Figures and tables saved to ./outputs/"
echo "Expected: 5 PDF figures + 2 LaTeX tables"
ls -la ./outputs/
