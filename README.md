# PersonaBEAM: Code for Cross-Model Persona Conditioning Evaluation

This repository contains the inference and analysis code for the PersonaBEAM benchmark, which evaluates persona conditioning effects in vision-language model (VLM)-based embodied agent control.

**Dataset:** [huggingface.co/datasets/qzkiyoshi/personabeam](https://huggingface.co/datasets/qzkiyoshi/personabeam)

## Repository Structure

```
personabeam-code/
├── README.md                  # this file
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT
├── reproduce_all.sh           # one-command reproduction
├── run_inference.py           # cross-model inference pipeline
├── run_analysis.py            # main analysis: figures 1-4, tables, ablation (§5.5)
├── run_semantic_analysis.py   # semantic analysis of reasoning texts (§5.4)
└── outputs/                   # default output directory (gitignored)
```

## Quick Reproduction

```bash
git clone https://github.com/artisticsciencex/personabeam-code.git
cd personabeam-code
bash reproduce_all.sh
```

This downloads the dataset from Hugging Face, runs the full analysis, and saves all figures and tables to `outputs/`.

## Setup (Manual)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API Keys (for inference only)

If you want to re-run inference, set the following environment variables:

```bash
export OPENAI_API_KEY="sk-..."           # for GPT-5.5
export ANTHROPIC_API_KEY="sk-ant-..."    # for Claude Opus 4.7
export GOOGLE_CLOUD_PROJECT="my-project" # for Gemini 3.1 Pro (Vertex AI)
```

For open-weight models (Qwen3.6-35B-A3B, Gemma 4 31B), you need a local serving backend:
- **Qwen**: [vLLM](https://github.com/vllm-project/vllm) with `--served-model-name Qwen/Qwen3.6-35B-A3B-FP8`
- **Gemma**: [Ollama](https://ollama.ai/) with `ollama pull gemma4:31b`

## Usage

### 1. Reproduce Paper Results (Analysis Only)

**Main analysis** (figures 1-4, tables, statistics):

```bash
python run_analysis.py \
    --data ./personabeam/data/responses.parquet \
    --output_dir ./outputs
```

**With ablation analysis** (adds figures 5-6, ablation table, JSD/sensitivity stats):

```bash
python run_analysis.py \
    --data ./personabeam/data/responses.parquet \
    --ablation_dir ./personabeam/ablation \
    --output_dir ./outputs
```

**Semantic analysis** (TF-IDF vocabulary, embedding similarity):

```bash
python run_semantic_analysis.py \
    --parquet ./personabeam/data/responses.parquet \
    --output_dir ./outputs
```

This generates:

| Output | Description |
|--------|-------------|
| `fig1_command_by_persona_model.pdf` | Command distribution heatmaps (Fig 1) |
| `fig2_cramers_v_env_model.pdf` | Cramer's V by environment x model (Fig 2) |
| `fig3_cross_model_agreement.pdf` | Agreement rates + pairwise matrix (Fig 3) |
| `fig4_explorer_by_environment.pdf` | Explorer heatmap by env x model (Fig 4) |
| `fig5_ablation_nopersona.pdf` | No-persona ablation comparison (Fig 5) |
| `fig6_ablation_noimage.pdf` | Noise-image ablation comparison (Fig 6) |
| `fig7_semantic_similarity.pdf` | Semantic similarity by grouping (Fig 7) |
| `fig_latency_appendix.pdf` | Latency violin plots (Appendix) |
| `tab_command_distribution.tex` | Command distribution table (LaTeX) |
| `tab_cramers_v.tex` | Cramer's V table (LaTeX) |
| `tab_ablation_nopersona_n.tex` | Per-model ablation sample sizes (LaTeX) |
| `tab_semantic_similarity.tex` | Embedding similarity by grouping (LaTeX) |
| `tab_persona_vocabulary.tex` | Top TF-IDF terms per persona (LaTeX) |

The scripts also print full statistical summaries to the console, including chi-squared tests, Cramer's V, cross-model agreement rates, per-model ablation dropout, JSD from baseline, and sensitivity analyses.

### 2. Re-run Inference

To reproduce inference from scratch (requires API keys and/or local model servers):

```bash
# Run all five models
python run_inference.py \
    --image_dir ./personabeam/images \
    --models gpt55 gemini claude qwen gemma \
    --sample_per_env 200 \
    --output_dir ./outputs

# Run a single model
python run_inference.py \
    --image_dir ./personabeam/images \
    --models claude \
    --sample_per_env 200

# Resume interrupted run
python run_inference.py \
    --image_dir ./personabeam/images \
    --models gemma \
    --vllm_url_gemma http://localhost:11434/v1 \
    --resume
```

Each model produces a CSV file in the output directory (`results_<model>.csv`). These can then be concatenated and passed to `run_analysis.py`.

## Models Evaluated

| Model | Type | Backend | Config |
|-------|------|---------|--------|
| GPT-5.5 | Closed | OpenAI API | `temperature=0.0`, `reasoning_effort="none"` |
| Gemini 3.1 Pro | Closed | Vertex AI | `temperature=0.0`, thinking enabled (`thinking_level="LOW"`) |
| Claude Opus 4.7 | Closed | Anthropic API | `temperature=0.0`, thinking disabled |
| Qwen3.6-35B-A3B | Open | vLLM (local) | `temperature=0.0`, thinking disabled; MoE, 35B total / 3B active |
| Gemma 4 31B | Open | Ollama (local) | `temperature=0.0`; dense 31B |

## Personas

Three personas grounded in the BIS/BAS framework (Gray, 1982):

- **Eager Companion** — BAS-dominant, approach-oriented
- **Cautious Observer** — BIS-dominant, avoidance-oriented
- **Indifferent Explorer** — balanced, neutral

See `run_inference.py` for the exact prompt templates.

## Expected Outputs

When run on the full dataset (757 images x 3 personas x 5 models = 11,328 valid samples), the analysis script should reproduce:

- Pooled Cramer's V ~ 0.668
- Per-model V range: 0.638 (Claude) to 0.691 (Qwen)
- Unanimous agreement: ~60.4%
- Explorer agreement: ~42.2%

With ablation data (included in the dataset):

- No-persona ablation: 969 valid baseline responses (N=1000 expected; Qwen accounts for all 31 missing)
- Baseline-relative Cramer's V: Companion V=0.854, Observer V=0.474, Explorer V=0.250
- Noise-image ablation: 651 valid responses (N=750 expected; Gemini 36% valid rate)
- Mean JSD (noise vs real): 0.528 (all models), 0.514 (excluding Gemini)

## License

Code: MIT. See [LICENSE](LICENSE).
Dataset: CC-BY-4.0. See the [Hugging Face repository](https://huggingface.co/datasets/qzkiyoshi/personabeam).
