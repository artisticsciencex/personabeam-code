# PersonaBEAM: Code for Cross-Model Persona Conditioning Evaluation

This repository contains the inference and analysis code for the PersonaBEAM benchmark, which evaluates persona conditioning effects in vision-language model (VLM)-based embodied agent control.

## Repository Structure

```
personabeam-code/
├── README.md               # this file
├── requirements.txt        # Python dependencies
├── LICENSE                 # CC-BY-4.0
├── run_inference.py        # cross-model inference pipeline
├── run_analysis.py         # reproduces all paper figures and tables
└── outputs/                # default output directory
```

## Setup

```bash
# Clone the repository
git clone <this-repo-url>
cd personabeam-code

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
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

Download the dataset from the companion Hugging Face repository, then:

```bash
python run_analysis.py \
    --data ./personabeam/data/responses.parquet \
    --output_dir ./outputs
```

This generates:

| Output | Description |
|--------|-------------|
| `fig1_command_by_persona_model.pdf` | Command distribution heatmaps (Fig 1) |
| `fig2_cramers_v_env_model.pdf` | Cramer's V by environment x model (Fig 2) |
| `fig3_cross_model_agreement.pdf` | Agreement rates + pairwise matrix (Fig 3) |
| `fig4_explorer_by_environment.pdf` | Explorer heatmap by env x model (Fig 4) |
| `fig_latency_appendix.pdf` | Latency violin plots (Appendix) |
| `tab_command_distribution.tex` | Command distribution table (LaTeX) |
| `tab_cramers_v.tex` | Cramer's V table (LaTeX) |

The script also prints the full statistical summary (chi-squared tests, Cramer's V, cross-model agreement rates) to the console.

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

| Model | Type | Backend | Notes |
|-------|------|---------|-------|
| GPT-5.5 | Closed | OpenAI API | `reasoning_effort="none"` for determinism |
| Gemini 3.1 Pro | Closed | Vertex AI | `thinking_level="LOW"` |
| Claude Opus 4.7 | Closed | Anthropic API | Thinking off by default |
| Qwen3.6-35B-A3B | Open | vLLM (local) | MoE, 35B total / 3B active |
| Gemma 4 31B | Open | Ollama (local) | Dense 31B |

All models use `temperature=0.0` for reproducibility.

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

## License

CC-BY-4.0. See [LICENSE](LICENSE) for details.
