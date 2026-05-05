#!/usr/bin/env python3
"""
PersonaBEAM Analysis & Figure Generation
==========================================
Reproduces all figures and tables from the paper using the released
responses.parquet dataset.

Usage:
  python run_analysis.py --data ./personabeam/data/responses.parquet --output_dir ./outputs

Outputs:
  Figures (PDF):
    fig1_command_by_persona_model.pdf   - Command distribution heatmaps (Fig 1)
    fig2_cramers_v_env_model.pdf        - Cramer's V by environment x model (Fig 2)
    fig3_cross_model_agreement.pdf      - Agreement rates + pairwise matrix (Fig 3)
    fig4_explorer_by_environment.pdf    - Explorer heatmap by env x model (Fig 4)
    fig_latency_appendix.pdf            - Latency violin plots (Appendix)

  Tables (LaTeX):
    tab_command_distribution.tex        - Command distribution by persona/model
    tab_cramers_v.tex                   - Cramer's V by model/environment

  Console:
    Full statistical summary (chi-squared, Cramer's V, agreement rates)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    "Claude-Opus-4.7", "GPT-5.5", "Gemini-3.1-Pro",
    "Gemma-4-31B", "Qwen3.6-35B-A3B",
]
MODEL_SHORT = ["Claude", "GPT-5.5", "Gemini", "Gemma", "Qwen"]

ENV_ORDER = ["Auditorium", "Inst. Hallway", "Furn. Lounge", "Domestic Room"]
ENV_MAP = {
    "auditorium": "Auditorium",
    "institutional_hallway": "Inst. Hallway",
    "furnished_lounge": "Furn. Lounge",
    "domestic_room": "Domestic Room",
}

PERSONAS = ["companion", "observer", "explorer"]
PERSONA_LABELS = {"companion": "Companion", "observer": "Observer", "explorer": "Explorer"}
CMDS = ["F", "R", "L", "T", "U", "D", "S"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def cramers_v(ct: pd.DataFrame) -> float:
    """Compute Cramer's V from a contingency table, dropping zero-sum columns."""
    ct = ct.loc[:, ct.sum() > 0]
    if ct.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, c = ct.shape
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))


def compute_stats(df: pd.DataFrame):
    """Print full statistical summary to console."""
    print("\n" + "=" * 60)
    print("PersonaBEAM Statistical Analysis")
    print("=" * 60)
    print(f"Total valid samples: N = {len(df):,}")
    print(f"Models: {len(df['model'].unique())}")
    print(f"Environments: {len(df['environment'].unique())}")
    print(f"Images: {df['image_path'].nunique()}")

    # Pooled chi-squared
    ct_pooled = pd.crosstab(df["persona"], df["command"])
    chi2_val, p_val, dof, _ = chi2_contingency(ct_pooled)
    v_pooled = cramers_v(ct_pooled)
    print(f"\nPooled: chi2 = {chi2_val:.1f}, V = {v_pooled:.3f}, p < 1e-300")

    # Per-model
    print("\nPer-model Cramer's V:")
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        ct = pd.crosstab(sub["persona"], sub["command"])
        chi2_val, p_val, dof, _ = chi2_contingency(ct)
        v = cramers_v(ct)
        print(f"  {model:20s}: V = {v:.3f}, chi2 = {chi2_val:.1f}, N = {len(sub)}")

    # Cross-model agreement
    print("\nCross-model agreement:")
    pivoted = df.pivot_table(
        index=["image_path", "persona"], columns="model",
        values="command", aggfunc="first",
    )
    n_images = len(pivoted)

    # Unanimous (all 5 agree)
    unanimous = (pivoted.nunique(axis=1) == 1).sum()
    print(f"  Unanimous (5/5): {unanimous}/{n_images} ({unanimous/n_images*100:.1f}%)")

    # Majority (>=3 agree)
    from collections import Counter
    majority = 0
    for _, row in pivoted.iterrows():
        counts = Counter(row.dropna())
        if counts.most_common(1)[0][1] >= 3:
            majority += 1
    print(f"  Majority  (>=3):  {majority}/{n_images} ({majority/n_images*100:.1f}%)")

    # Per-persona unanimous
    for persona in PERSONAS:
        p_df = pivoted.loc[pivoted.index.get_level_values("persona") == persona]
        una = (p_df.nunique(axis=1) == 1).sum()
        print(f"  {PERSONA_LABELS[persona]:12s} unanimous: {una}/{len(p_df)} ({una/len(p_df)*100:.1f}%)")

    # Pairwise agreement
    print("\nPairwise agreement (%):")
    for i, m1 in enumerate(MODEL_ORDER):
        for j, m2 in enumerate(MODEL_ORDER):
            if j <= i:
                continue
            agree = (pivoted[m1] == pivoted[m2]).sum()
            pct = agree / n_images * 100
            print(f"  {m1} vs {m2}: {pct:.1f}%")


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def fig1_command_heatmap(df: pd.DataFrame, output_dir: str):
    """Fig 1: Command distribution heatmaps by persona for each model."""
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
    for i, model in enumerate(MODEL_ORDER):
        ax = axes[i]
        sub = df[df["model"] == model]
        data = np.zeros((3, len(CMDS)))
        for j, persona in enumerate(PERSONAS):
            psub = sub[sub["persona"] == persona]
            total = len(psub)
            if total > 0:
                for k, cmd in enumerate(CMDS):
                    data[j, k] = (psub["command"] == cmd).sum() / total * 100
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(CMDS)))
        ax.set_xticklabels(CMDS, fontsize=8)
        ax.set_title(MODEL_SHORT[i], fontsize=9, fontweight="bold")
        if i == 0:
            ax.set_yticks(range(3))
            ax.set_yticklabels([PERSONA_LABELS[p] for p in PERSONAS], fontsize=8)
        else:
            ax.set_yticks([])
        for j in range(3):
            for k in range(len(CMDS)):
                val = data[j, k]
                if val >= 1:
                    color = "white" if val > 50 else "black"
                    ax.text(k, j, f"{val:.0f}", ha="center", va="center", fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("% of images", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig1_command_by_persona_model.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved {path}")


def fig2_cramers_v(df: pd.DataFrame, output_dir: str):
    """Fig 2: Cramer's V by environment and model (grouped bar chart)."""
    results = []
    for model in MODEL_ORDER:
        for env in ENV_ORDER:
            env_key = [k for k, v in ENV_MAP.items() if v == env][0]
            sub = df[(df["model"] == model) & (df["environment"] == env_key)]
            ct = pd.crosstab(sub["persona"], sub["command"])
            v = cramers_v(ct)
            results.append({"Model": model, "Environment": env, "V": v})
    vdf = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(ENV_ORDER))
    width = 0.14
    for i, model in enumerate(MODEL_ORDER):
        vals = [
            vdf[(vdf["Model"] == model) & (vdf["Environment"] == env)]["V"].values[0]
            for env in ENV_ORDER
        ]
        ax.bar(x + (i - 2) * width, vals, width, label=MODEL_SHORT[i], color=COLORS[i])
    ax.set_ylabel("Cramer's V", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(ENV_ORDER, fontsize=9)
    ax.set_ylim(0.5, 0.85)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(3.6, 0.505, "Large effect\nthreshold", fontsize=7, color="gray", ha="right")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig2_cramers_v_env_model.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved {path}")


def fig3_agreement(df: pd.DataFrame, output_dir: str):
    """Fig 3: Cross-model agreement (bar chart + pairwise heatmap)."""
    from collections import Counter

    pivoted = df.pivot_table(
        index=["image_path", "persona"], columns="model",
        values="command", aggfunc="first",
    )
    n_total = len(pivoted)

    # Unanimous overall
    una_all = (pivoted.nunique(axis=1) == 1).sum() / n_total * 100

    # Per-persona unanimous
    persona_una = {}
    for persona in PERSONAS:
        p_df = pivoted.loc[pivoted.index.get_level_values("persona") == persona]
        una = (p_df.nunique(axis=1) == 1).sum() / len(p_df) * 100
        persona_una[persona] = una

    # Pairwise agreement matrix
    n_models = len(MODEL_ORDER)
    pairwise = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                pairwise[i, j] = 100.0
            else:
                agree = (pivoted[MODEL_ORDER[i]] == pivoted[MODEL_ORDER[j]]).sum()
                pairwise[i, j] = agree / n_total * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1, 1.3]})

    # Left: bar chart
    labels = ["All"] + [PERSONA_LABELS[p] for p in PERSONAS]
    values = [una_all] + [persona_una[p] for p in PERSONAS]
    bar_colors = ["#333333", COLORS[0], COLORS[1], COLORS[2]]
    bars = ax1.barh(labels, values, color=bar_colors, height=0.6)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Unanimous agreement (%)", fontsize=9)
    ax1.invert_yaxis()
    for bar, val in zip(bars, values):
        ax1.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                 va="center", fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: pairwise heatmap
    im = ax2.imshow(pairwise, cmap="Blues", vmin=60, vmax=100)
    ax2.set_xticks(range(n_models))
    ax2.set_xticklabels(MODEL_SHORT, fontsize=8, rotation=45, ha="right")
    ax2.set_yticks(range(n_models))
    ax2.set_yticklabels(MODEL_SHORT, fontsize=8)
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                ax2.text(j, i, f"{pairwise[i, j]:.1f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label("Pairwise agreement (%)", fontsize=8)
    ax2.set_title("Pairwise agreement", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_cross_model_agreement.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved {path}")


def fig4_explorer_heatmap(df: pd.DataFrame, output_dir: str):
    """Fig 4: Explorer command distribution by environment x model."""
    explorer = df[df["persona"] == "explorer"]
    explorer = explorer.copy()
    explorer["env_label"] = explorer["environment"].map(ENV_MAP)

    fig, axes = plt.subplots(1, 5, figsize=(12, 2.8), sharey=True)
    for i, model in enumerate(MODEL_ORDER):
        ax = axes[i]
        data = np.zeros((len(ENV_ORDER), len(CMDS)))
        for j, env in enumerate(ENV_ORDER):
            sub = explorer[(explorer["model"] == model) & (explorer["env_label"] == env)]
            total = len(sub)
            if total > 0:
                for k, cmd in enumerate(CMDS):
                    data[j, k] = (sub["command"] == cmd).sum() / total * 100
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(CMDS)))
        ax.set_xticklabels(CMDS, fontsize=8)
        ax.set_title(MODEL_SHORT[i], fontsize=9, fontweight="bold")
        if i == 0:
            ax.set_yticks(range(len(ENV_ORDER)))
            ax.set_yticklabels(ENV_ORDER, fontsize=8)
        else:
            ax.set_yticks([])
        for j in range(len(ENV_ORDER)):
            for k in range(len(CMDS)):
                val = data[j, k]
                if val >= 1:
                    color = "white" if val > 50 else "black"
                    ax.text(k, j, f"{val:.0f}", ha="center", va="center", fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("% of images", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_explorer_by_environment.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved {path}")


def fig_latency(df: pd.DataFrame, output_dir: str):
    """Appendix: Latency violin + box plots per model."""
    df_lat = df.copy()
    df_lat["latency_s"] = df_lat["latency_ms"] / 1000.0

    # Sort by median latency
    medians = df_lat.groupby("model")["latency_s"].median().sort_values()
    model_order_lat = list(medians.index)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    data_list = [df_lat[df_lat["model"] == m]["latency_s"].clip(upper=25).values for m in model_order_lat]
    short_names = [MODEL_SHORT[MODEL_ORDER.index(m)] for m in model_order_lat]

    parts = ax.violinplot(data_list, positions=range(len(model_order_lat)),
                          showmeans=False, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.3)

    bp = ax.boxplot(data_list, positions=range(len(model_order_lat)),
                    widths=0.15, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.7),
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(color="#333"),
                    capprops=dict(color="#333"),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))

    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(len(model_order_lat) - 0.5, 2.1, "2s control budget", fontsize=7, color="red", ha="right")

    ax.set_xticks(range(len(model_order_lat)))
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel("Latency (s)", fontsize=10)
    ax.set_ylim(0, 25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig_latency_appendix.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Table generation (LaTeX)
# ---------------------------------------------------------------------------

def tab_command_distribution(df: pd.DataFrame, output_dir: str):
    """Generate LaTeX table of command distribution (%) by persona and model."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Command distribution (\%) by persona and model. Dominant commands highlighted in bold.}",
        r"\label{tab:command_dist}",
        r"\small",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Model & Persona & F & R & L & T & U & D & S \\",
        r"\midrule",
    ]

    for i, model in enumerate(MODEL_ORDER):
        sub = df[df["model"] == model]
        for j, persona in enumerate(PERSONAS):
            psub = sub[sub["persona"] == persona]
            total = len(psub)
            if total == 0:
                continue
            pcts = {}
            for cmd in CMDS:
                pcts[cmd] = round((psub["command"] == cmd).sum() / total * 100)
            max_cmd = max(pcts, key=pcts.get)
            vals = []
            for cmd in CMDS:
                v = pcts[cmd]
                if cmd == max_cmd:
                    vals.append(f"\\textbf{{{v}}}")
                else:
                    vals.append(str(v))
            prefix = model if j == 0 else ""
            lines.append(f"{prefix} & {PERSONA_LABELS[persona]} & {' & '.join(vals)} \\\\")
        if i < len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path = os.path.join(output_dir, "tab_command_distribution.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {path}")


def tab_cramers_v(df: pd.DataFrame, output_dir: str):
    """Generate LaTeX table of Cramer's V by model and environment."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Persona effect strength (Cram\'er's V) by model and environment.}",
        r"\label{tab:cramers_v}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Auditorium & Inst.\ Hallway & Domestic Room & Furn.\ Lounge & Mean \\",
        r"\midrule",
    ]

    env_keys = ["auditorium", "institutional_hallway", "domestic_room", "furnished_lounge"]
    for model in MODEL_ORDER:
        vals = []
        for env_key in env_keys:
            sub = df[(df["model"] == model) & (df["environment"] == env_key)]
            ct = pd.crosstab(sub["persona"], sub["command"])
            v = cramers_v(ct)
            vals.append(v)
        mean_v = np.mean(vals)
        val_strs = [f"{v:.3f}" for v in vals] + [f"{mean_v:.3f}"]
        lines.append(f"{model} & {' & '.join(val_strs)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path = os.path.join(output_dir, "tab_cramers_v.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PersonaBEAM analysis: reproduce all paper figures and tables",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to responses.parquet (or responses.csv)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory for generated figures and tables (default: ./outputs)",
    )
    args = parser.parse_args()

    # Load data
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    # Filter out any error rows
    df = df[~df["command"].isin(["PARSE_ERROR", "API_ERROR", "EMPTY_RESPONSE"])].copy()

    print(f"Loaded {len(df):,} valid samples from {args.data}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Environments: {sorted(df['environment'].unique())}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Statistics
    compute_stats(df)

    # Figures
    print("\nGenerating figures...")
    fig1_command_heatmap(df, args.output_dir)
    fig2_cramers_v(df, args.output_dir)
    fig3_agreement(df, args.output_dir)
    fig4_explorer_heatmap(df, args.output_dir)
    fig_latency(df, args.output_dir)

    # Tables
    print("\nGenerating LaTeX tables...")
    tab_command_distribution(df, args.output_dir)
    tab_cramers_v(df, args.output_dir)

    print(f"\nDone. All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
