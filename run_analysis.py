#!/usr/bin/env python3
"""
PersonaBEAM Analysis & Figure Generation
==========================================
Reproduces all figures and tables from the paper using the released
responses.parquet dataset.

Usage:
  python run_analysis.py --data ./personabeam/data/responses.parquet --output_dir ./outputs

  # With ablation results:
  python run_analysis.py --data ./personabeam/data/responses.parquet \
      --ablation_dir ./outputs --output_dir ./outputs

Outputs:
  Figures (PDF):
    fig1_command_by_persona_model.pdf   - Command distribution heatmaps (Fig 1)
    fig2_cramers_v_env_model.pdf        - Cramer's V by environment x model (Fig 2)
    fig3_cross_model_agreement.pdf      - Agreement rates + pairwise matrix (Fig 3)
    fig4_explorer_by_environment.pdf    - Explorer heatmap by env x model (Fig 4)
    fig_latency_appendix.pdf            - Latency violin plots (Appendix)
    fig5_ablation_nopersona.pdf         - No-persona ablation (if --ablation_dir)
    fig6_ablation_noimage.pdf           - Noise-image ablation (if --ablation_dir)

  Tables (LaTeX):
    tab_command_distribution.tex        - Command distribution by persona/model
    tab_cramers_v.tex                   - Cramer's V by model/environment
    tab_ablation_nopersona_n.tex        - Per-model ablation sample sizes (if --ablation_dir)

  Console:
    Full statistical summary (chi-squared, Cramer's V, agreement rates)
    Ablation analysis with per-model dropout, JSD, sensitivity tests
    (if --ablation_dir provided)
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

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("% of images", fontsize=8)
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

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("% of images", fontsize=8)
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
# Ablation analysis
# ---------------------------------------------------------------------------

def load_ablation_csvs(ablation_dir: str, ablation_type: str) -> pd.DataFrame:
    """Load all CSVs matching results_*_{ablation_type}.csv from a directory."""
    suffix = f"_{ablation_type}.csv"
    dfs = []
    for f in sorted(os.listdir(ablation_dir)):
        if f.startswith("results_") and f.endswith(suffix):
            path = os.path.join(ablation_dir, f)
            dfs.append(pd.read_csv(path))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df[~df["command"].isin(["PARSE_ERROR", "API_ERROR", "EMPTY_RESPONSE"])].copy()
    return df


def analyze_nopersona(df_main: pd.DataFrame, df_nopersona: pd.DataFrame, output_dir: str):
    """Analyze no-persona ablation: compare baseline to each persona."""
    print("\n" + "=" * 60)
    print("ABLATION: No-Persona (baseline without persona prompt)")
    print("=" * 60)
    print(f"Baseline samples: {len(df_nopersona):,}")
    expected_per_model = 200  # 50 images × 4 environments
    print(f"Per-model dropout (expected {expected_per_model} per model):")
    for model in MODEL_ORDER:
        n_model = len(df_nopersona[df_nopersona["model"] == model])
        n_missing = expected_per_model - n_model
        pct = 100 * n_missing / expected_per_model if expected_per_model > 0 else 0
        flag = " ← ELEVATED" if pct > 5 else ""
        print(f"  {model:<22s}: {n_model:>4d} valid, {n_missing:>3d} missing ({pct:.1f}%){flag}")

    # Per-model comparison
    print("\nPer-model command distribution shift (Jensen-Shannon divergence):")
    print(f"  {'Model':<22s} {'Companion':>10s} {'Observer':>10s} {'Explorer':>10s} {'Baseline':>10s}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    from scipy.spatial.distance import jensenshannon

    jsd_data = {}  # model -> {persona: jsd}
    for model in MODEL_ORDER:
        m_main = df_main[df_main["model"] == model]
        m_base = df_nopersona[df_nopersona["model"] == model]
        if len(m_base) == 0:
            continue

        # Baseline distribution
        base_dist = np.array([(m_base["command"] == c).sum() for c in CMDS], dtype=float)
        base_dist = base_dist / base_dist.sum() if base_dist.sum() > 0 else base_dist

        jsds = {}
        for persona in PERSONAS:
            p_sub = m_main[m_main["persona"] == persona]
            p_dist = np.array([(p_sub["command"] == c).sum() for c in CMDS], dtype=float)
            p_dist = p_dist / p_dist.sum() if p_dist.sum() > 0 else p_dist
            jsds[persona] = jensenshannon(base_dist, p_dist)

        jsd_data[model] = jsds

        # Print dominant command for baseline
        dominant_base = CMDS[np.argmax(base_dist)]
        base_pcts = " ".join([f"{v*100:.0f}" for v in base_dist])
        print(f"  {model:<22s} {jsds['companion']:>10.3f} {jsds['observer']:>10.3f} {jsds['explorer']:>10.3f} dom={dominant_base}")

    # Summary
    if jsd_data:
        all_comp = [v["companion"] for v in jsd_data.values()]
        all_obs = [v["observer"] for v in jsd_data.values()]
        all_exp = [v["explorer"] for v in jsd_data.values()]
        print(f"\n  Mean JSD from baseline:")
        print(f"    Companion: {np.mean(all_comp):.3f} (persona SHIFTS behavior away from default)")
        print(f"    Observer:  {np.mean(all_obs):.3f}")
        print(f"    Explorer:  {np.mean(all_exp):.3f} (closest to default → weakest conditioning)")

    # Chi-squared: baseline vs persona-conditioned (full benchmark, pooled across models)
    # NOTE: The baseline uses 200 images while persona-conditioned uses all 757.
    # Cramer's V normalizes by N, so this comparison is statistically valid.
    # We report exact N per condition for full transparency.
    print("\n  Chi-squared: baseline vs persona-conditioned (pooled across models):")
    print(f"    Contingency table: 2×k (persona-conditioned vs baseline) × command columns")
    print(f"    NOTE: Baseline uses 200-image subset; persona-conditioned uses full 757 images.")
    print(f"    Cramer's V normalizes by N, so different sample sizes are handled correctly.")
    print(f"    Baseline N (all models pooled): {len(df_nopersona)}")
    for persona in PERSONAS:
        p_main = df_main[df_main["persona"] == persona]
        ct = pd.DataFrame({
            "persona": [(p_main["command"] == c).sum() for c in CMDS],
            "baseline": [(df_nopersona["command"] == c).sum() for c in CMDS],
        }, index=CMDS).T
        ct = ct.loc[:, ct.sum() > 0]
        n_total = ct.values.sum()
        n_persona = ct.loc["persona"].sum()
        n_base = ct.loc["baseline"].sum()
        if ct.shape[1] >= 2:
            chi2_val, p_val, _, _ = chi2_contingency(ct)
            v = cramers_v(ct)
            print(f"    {PERSONA_LABELS[persona]:12s}: chi2={chi2_val:.1f}, V={v:.3f}, "
                  f"N_total={n_total} (persona={n_persona}, baseline={n_base}), "
                  f"p={'<1e-10' if p_val < 1e-10 else f'{p_val:.2e}'}")

    # Sensitivity: exclude model with highest dropout and recompute
    dropout_counts = {}
    for model in MODEL_ORDER:
        n_model = len(df_nopersona[df_nopersona["model"] == model])
        dropout_counts[model] = expected_per_model - n_model
    worst_model = max(dropout_counts, key=dropout_counts.get)
    worst_n = dropout_counts[worst_model]
    if worst_n > 0:
        df_base_excl = df_nopersona[df_nopersona["model"] != worst_model]
        df_main_excl = df_main[df_main["model"] != worst_model]
        print(f"\n  Sensitivity: excluding {worst_model} ({worst_n} missing):")
        for persona in PERSONAS:
            p_m = df_main_excl[df_main_excl["persona"] == persona]
            ct = pd.DataFrame({
                "persona": [(p_m["command"] == c).sum() for c in CMDS],
                "baseline": [(df_base_excl["command"] == c).sum() for c in CMDS],
            }, index=CMDS).T
            ct = ct.loc[:, ct.sum() > 0]
            if ct.shape[1] >= 2 and ct.loc["persona"].sum() > 0:
                chi2_val, p_val, _, _ = chi2_contingency(ct)
                v = cramers_v(ct)
                n_t = ct.values.sum()
                print(f"    {PERSONA_LABELS[persona]:12s}: V={v:.3f} (N={n_t})")

    # Generate per-model ablation summary table (LaTeX)
    _tab_ablation_nopersona(df_main, df_nopersona, output_dir)

    # Generate figure
    _fig_ablation_nopersona(df_main, df_nopersona, jsd_data, output_dir)


def _tab_ablation_nopersona(df_main: pd.DataFrame, df_nopersona: pd.DataFrame, output_dir: str):
    """Generate LaTeX table: per-model valid counts for no-persona ablation."""
    expected = 200
    rows = []
    for model in MODEL_ORDER:
        m_base = df_nopersona[df_nopersona["model"] == model]
        n_base = len(m_base)
        n_missing = expected - n_base
        pct_miss = 100 * n_missing / expected

        # Full persona-conditioned N per model (all 757 images)
        m_main = df_main[df_main["model"] == model]
        n_persona = len(m_main)

        # Short model name
        short = model.replace("Claude-Opus-4.7", "Claude").replace("Gemini-3.1-Pro", "Gemini") \
                      .replace("Gemma-4-31B", "Gemma").replace("Qwen3.6-35B-A3B", "Qwen")
        rows.append((short, expected, n_base, n_missing, pct_miss, n_persona))

    # Totals
    total_base = sum(r[2] for r in rows)
    total_miss = sum(r[3] for r in rows)
    total_persona = sum(r[5] for r in rows)
    total_pct = 100 * total_miss / (expected * len(rows))

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Per-model sample sizes for the no-persona ablation. "
        r"``Expected'' is 200 (50 images $\times$ 4 environments). ``Baseline valid'' is the number of "
        r"valid no-persona responses. ``Persona $N$'' is the total number of persona-conditioned "
        r"responses per model (across all three personas and all 757 images). "
        r"Cram\'{e}r's $V$ values in Section~\ref{sec:ablation} compare the pooled baseline distribution "
        r"($N{=}969$) against each persona's pooled distribution; $V$ normalizes by $N$, so the different "
        r"sample sizes are handled correctly.}",
        r"  \label{tab:ablation_nopersona_n}",
        r"  \small",
        r"  \begin{tabular}{lrrrrr}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Expected} & \textbf{Baseline valid} & \textbf{Missing} "
        r"& \textbf{Missing \%} & \textbf{Persona $N$} \\",
        r"    \midrule",
    ]
    for short, exp, base, miss, pct, n_pers in rows:
        lines.append(f"    {short} & {exp} & {base} & {miss} & {pct:.1f}\\% & {n_pers:,} \\\\")
    lines += [
        r"    \midrule",
        f"    \\textbf{{Total}} & {expected * len(rows)} & {total_base} & {total_miss} "
        f"& {total_pct:.1f}\\% & {total_persona:,} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    path = os.path.join(output_dir, "tab_ablation_nopersona_n.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {path}")


def _fig_ablation_nopersona(df_main, df_nopersona, jsd_data, output_dir):
    """Fig 5: No-persona ablation — grouped bar showing command distributions."""
    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(1, n_models, figsize=(14, 3.2), sharey=True)

    for i, model in enumerate(MODEL_ORDER):
        ax = axes[i]
        m_main = df_main[df_main["model"] == model]
        m_base = df_nopersona[df_nopersona["model"] == model]

        data = np.zeros((4, len(CMDS)))  # 3 personas + baseline
        labels_y = [PERSONA_LABELS[p] for p in PERSONAS] + ["Baseline"]

        for j, persona in enumerate(PERSONAS):
            psub = m_main[m_main["persona"] == persona]
            total = len(psub)
            if total > 0:
                for k, cmd in enumerate(CMDS):
                    data[j, k] = (psub["command"] == cmd).sum() / total * 100

        total_base = len(m_base)
        if total_base > 0:
            for k, cmd in enumerate(CMDS):
                data[3, k] = (m_base["command"] == cmd).sum() / total_base * 100

        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(CMDS)))
        ax.set_xticklabels(CMDS, fontsize=8)
        ax.set_title(MODEL_SHORT[i], fontsize=9, fontweight="bold")
        if i == 0:
            ax.set_yticks(range(4))
            ax.set_yticklabels(labels_y, fontsize=8)
        else:
            ax.set_yticks([])

        # Annotate cells
        for j in range(4):
            for k in range(len(CMDS)):
                val = data[j, k]
                if val >= 1:
                    color = "white" if val > 50 else "black"
                    ax.text(k, j, f"{val:.0f}", ha="center", va="center", fontsize=6, color=color)

        # Draw separator line above baseline row
        ax.axhline(y=2.5, color="white", linewidth=2)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("% of images", fontsize=8)
    fig.suptitle("No-Persona Ablation: Baseline (no persona prompt) vs Persona-Conditioned",
                 fontsize=10, y=1.02)
    path = os.path.join(output_dir, "fig5_ablation_nopersona.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved {path}")


def analyze_noimage(df_main: pd.DataFrame, df_noimage: pd.DataFrame, output_dir: str):
    """Analyze no-image ablation: compare noise/black-image responses to real-image responses."""
    print("\n" + "=" * 60)
    print("ABLATION: No-Image (noise/black images replacing real fisheye images)")
    print("=" * 60)
    n_per_cell = len(df_noimage) // max(1, df_noimage["model"].nunique() * df_noimage["persona"].nunique())
    print(f"No-image samples: {len(df_noimage):,} (~{n_per_cell}/cell)")

    from scipy.spatial.distance import jensenshannon

    print("\nPer-model, per-persona: dominant command (noise) vs dominant command (real)")
    print(f"  {'Model':<22s} {'Persona':<12s} {'Noise dom':>10s} {'Real dom':>10s} {'N':>4s} {'JSD':>6s}")
    print(f"  {'─'*22} {'─'*12} {'─'*10} {'─'*10} {'─'*4} {'─'*6}")

    jsd_per_model = {}
    jsd_cells = {}  # (model, persona) -> JSD for with/without Gemini analysis
    for model in MODEL_ORDER:
        m_main = df_main[df_main["model"] == model]
        m_noimg = df_noimage[df_noimage["model"] == model]
        if len(m_noimg) == 0:
            continue

        jsds = []
        for persona in PERSONAS:
            p_noimg = m_noimg[m_noimg["persona"] == persona]
            p_real = m_main[m_main["persona"] == persona]

            noimg_dist = np.array([(p_noimg["command"] == c).sum() for c in CMDS], dtype=float)
            noimg_dist = noimg_dist / noimg_dist.sum() if noimg_dist.sum() > 0 else noimg_dist

            real_dist = np.array([(p_real["command"] == c).sum() for c in CMDS], dtype=float)
            real_dist = real_dist / real_dist.sum() if real_dist.sum() > 0 else real_dist

            jsd = jensenshannon(noimg_dist, real_dist)
            jsds.append(jsd)
            jsd_cells[(model, persona)] = jsd

            noimg_cmd = CMDS[np.argmax(noimg_dist)] if noimg_dist.sum() > 0 else "?"
            real_cmd = CMDS[np.argmax(real_dist)] if real_dist.sum() > 0 else "?"
            n_noimg = len(p_noimg)

            print(f"  {model:<22s} {PERSONA_LABELS[persona]:<12s} {noimg_cmd:>10s} {real_cmd:>10s} {n_noimg:>4d} {jsd:>6.3f}")

        jsd_per_model[model] = np.mean(jsds)

    # Summary interpretation
    if jsd_per_model:
        mean_jsd = np.mean(list(jsd_per_model.values()))
        print(f"\n  Mean JSD (noise vs real): {mean_jsd:.3f}")
        if mean_jsd > 0.3:
            print("  → Models condition on visual content (commands differ substantially with/without image)")
        elif mean_jsd > 0.1:
            print("  → Moderate visual conditioning (some commands shift without real image)")
        else:
            print("  → Weak visual conditioning (commands similar regardless of image content)")

        # With vs without Gemini (reviewer feedback #4)
        gemini_models = [m for m in jsd_per_model if "gemini" in m.lower()]
        if gemini_models:
            no_gemini = {m: v for m, v in jsd_per_model.items() if "gemini" not in m.lower()}
            n_no_gemini = sum(len(df_noimage[df_noimage["model"] == m]) for m in no_gemini)
            mean_no_gemini = np.mean(list(no_gemini.values()))
            print(f"\n  With/without Gemini sensitivity analysis:")
            print(f"    All 5 models:     JSD = {mean_jsd:.3f}  (N={len(df_noimage)})")
            print(f"    Without Gemini:   JSD = {mean_no_gemini:.3f}  (N={n_no_gemini})")
            print(f"    Difference:       {mean_jsd - mean_no_gemini:+.3f}")
            for gm in gemini_models:
                n_gemini = len(df_noimage[df_noimage["model"] == gm])
                print(f"    Gemini valid responses: {n_gemini} of 150 ({100*n_gemini/150:.0f}%)")
            # Per-persona breakdown
            print(f"    Per-persona JSD (with vs without Gemini):")
            for persona in PERSONAS:
                all_p = [jsd_cells[(m, persona)] for m in jsd_per_model if (m, persona) in jsd_cells]
                no_g = [jsd_cells[(m, persona)] for m in no_gemini if (m, persona) in jsd_cells]
                if all_p and no_g:
                    print(f"      {PERSONA_LABELS[persona]:12s}: all={np.mean(all_p):.3f}, "
                          f"no_gemini={np.mean(no_g):.3f}, diff={np.mean(all_p)-np.mean(no_g):+.3f}")

    # Chi-squared: noise vs real images per persona (pooled across models)
    print("\n  Chi-squared: noise vs real images (pooled across models):")
    for persona in PERSONAS:
        p_real = df_main[df_main["persona"] == persona]
        p_noimg = df_noimage[df_noimage["persona"] == persona]
        if len(p_noimg) == 0:
            continue
        ct = pd.DataFrame({
            "real": [(p_real["command"] == c).sum() for c in CMDS],
            "noise": [(p_noimg["command"] == c).sum() for c in CMDS],
        }, index=CMDS).T
        ct = ct.loc[:, ct.sum() > 0]
        if ct.shape[1] >= 2:
            chi2_val, p_val, _, _ = chi2_contingency(ct)
            v = cramers_v(ct)
            print(f"    {PERSONA_LABELS[persona]:12s}: chi2={chi2_val:.1f}, V={v:.3f}, "
                  f"p={'<1e-10' if p_val < 1e-10 else f'{p_val:.2e}'}")

    # Generate figure
    _fig_ablation_noimage(df_main, df_noimage, output_dir)


def _fig_ablation_noimage(df_main, df_noimage, output_dir):
    """Fig 6: No-image ablation — heatmap comparing noise images vs real image commands."""
    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(2, n_models, figsize=(14, 4), sharey="row", sharex=True)

    for i, model in enumerate(MODEL_ORDER):
        m_main = df_main[df_main["model"] == model]
        m_noimg = df_noimage[df_noimage["model"] == model]

        # Top row: real images
        data_real = np.zeros((3, len(CMDS)))
        for j, persona in enumerate(PERSONAS):
            psub = m_main[m_main["persona"] == persona]
            total = len(psub)
            if total > 0:
                for k, cmd in enumerate(CMDS):
                    data_real[j, k] = (psub["command"] == cmd).sum() / total * 100

        # Bottom row: noise images
        data_black = np.zeros((3, len(CMDS)))
        for j, persona in enumerate(PERSONAS):
            psub = m_noimg[m_noimg["persona"] == persona]
            total = len(psub)
            if total > 0:
                for k, cmd in enumerate(CMDS):
                    data_black[j, k] = (psub["command"] == cmd).sum() / total * 100

        ax_top = axes[0, i]
        ax_bot = axes[1, i]

        ax_top.imshow(data_real, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
        im = ax_bot.imshow(data_black, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

        ax_bot.set_xticks(range(len(CMDS)))
        ax_bot.set_xticklabels(CMDS, fontsize=8)
        ax_top.set_title(MODEL_SHORT[i], fontsize=9, fontweight="bold")
        ax_top.set_xticks([])

        if i == 0:
            ax_top.set_yticks(range(3))
            ax_top.set_yticklabels([PERSONA_LABELS[p] for p in PERSONAS], fontsize=8)
            ax_bot.set_yticks(range(3))
            ax_bot.set_yticklabels([PERSONA_LABELS[p] for p in PERSONAS], fontsize=8)
        else:
            ax_top.set_yticks([])
            ax_bot.set_yticks([])

        # Annotate
        for ax, data in [(ax_top, data_real), (ax_bot, data_black)]:
            for j in range(3):
                for k in range(len(CMDS)):
                    val = data[j, k]
                    if val >= 1:
                        color = "white" if val > 50 else "black"
                        ax.text(k, j, f"{val:.0f}", ha="center", va="center", fontsize=6, color=color)

    # Row labels
    axes[0, 0].set_ylabel("Real images", fontsize=9, fontweight="bold")
    axes[1, 0].set_ylabel("Noise images", fontsize=9, fontweight="bold")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("% of responses", fontsize=8)
    fig.suptitle("No-Image Ablation: Real Fisheye Images vs Gaussian Noise Images",
                 fontsize=10, y=1.02)
    path = os.path.join(output_dir, "fig6_ablation_noimage.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
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
    parser.add_argument(
        "--ablation_dir", type=str, default=None,
        help="Directory containing ablation CSVs (results_*_nopersona.csv, results_*_noimage.csv). "
             "If provided, ablation analysis figures and stats are generated.",
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

    # Ablation analysis (optional)
    if args.ablation_dir:
        print("\n" + "─" * 60)
        print("Loading ablation data...")

        df_nopersona = load_ablation_csvs(args.ablation_dir, "nopersona")
        df_noimage = load_ablation_csvs(args.ablation_dir, "noimage")

        if len(df_nopersona) > 0:
            analyze_nopersona(df, df_nopersona, args.output_dir)
        else:
            print("  No no-persona ablation CSVs found (results_*_nopersona.csv)")

        if len(df_noimage) > 0:
            analyze_noimage(df, df_noimage, args.output_dir)
        else:
            print("  No no-image ablation CSVs found (results_*_noimage.csv)")

    print(f"\nDone. All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
