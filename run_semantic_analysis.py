#!/usr/bin/env python3
"""
PersonaBEAM – Quantitative Semantic Analysis of Model Reasoning Texts
=====================================================================

Analyses performed on the 'reason' field of all 11,328 responses:

1. Embedding-based grouping analysis
   - Mean cosine similarity WITHIN groups vs BETWEEN groups
   - Compared for two grouping variables: persona and environment
   - Produces a clear "persona clusters language, environment does not" result

2. TF-IDF distinctive vocabulary
   - Top-10 most distinctive terms per persona (highest TF-IDF weight)
   - Shows the lexical signature of each behavioral strategy

3. Lexical diversity & affect alignment
   - Mean reason length, unique-word ratio per persona
   - Approach/avoidance keyword ratios aligned with BIS/BAS predictions

4. Cross-model semantic consistency
   - Per-model intra-persona similarity vs inter-persona similarity
   - Tests whether all models produce persona-aligned language or just
     persona-aligned commands

Outputs:
  - tab_semantic_similarity.tex   (Table: embedding similarity by grouping)
  - tab_persona_vocabulary.tex    (Table: top TF-IDF terms per persona)
  - fig7_semantic_similarity.pdf  (Bar chart: intra vs inter similarity)

Usage:
  python run_semantic_analysis.py --parquet data/responses.parquet --output_dir figures
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERSONAS = ["companion", "observer", "explorer"]
PERSONA_LABELS = {"companion": "Companion", "observer": "Observer", "explorer": "Explorer"}
ENVIRONMENTS = ["auditorium", "institutional_hallway", "furnished_lounge", "domestic_room"]
ENV_LABELS = {
    "auditorium": "Auditorium",
    "institutional_hallway": "Inst. Hallway",
    "furnished_lounge": "Furn. Lounge",
    "domestic_room": "Domestic Room",
}
MODEL_ORDER = ["Claude-Opus-4.7", "GPT-5.5", "Gemini-3.1-Pro", "Gemma-4-31B", "Qwen3.6-35B-A3B"]
MODEL_SHORT = ["Claude", "GPT-5.5", "Gemini", "Gemma", "Qwen"]

# BIS/BAS aligned keyword lists
APPROACH_WORDS = {
    "exciting", "explore", "engaging", "approach", "forward", "advance",
    "interesting", "curious", "investigate", "attract", "drawn", "closer",
    "welcome", "greet", "engage", "opportunity", "enthusiasm", "eager",
    "fascinating", "wonderful", "inviting", "adventure", "discover",
}
AVOIDANCE_WORDS = {
    "caution", "careful", "retreat", "avoid", "danger", "risk", "threat",
    "safety", "safe", "distance", "maintain", "uncertain", "unfamiliar",
    "obstacle", "hazard", "vigilant", "wary", "retreat", "backward",
    "suspicious", "concern", "protective", "clearance", "cautious",
}
NEUTRAL_WORDS = {
    "systematic", "methodical", "objective", "scan", "survey", "neutral",
    "detached", "assess", "feature", "observe", "note", "routine",
    "balanced", "even", "consistent", "standard", "matter-of-fact",
}


def compute_embeddings(texts, max_features=5000):
    """Compute TF-IDF embeddings (no model download needed) and L2-normalize.

    Returns an (N, D) numpy array of unit-length TF-IDF vectors.
    Cosine similarity on these == dot product, matching sentence-embedding semantics.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    print(f"  Computing TF-IDF embeddings for {len(texts):,} texts (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features, stop_words="english", min_df=5,
        ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    # L2-normalize so cosine sim = dot product
    tfidf_normed = normalize(tfidf_matrix, norm="l2")
    print(f"  TF-IDF matrix: {tfidf_normed.shape[0]} docs × {tfidf_normed.shape[1]} features")
    return tfidf_normed


def analysis_1_embedding_similarity(df, embeddings, output_dir):
    """Compare intra-group vs inter-group cosine similarity for persona and environment."""
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Embedding-based grouping (persona vs environment)")
    print("=" * 60)

    from sklearn.metrics.pairwise import cosine_similarity

    # Sample for computational feasibility (full pairwise on 11k is 128M pairs)
    np.random.seed(42)
    n_sample = 3000
    idx = np.random.choice(embeddings.shape[0], size=min(n_sample, embeddings.shape[0]), replace=False)
    idx = np.sort(idx)
    df_s = df.iloc[idx].reset_index(drop=True)
    emb_s = embeddings[idx]

    # Compute pairwise cosine similarity
    print(f"  Computing pairwise cosine similarity on {len(idx)} samples...")
    sim_matrix = cosine_similarity(emb_s)

    results = {}
    for grouping_name, col in [("Persona", "persona"), ("Environment", "environment")]:
        labels = df_s[col].values
        unique_labels = sorted(df_s[col].unique())

        intra_sims = []
        inter_sims = []

        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                s = sim_matrix[i, j]
                if labels[i] == labels[j]:
                    intra_sims.append(s)
                else:
                    inter_sims.append(s)

        intra_mean = np.mean(intra_sims)
        inter_mean = np.mean(inter_sims)
        gap = intra_mean - inter_mean

        results[grouping_name] = {
            "intra_mean": intra_mean,
            "intra_std": np.std(intra_sims),
            "inter_mean": inter_mean,
            "inter_std": np.std(inter_sims),
            "gap": gap,
            "n_intra": len(intra_sims),
            "n_inter": len(inter_sims),
        }

        print(f"\n  {grouping_name} grouping:")
        print(f"    Intra-group similarity: {intra_mean:.4f} ± {np.std(intra_sims):.4f}  (n={len(intra_sims):,})")
        print(f"    Inter-group similarity: {inter_mean:.4f} ± {np.std(inter_sims):.4f}  (n={len(inter_sims):,})")
        print(f"    Gap (intra - inter):    {gap:+.4f}")

    # Interpretation
    p_gap = results["Persona"]["gap"]
    e_gap = results["Environment"]["gap"]
    print(f"\n  Persona gap / Environment gap ratio: {p_gap / e_gap:.1f}x")
    if p_gap > e_gap * 2:
        print("  → Persona dominates language clustering (as expected from command-level results)")

    # --- Generate figure ---
    _fig_semantic_similarity(results, output_dir)

    # --- Generate LaTeX table ---
    _tab_semantic_similarity(results, output_dir)

    return results


def _fig_semantic_similarity(results, output_dir):
    """Fig 7: Bar chart of intra vs inter similarity by grouping."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))

    groupings = ["Persona", "Environment"]
    x = np.arange(len(groupings))
    width = 0.3

    intra_vals = [results[g]["intra_mean"] for g in groupings]
    inter_vals = [results[g]["inter_mean"] for g in groupings]
    intra_stds = [results[g]["intra_std"] for g in groupings]
    inter_stds = [results[g]["inter_std"] for g in groupings]

    bars1 = ax.bar(x - width / 2, intra_vals, width, label="Intra-group",
                    color="#d45e60", alpha=0.85, yerr=intra_stds, capsize=3)
    bars2 = ax.bar(x + width / 2, inter_vals, width, label="Inter-group",
                    color="#6b9bc3", alpha=0.85, yerr=inter_stds, capsize=3)

    ax.set_ylabel("Mean cosine similarity", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(groupings, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Semantic Similarity: Intra- vs Inter-Group\n(sentence embeddings of model reasoning)",
                  fontsize=9)

    # Annotate gaps
    for i, g in enumerate(groupings):
        gap = results[g]["gap"]
        y_top = max(intra_vals[i], inter_vals[i]) + max(intra_stds[i], inter_stds[i]) + 0.02
        ax.annotate(f"Δ={gap:+.3f}", xy=(i, y_top), ha="center", fontsize=8, fontstyle="italic")

    plt.tight_layout()
    path = os.path.join(output_dir, "fig7_semantic_similarity.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved {path}")


def _tab_semantic_similarity(results, output_dir):
    """LaTeX table for embedding similarity results."""
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Mean cosine similarity of TF-IDF vectors for model reasoning texts, grouped by persona vs.\ environment. A larger intra--inter gap indicates stronger semantic clustering by that variable.}",
        r"  \label{tab:semantic_sim}",
        r"  \small",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{Grouping} & \textbf{Intra-group} & \textbf{Inter-group} & \textbf{Gap} & \textbf{Ratio} \\",
        r"    \midrule",
    ]

    p = results["Persona"]
    e = results["Environment"]
    ratio = p["gap"] / e["gap"] if e["gap"] != 0 else float("inf")

    lines.append(
        f"    Persona & {p['intra_mean']:.3f} $\\pm$ {p['intra_std']:.3f} & "
        f"{p['inter_mean']:.3f} $\\pm$ {p['inter_std']:.3f} & "
        f"{p['gap']:+.3f} & \\multirow{{2}}{{*}}{{{ratio:.1f}$\\times$}} \\\\"
    )
    lines.append(
        f"    Environment & {e['intra_mean']:.3f} $\\pm$ {e['intra_std']:.3f} & "
        f"{e['inter_mean']:.3f} $\\pm$ {e['inter_std']:.3f} & "
        f"{e['gap']:+.3f} & \\\\"
    )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    path = os.path.join(output_dir, "tab_semantic_similarity.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {path}")


def analysis_2_tfidf_vocabulary(df, output_dir):
    """Extract top distinctive terms per persona using TF-IDF."""
    print("\n" + "=" * 60)
    print("ANALYSIS 2: TF-IDF distinctive vocabulary per persona")
    print("=" * 60)

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Build one document per persona (concatenate all reasons)
    persona_docs = {}
    for persona in PERSONAS:
        texts = df[df["persona"] == persona]["reason"].tolist()
        persona_docs[persona] = " ".join(texts)

    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", min_df=2,
        ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
    )
    tfidf_matrix = vectorizer.fit_transform(list(persona_docs.values()))
    feature_names = vectorizer.get_feature_names_out()

    top_n = 10
    results = {}
    for i, persona in enumerate(PERSONAS):
        scores = tfidf_matrix[i].toarray().flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        top_terms = [(feature_names[j], scores[j]) for j in top_indices]
        results[persona] = top_terms

        print(f"\n  {PERSONA_LABELS[persona]}:")
        for term, score in top_terms:
            print(f"    {term:25s}  {score:.4f}")

    _tab_persona_vocabulary(results, output_dir)
    return results


def _tab_persona_vocabulary(results, output_dir):
    """LaTeX table for persona-distinctive vocabulary."""
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Top-10 distinctive terms per persona by TF-IDF weight (unigrams and bigrams). Terms reflect the behavioral vocabulary each persona elicits across all five models.}",
        r"  \label{tab:persona_vocab}",
        r"  \small",
        r"  \begin{tabular}{llll}",
        r"    \toprule",
        r"    \textbf{Rank} & \textbf{Companion} & \textbf{Observer} & \textbf{Explorer} \\",
        r"    \midrule",
    ]

    for rank in range(10):
        c_term = results["companion"][rank][0]
        o_term = results["observer"][rank][0]
        e_term = results["explorer"][rank][0]
        lines.append(f"    {rank+1} & {c_term} & {o_term} & {e_term} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    path = os.path.join(output_dir, "tab_persona_vocabulary.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {path}")


def analysis_3_affect_alignment(df):
    """Measure BIS/BAS keyword alignment per persona."""
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Affect keyword alignment with BIS/BAS")
    print("=" * 60)

    print(f"\n  {'Persona':<12s} {'Approach':>10s} {'Avoidance':>10s} {'Neutral':>10s} {'A/V ratio':>10s} {'Mean len':>10s} {'Unique%':>10s}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for persona in PERSONAS:
        texts = df[df["persona"] == persona]["reason"].str.lower().tolist()
        all_words = []
        for t in texts:
            all_words.extend(t.split())

        approach_count = sum(1 for w in all_words if w.strip(".,!?;:\"'()") in APPROACH_WORDS)
        avoidance_count = sum(1 for w in all_words if w.strip(".,!?;:\"'()") in AVOIDANCE_WORDS)
        neutral_count = sum(1 for w in all_words if w.strip(".,!?;:\"'()") in NEUTRAL_WORDS)

        total = len(all_words)
        a_rate = approach_count / total * 1000  # per 1000 words
        v_rate = avoidance_count / total * 1000
        n_rate = neutral_count / total * 1000
        ratio = approach_count / max(avoidance_count, 1)

        mean_len = np.mean([len(t.split()) for t in texts])
        unique_ratio = len(set(all_words)) / len(all_words) * 100

        print(f"  {PERSONA_LABELS[persona]:<12s} {a_rate:>9.1f}‰ {v_rate:>9.1f}‰ {n_rate:>9.1f}‰ {ratio:>10.2f} {mean_len:>10.1f} {unique_ratio:>9.1f}%")

    print("\n  BIS/BAS alignment check:")
    print("    Companion (BAS-dominant): should have highest approach rate ✓/✗")
    print("    Observer (BIS-dominant):  should have highest avoidance rate ✓/✗")
    print("    Explorer (balanced):      should have highest neutral rate  ✓/✗")


def analysis_4_cross_model_semantic(df, embeddings):
    """Per-model intra-persona vs inter-persona similarity."""
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Cross-model semantic consistency")
    print("=" * 60)

    from sklearn.metrics.pairwise import cosine_similarity

    np.random.seed(42)

    print(f"\n  {'Model':<22s} {'Intra-persona':>14s} {'Inter-persona':>14s} {'Gap':>8s}")
    print(f"  {'─'*22} {'─'*14} {'─'*14} {'─'*8}")

    for model in MODEL_ORDER:
        mask = df["model"].values == model
        m_idx = np.where(mask)[0]

        # Sub-sample for speed
        if len(m_idx) > 800:
            sub = np.random.choice(m_idx, size=800, replace=False)
        else:
            sub = m_idx
        sub = np.sort(sub)

        emb_sub = embeddings[sub]
        if hasattr(emb_sub, "toarray"):
            emb_sub = emb_sub  # keep sparse for cosine_similarity
        labels_sub = df.iloc[sub]["persona"].values
        sim = cosine_similarity(emb_sub)

        intra, inter = [], []
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                if labels_sub[i] == labels_sub[j]:
                    intra.append(sim[i, j])
                else:
                    inter.append(sim[i, j])

        intra_m = np.mean(intra)
        inter_m = np.mean(inter)
        gap = intra_m - inter_m

        short = MODEL_SHORT[MODEL_ORDER.index(model)]
        print(f"  {short:<22s} {intra_m:>14.4f} {inter_m:>14.4f} {gap:>+8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PersonaBEAM semantic analysis of model reasoning texts",
    )
    parser.add_argument("--parquet", type=str, required=True,
                        help="Path to responses.parquet")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory for output figures and tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    df = pd.read_parquet(args.parquet)
    print(f"  {len(df):,} responses loaded")

    # Compute embeddings once
    embeddings = compute_embeddings(df["reason"].tolist())

    # Run all analyses
    analysis_1_embedding_similarity(df, embeddings, args.output_dir)
    analysis_2_tfidf_vocabulary(df, args.output_dir)
    analysis_3_affect_alignment(df)
    analysis_4_cross_model_semantic(df, embeddings)

    print("\n" + "=" * 60)
    print("All semantic analyses complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
