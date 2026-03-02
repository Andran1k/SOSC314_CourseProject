from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from text_model_utils import (
    evaluate_binary_model_cv,
    evaluate_dummy_cv,
    fit_text_model_with_vocab,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "trending_music_processed.csv"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3
TOP_N_TERMS = 20


def _as_text_series(series: pd.Series) -> list[str]:
    return series.fillna("").astype(str).tolist()


def summarize_result(out: dict) -> dict:
    return {
        "auc_mean": out["auc_mean"],
        "auc_sd": out["auc_sd"],
        "acc_mean": out["acc_mean"],
        "acc_sd": out["acc_sd"],
        "n_folds": out["n_folds"],
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X_promo = _as_text_series(df["tokens_joined_promo"])
    X_semantic = _as_text_series(df["tokens_joined_semantic"])
    y_median = df["high_views"].astype(int).to_numpy()
    q75 = float(df["view_count"].quantile(0.75))
    y_top_quartile = (df["view_count"] > q75).astype(int).to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)

    results = []

    def add_result(task: str, diagnostic: str, config: str, out: dict, notes: str = ""):
        row = {
            "task": task,
            "diagnostic": diagnostic,
            "config": config,
            **summarize_result(out),
            "notes": notes,
        }
        results.append(row)

    out_dummy = evaluate_dummy_cv(
        y_median,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    add_result("median_split", "sanity_baseline", "Dummy (most_frequent)", out_dummy, "Majority-class baseline.")

    y_shuf = rng.permutation(y_median)
    out_shuf = evaluate_binary_model_cv(
        X_promo,
        y_shuf,
        vectorizer_kind="count",
        ngram_range=(1, 2),
        min_df=2,
        c=1.0,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    add_result(
        "median_split",
        "sanity_label_shuffle",
        "Shuffled labels (BoW Count, min_df=2)",
        out_shuf,
        "Should be close to 0.50 AUC if the pipeline is not leaking.",
    )

    for texts, label in [(X_promo, "promo"), (X_semantic, "semantic")]:
        out = evaluate_binary_model_cv(
            texts,
            y_median,
            vectorizer_kind="count",
            ngram_range=(1, 2),
            min_df=2,
            c=1.0,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
        )
        add_result("median_split", "representation", f"Count (min_df=2) | {label}", out)

    for min_df in [1, 2, 5, 10]:
        out = evaluate_binary_model_cv(
            X_promo,
            y_median,
            vectorizer_kind="count",
            ngram_range=(1, 2),
            min_df=min_df,
            c=1.0,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
        )
        add_result("median_split", "min_df", f"Count | min_df={min_df}", out)

    out_count = evaluate_binary_model_cv(
        X_promo,
        y_median,
        vectorizer_kind="count",
        ngram_range=(1, 2),
        min_df=2,
        c=1.0,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    add_result("median_split", "vectorizer", "Count (min_df=2)", out_count)

    out_tfidf = evaluate_binary_model_cv(
        X_promo,
        y_median,
        vectorizer_kind="tfidf",
        ngram_range=(1, 2),
        min_df=2,
        c=1.0,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    add_result("median_split", "vectorizer", "TFIDF (min_df=2)", out_tfidf)

    best_c = None
    best_auc = -1.0
    for c in [0.1, 0.5, 1.0, 5.0, 10.0]:
        out = evaluate_binary_model_cv(
            X_promo,
            y_median,
            vectorizer_kind="tfidf",
            ngram_range=(1, 2),
            min_df=2,
            c=c,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
        )
        if out["auc_mean"] > best_auc:
            best_auc = out["auc_mean"]
            best_c = c
    add_result(
        "median_split",
        "tfidf_tuning",
        f"Best TFIDF C={best_c} (min_df=2)",
        {"auc_mean": best_auc, "auc_sd": np.nan, "acc_mean": np.nan, "acc_sd": np.nan, "n_folds": N_SPLITS * N_REPEATS},
        "Best repeated-CV AUC across the tested C grid.",
    )

    out_q75 = evaluate_binary_model_cv(
        X_promo,
        y_top_quartile,
        vectorizer_kind="tfidf",
        ngram_range=(1, 2),
        min_df=2,
        c=best_c or 1.0,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    add_result(
        "top_quartile",
        "label_choice",
        "TFIDF (min_df=2) | promo",
        out_q75,
        f"high_views = 1 if view_count > 75th percentile ({q75:.0f}).",
    )

    results_df = pd.DataFrame(results)
    out_table = PROJECT_ROOT / "data" / "processed" / "diagnostics_week5_results_cv.csv"
    results_df.to_csv(out_table, index=False)

    vec, model = fit_text_model_with_vocab(
        X_promo,
        y_median,
        vectorizer_kind="tfidf",
        ngram_range=(1, 2),
        min_df=2,
        c=best_c or 1.0,
    )
    coef = model.coef_
    vocab = np.array(vec.vocabulary_)
    top_pos_idx = np.argsort(coef)[-TOP_N_TERMS:][::-1]
    top_neg_idx = np.argsort(coef)[:TOP_N_TERMS]

    rows = []
    for rank, idx in enumerate(top_pos_idx, start=1):
        rows.append({"label": f"promo_tfidf_C={best_c}", "direction": "positive", "rank": rank, "term": vocab[idx], "coef": float(coef[idx])})
    for rank, idx in enumerate(top_neg_idx, start=1):
        rows.append({"label": f"promo_tfidf_C={best_c}", "direction": "negative", "rank": rank, "term": vocab[idx], "coef": float(coef[idx])})

    vec_sem, model_sem = fit_text_model_with_vocab(
        X_semantic,
        y_median,
        vectorizer_kind="tfidf",
        ngram_range=(1, 2),
        min_df=2,
        c=best_c or 1.0,
    )
    coef_sem = model_sem.coef_
    vocab_sem = np.array(vec_sem.vocabulary_)
    top_pos_idx_sem = np.argsort(coef_sem)[-TOP_N_TERMS:][::-1]
    top_neg_idx_sem = np.argsort(coef_sem)[:TOP_N_TERMS]
    for rank, idx in enumerate(top_pos_idx_sem, start=1):
        rows.append({"label": f"semantic_tfidf_C={best_c}", "direction": "positive", "rank": rank, "term": vocab_sem[idx], "coef": float(coef_sem[idx])})
    for rank, idx in enumerate(top_neg_idx_sem, start=1):
        rows.append({"label": f"semantic_tfidf_C={best_c}", "direction": "negative", "rank": rank, "term": vocab_sem[idx], "coef": float(coef_sem[idx])})

    out_terms = PROJECT_ROOT / "data" / "processed" / "diagnostics_week5_top_terms.csv"
    pd.DataFrame(rows).to_csv(out_terms, index=False)

    repr_rows = results_df[(results_df["task"] == "median_split") & (results_df["diagnostic"] == "representation")]
    min_df_rows = results_df[(results_df["task"] == "median_split") & (results_df["diagnostic"] == "min_df")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].boxplot(out_count["fold_aucs"], vert=True, widths=0.5)
    axes[0].set_title("Stability across CV folds\n(Count, promo, min_df=2)")
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks([1])
    axes[0].set_xticklabels([f"{N_SPLITS}x{N_REPEATS} CV"])

    axes[1].bar(
        repr_rows["config"],
        repr_rows["auc_mean"],
        yerr=repr_rows["auc_sd"],
        capsize=4,
        color=["steelblue", "coral"],
        edgecolor="black",
    )
    axes[1].set_title("Promo vs semantic\n(mean +/- sd AUC)")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(
        min_df_rows["config"],
        min_df_rows["auc_mean"],
        yerr=min_df_rows["auc_sd"],
        capsize=4,
        color="steelblue",
        edgecolor="black",
    )
    axes[2].set_title("Sensitivity to min_df\n(mean +/- sd AUC)")
    axes[2].set_ylabel("AUC")
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", rotation=20)

    plt.suptitle("Week 5 diagnostics (CV): predictive performance (high_views)", fontsize=11)
    plt.tight_layout()
    fig_path = FIG_DIR / "diagnostics_week5_auc_cv.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print("=== Week 5 diagnostics (CV) summary ===")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {out_table}")
    print(f"Top terms saved to {out_terms}")
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
