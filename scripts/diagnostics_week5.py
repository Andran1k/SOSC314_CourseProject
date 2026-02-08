"""
Week 5 diagnostics (improved):
- Use repeated stratified cross-validation (instead of a single 75/25 split) to measure stability.
- Add sanity checks: majority-class baseline and label-shuffle AUC (should be ~0.50).
- Compare representations (promo vs semantic), preprocessing (min_df), and vectorizers (Count vs TF–IDF).
- Tune logistic regression regularization (C) with CV for TF–IDF.
- Save top-weighted terms to support interpretation.

Outputs
- data/processed/diagnostics_week5_results_cv.csv
- data/processed/diagnostics_week5_top_terms.csv
- figures/diagnostics_week5_auc_cv.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "trending_music_processed.csv"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3  # total folds = N_SPLITS * N_REPEATS

TOP_N_TERMS = 20


def _as_text_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def evaluate_cv(
    X_text: pd.Series,
    y: np.ndarray,
    vectorizer,
    clf,
    cv,
) -> dict:
    pipe = Pipeline([("vec", vectorizer), ("clf", clf)])
    scores = cross_validate(
        pipe,
        X_text,
        y,
        cv=cv,
        scoring={"auc": "roc_auc", "acc": "accuracy"},
        n_jobs=-1,
        return_train_score=False,
    )
    return {
        "auc_mean": float(np.mean(scores["test_auc"])),
        "auc_sd": float(np.std(scores["test_auc"], ddof=1)),
        "acc_mean": float(np.mean(scores["test_acc"])),
        "acc_sd": float(np.std(scores["test_acc"], ddof=1)),
        "n_folds": int(len(scores["test_auc"])),
    }


def fit_and_extract_top_terms(X_text: pd.Series, y: np.ndarray, pipe: Pipeline, label: str) -> pd.DataFrame:
    """
    Fit a vectorizer+logit pipeline on full data and extract top +/− weighted terms.
    Note: coefficients reflect association in this sample, not causal effects.
    """
    pipe.fit(X_text, y)
    vec = pipe.named_steps["vec"]
    clf = pipe.named_steps["clf"]

    feature_names = vec.get_feature_names_out()
    coef = clf.coef_.ravel()

    top_pos_idx = np.argsort(coef)[-TOP_N_TERMS:][::-1]
    top_neg_idx = np.argsort(coef)[:TOP_N_TERMS]

    rows = []
    for rank, idx in enumerate(top_pos_idx, start=1):
        rows.append({"label": label, "direction": "positive", "rank": rank, "term": feature_names[idx], "coef": float(coef[idx])})
    for rank, idx in enumerate(top_neg_idx, start=1):
        rows.append({"label": label, "direction": "negative", "rank": rank, "term": feature_names[idx], "coef": float(coef[idx])})

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_PATH)
    df["tokens_joined_promo"] = _as_text_series(df["tokens_joined_promo"])
    df["tokens_joined_semantic"] = _as_text_series(df["tokens_joined_semantic"])

    # Labels
    y_median = df["high_views"].astype(int).to_numpy()
    q75 = float(df["view_count"].quantile(0.75))
    y_top_quartile = (df["view_count"] > q75).astype(int).to_numpy()

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    results = []

    def add_result(task: str, diagnostic: str, config: str, out: dict, notes: str = ""):
        results.append(
            {
                "task": task,
                "diagnostic": diagnostic,
                "config": config,
                "auc_mean": out["auc_mean"],
                "auc_sd": out["auc_sd"],
                "acc_mean": out["acc_mean"],
                "acc_sd": out["acc_sd"],
                "n_folds": out["n_folds"],
                "notes": notes,
            }
        )

    # Shared classifiers
    logit = LogisticRegression(max_iter=5000, solver="liblinear")
    dummy = DummyClassifier(strategy="most_frequent")

    # -----------------------------
    # Sanity checks (median label, promo)
    # -----------------------------
    X_promo = df["tokens_joined_promo"]
    out_dummy = evaluate_cv(X_promo, y_median, CountVectorizer(ngram_range=(1, 2), min_df=2), dummy, cv)
    add_result("median_split", "sanity_baseline", "Dummy (most_frequent)", out_dummy, notes="Majority-class baseline.")

    y_shuf = rng.permutation(y_median)
    out_shuf = evaluate_cv(X_promo, y_shuf, CountVectorizer(ngram_range=(1, 2), min_df=2), logit, cv)
    add_result("median_split", "sanity_label_shuffle", "Shuffled labels (BoW Count, min_df=2)", out_shuf, notes="Should be ~0.50 AUC if pipeline is not leaking.")

    # -----------------------------
    # Representation: promo vs semantic (median label)
    # -----------------------------
    for col, label in [("tokens_joined_promo", "promo"), ("tokens_joined_semantic", "semantic")]:
        out = evaluate_cv(df[col], y_median, CountVectorizer(ngram_range=(1, 2), min_df=2), logit, cv)
        add_result("median_split", "representation", f"Count (min_df=2) | {label}", out)

    # -----------------------------
    # Preprocessing sensitivity: min_df (median label, promo)
    # -----------------------------
    for min_df in [1, 2, 5, 10]:
        out = evaluate_cv(X_promo, y_median, CountVectorizer(ngram_range=(1, 2), min_df=min_df), logit, cv)
        add_result("median_split", "min_df", f"Count | min_df={min_df}", out)

    # -----------------------------
    # Vectorizer comparison: Count vs TF–IDF (median label, promo)
    # -----------------------------
    out_count = evaluate_cv(X_promo, y_median, CountVectorizer(ngram_range=(1, 2), min_df=2), logit, cv)
    add_result("median_split", "vectorizer", "Count (min_df=2)", out_count)

    out_tfidf = evaluate_cv(X_promo, y_median, TfidfVectorizer(ngram_range=(1, 2), min_df=2), logit, cv)
    add_result("median_split", "vectorizer", "TFIDF (min_df=2)", out_tfidf)

    # -----------------------------
    # Tune C for TF–IDF (median label, promo)
    # -----------------------------
    tfidf_pipe = Pipeline(
        [
            ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=5000, solver="liblinear")),
        ]
    )
    grid = GridSearchCV(
        tfidf_pipe,
        param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_promo, y_median)
    best_auc = float(grid.best_score_)
    add_result(
        "median_split",
        "tfidf_tuning",
        f"Best TFIDF C={grid.best_params_['clf__C']} (min_df=2)",
        {
            "auc_mean": best_auc,
            "auc_sd": float("nan"),
            "acc_mean": float("nan"),
            "acc_sd": float("nan"),
            "n_folds": int(N_SPLITS * N_REPEATS),
        },
        notes="GridSearchCV reports mean CV AUC for the best setting; sd not stored.",
    )

    # -----------------------------
    # Label robustness: top quartile (harder label, more imbalanced)
    # -----------------------------
    out_q75 = evaluate_cv(X_promo, y_top_quartile, TfidfVectorizer(ngram_range=(1, 2), min_df=2), logit, cv)
    add_result("top_quartile", "label_choice", "TFIDF (min_df=2) | promo", out_q75, notes=f"high_views = 1 if view_count > 75th percentile ({q75:.0f}).")

    # -----------------------------
    # Save summary table
    # -----------------------------
    results_df = pd.DataFrame(results)
    out_table = PROJECT_ROOT / "data" / "processed" / "diagnostics_week5_results_cv.csv"
    results_df.to_csv(out_table, index=False)
    print("=== Week 5 diagnostics (CV) summary ===")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {out_table}")

    # -----------------------------
    # Top terms (for interpretation)
    # -----------------------------
    top_terms = []
    # Use tuned TF–IDF model for median label (promo)
    best_C = float(grid.best_params_["clf__C"])
    best_pipe = Pipeline(
        [
            ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=5000, solver="liblinear", C=best_C)),
        ]
    )
    top_terms.append(fit_and_extract_top_terms(df["tokens_joined_promo"], y_median, best_pipe, label=f"promo_tfidf_C={best_C}"))
    top_terms.append(fit_and_extract_top_terms(df["tokens_joined_semantic"], y_median, best_pipe, label=f"semantic_tfidf_C={best_C}"))

    top_terms_df = pd.concat(top_terms, ignore_index=True)
    out_terms = PROJECT_ROOT / "data" / "processed" / "diagnostics_week5_top_terms.csv"
    top_terms_df.to_csv(out_terms, index=False)
    print(f"Top terms saved to {out_terms}")

    # -----------------------------
    # Figure (simple, report-friendly)
    # -----------------------------
    # Panel 1: distribution of AUC across folds for baseline model (Count, promo, min_df=2)
    # To get fold-level AUCs we re-run cross_validate once here and keep the raw scores.
    base_pipe = Pipeline([("vec", CountVectorizer(ngram_range=(1, 2), min_df=2)), ("clf", logit)])
    base_scores = cross_validate(base_pipe, X_promo, y_median, cv=cv, scoring="roc_auc", n_jobs=-1)
    fold_auc = base_scores["test_score"]

    # Panel 2: representation mean±sd (Count, min_df=2)
    repr_rows = results_df[(results_df["task"] == "median_split") & (results_df["diagnostic"] == "representation")].copy()

    # Panel 3: min_df mean±sd (Count, promo)
    min_df_rows = results_df[(results_df["task"] == "median_split") & (results_df["diagnostic"] == "min_df")].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax1 = axes[0]
    ax1.boxplot(fold_auc, vert=True, widths=0.5)
    ax1.set_title("Stability across CV folds\n(Count, promo, min_df=2)")
    ax1.set_ylabel("AUC")
    ax1.set_ylim(0, 1)
    ax1.set_xticks([1])
    ax1.set_xticklabels([f"{N_SPLITS}x{N_REPEATS} CV"])

    ax2 = axes[1]
    ax2.bar(
        repr_rows["config"],
        repr_rows["auc_mean"],
        yerr=repr_rows["auc_sd"],
        capsize=4,
        color=["steelblue", "coral"],
        edgecolor="black",
    )
    ax2.set_title("Promo vs semantic\n(mean±sd AUC)")
    ax2.set_ylabel("AUC")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="x", rotation=20)

    ax3 = axes[2]
    ax3.bar(min_df_rows["config"], min_df_rows["auc_mean"], yerr=min_df_rows["auc_sd"], capsize=4, color="steelblue", edgecolor="black")
    ax3.set_title("Sensitivity to min_df\n(mean±sd AUC)")
    ax3.set_ylabel("AUC")
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis="x", rotation=20)

    plt.suptitle("Week 5 diagnostics (CV): predictive performance (high_views)", fontsize=11)
    plt.tight_layout()
    fig_path = FIG_DIR / "diagnostics_week5_auc_cv.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
