from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from text_model_utils import build_numeric_matrix, evaluate_binary_model_cv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "trending_music_processed.csv"
DIAG_PATH = PROJECT_ROOT / "data" / "processed" / "diagnostics_week5_results_cv.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)


def get_best_c() -> float:
    if DIAG_PATH.exists():
        diag = pd.read_csv(DIAG_PATH)
        row = diag[diag["diagnostic"] == "tfidf_tuning"]
        if not row.empty:
            config = str(row.iloc[0]["config"])
            if "C=" in config:
                return float(config.split("C=")[1].split()[0])
    return 5.0


def result_row(label: str, out: dict) -> dict:
    return {
        "model": label,
        "auc_mean": out["auc_mean"],
        "auc_sd": out["auc_sd"],
        "acc_mean": out["acc_mean"],
        "acc_sd": out["acc_sd"],
        "n_folds": out["n_folds"],
    }


def main():
    df = pd.read_csv(DATA_PATH)
    y = df["high_views"].astype(int).to_numpy()
    best_c = get_best_c()

    title_texts = df["tokens_joined_title"].fillna("").astype(str).tolist()
    desc_texts = df["tokens_joined_description"].fillna("").astype(str).tolist()
    combo_texts = df["tokens_joined_promo"].fillna("").astype(str).tolist()

    title_results = []
    for label, texts in [
        ("Title only", title_texts),
        ("Description only", desc_texts),
        ("Title + description", combo_texts),
    ]:
        out = evaluate_binary_model_cv(texts, y, vectorizer_kind="tfidf", ngram_range=(1, 2), min_df=2, c=best_c)
        title_results.append(result_row(label, out))

    title_df = pd.DataFrame(title_results)
    title_df.to_csv(OUT_DIR / "final_report_title_description_results.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(title_df["model"], title_df["auc_mean"], yerr=title_df["auc_sd"], capsize=4, color=["steelblue", "tan", "darkorange"], edgecolor="black")
    plt.ylabel("AUC")
    plt.ylim(0, 1)
    plt.title("Where the predictive signal comes from")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_title_description_auc.png", dpi=150)
    plt.close()

    timing_cols = [
        "upload_hour_utc",
        "upload_weekday_utc",
        "is_weekend_utc",
        "lag_hours_first_seen",
        "title_char_len",
        "description_char_len",
        "promo_share",
        "lexdiv_promo",
    ]
    numeric_matrix = build_numeric_matrix(df, timing_cols)

    timing_model_rows = []
    timing_model_rows.append(
        result_row(
            "Text only",
            evaluate_binary_model_cv(combo_texts, y, vectorizer_kind="tfidf", ngram_range=(1, 2), min_df=2, c=best_c),
        )
    )
    timing_model_rows.append(
        result_row(
            "Timing + structure only",
            evaluate_binary_model_cv(None, y, numeric_features=numeric_matrix, c=best_c),
        )
    )
    timing_model_rows.append(
        result_row(
            "Text + timing + structure",
            evaluate_binary_model_cv(combo_texts, y, vectorizer_kind="tfidf", ngram_range=(1, 2), min_df=2, numeric_features=numeric_matrix, c=best_c),
        )
    )

    weekday_rates = (
        df.groupby("upload_weekday_name_utc", sort=False)["high_views"]
        .mean()
        .rename("high_view_rate")
        .reset_index()
    )
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_rates["upload_weekday_name_utc"] = pd.Categorical(weekday_rates["upload_weekday_name_utc"], categories=weekday_order, ordered=True)
    weekday_rates = weekday_rates.sort_values("upload_weekday_name_utc")

    lag_summary = (
        df.groupby("high_views")["lag_hours_first_seen"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "lag_hours_mean", "median": "lag_hours_median"})
    )

    timing_results = pd.concat(
        [
            pd.DataFrame(timing_model_rows).assign(section="model_comparison"),
            weekday_rates.rename(columns={"upload_weekday_name_utc": "model"}).assign(section="weekday_high_view_rate", auc_mean=lambda x: x["high_view_rate"], auc_sd=np.nan, acc_mean=np.nan, acc_sd=np.nan, n_folds=np.nan),
            lag_summary.rename(columns={"high_views": "model"}).assign(section="lag_summary", auc_mean=lambda x: x["lag_hours_mean"], auc_sd=lambda x: x["lag_hours_median"], acc_mean=lambda x: x["count"], acc_sd=np.nan, n_folds=np.nan),
        ],
        ignore_index=True,
        sort=False,
    )
    timing_results.to_csv(OUT_DIR / "final_report_timing_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lag_groups = [
        df.loc[df["high_views"] == 0, "lag_hours_first_seen"].to_numpy(),
        df.loc[df["high_views"] == 1, "lag_hours_first_seen"].to_numpy(),
    ]
    axes[0].boxplot(lag_groups, labels=["Below median", "Above median"])
    axes[0].set_title("Lag to first collection")
    axes[0].set_ylabel("Hours")

    axes[1].bar(weekday_rates["upload_weekday_name_utc"].astype(str), weekday_rates["high_view_rate"], color="slategray", edgecolor="black")
    axes[1].set_title("High-view rate by upload weekday")
    axes[1].set_ylabel("Share above median")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("Timing patterns in the English-only sample")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_timing_patterns.png", dpi=150)
    plt.close()

    release_rows = []
    for feature, label in [
        ("has_official_video", "Official video"),
        ("has_lyrics", "Lyrics / lyric video"),
        ("has_feat_collab", "Feat / collaboration"),
    ]:
        mask = df[feature] == 1
        release_rows.append(
            {
                "label": label,
                "count": int(mask.sum()),
                "share_of_sample": float(mask.mean()),
                "high_view_rate": float(df.loc[mask, "high_views"].mean()) if mask.any() else np.nan,
                "non_label_high_view_rate": float(df.loc[~mask, "high_views"].mean()),
            }
        )

    release_df = pd.DataFrame(release_rows)
    release_df.to_csv(OUT_DIR / "final_report_release_labels.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(release_df["label"], release_df["high_view_rate"], color=["steelblue", "darkorange", "seagreen"], edgecolor="black")
    plt.ylabel("Share above median views")
    plt.ylim(0, 1)
    plt.title("Packaging labels and relative engagement")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_release_type_high_views.png", dpi=150)
    plt.close()

    structure_cols = ["promo_share", "title_char_len", "description_char_len", "lexdiv_promo"]
    structure_matrix = build_numeric_matrix(df, structure_cols)

    structure_rows = []
    structure_rows.append(
        result_row(
            "Text only",
            evaluate_binary_model_cv(combo_texts, y, vectorizer_kind="tfidf", ngram_range=(1, 2), min_df=2, c=best_c),
        )
    )
    structure_rows.append(
        result_row(
            "Promo + structure only",
            evaluate_binary_model_cv(None, y, numeric_features=structure_matrix, c=best_c),
        )
    )
    structure_rows.append(
        result_row(
            "Text + promo + structure",
            evaluate_binary_model_cv(combo_texts, y, vectorizer_kind="tfidf", ngram_range=(1, 2), min_df=2, numeric_features=structure_matrix, c=best_c),
        )
    )

    promo_quartiles = pd.qcut(df["promo_share"], q=4, duplicates="drop")
    promo_summary = (
        df.assign(promo_quartile=promo_quartiles)
        .groupby("promo_quartile", observed=False)["high_views"]
        .mean()
        .reset_index()
        .rename(columns={"high_views": "high_view_rate"})
    )

    structure_results = pd.concat(
        [
            pd.DataFrame(structure_rows).assign(section="model_comparison"),
            promo_summary.rename(columns={"promo_quartile": "model"}).assign(section="promo_quartile_rate", auc_mean=lambda x: x["high_view_rate"], auc_sd=np.nan, acc_mean=np.nan, acc_sd=np.nan, n_folds=np.nan),
        ],
        ignore_index=True,
        sort=False,
    )
    structure_results.to_csv(OUT_DIR / "final_report_structure_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    promo_groups = [
        df.loc[df["high_views"] == 0, "promo_share"].to_numpy(),
        df.loc[df["high_views"] == 1, "promo_share"].to_numpy(),
    ]
    axes[0].boxplot(promo_groups, labels=["Below median", "Above median"])
    axes[0].set_title("Promo share by outcome")
    axes[0].set_ylabel("Promo-share ratio")

    structure_df = pd.DataFrame(structure_rows)
    axes[1].bar(structure_df["model"], structure_df["auc_mean"], yerr=structure_df["auc_sd"], capsize=4, color=["steelblue", "tan", "darkorange"], edgecolor="black")
    axes[1].set_title("Text vs promo/structure signal")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)

    plt.suptitle("Packaging structure and predictive signal")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_promo_structure_signal.png", dpi=150)
    plt.close()

    print("=== Final report extension summary ===")
    print(title_df.to_string(index=False))
    print()
    print(pd.DataFrame(timing_model_rows).to_string(index=False))
    print()
    print(release_df.to_string(index=False))
    print()
    print(pd.DataFrame(structure_rows).to_string(index=False))


if __name__ == "__main__":
    main()
