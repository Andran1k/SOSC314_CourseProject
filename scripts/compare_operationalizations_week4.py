import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from text_model_utils import SimpleVectorizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "trending_music_processed.csv"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_top_features_barh(feature_names, values, title, out_path, top_n=20):
    if len(feature_names) == 0:
        return
    idx = np.argsort(values)[-top_n:][::-1]
    top_feats = np.array(feature_names)[idx][::-1]
    top_vals = np.array(values)[idx][::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top_feats, top_vals, color="steelblue")
    plt.xlabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


df = pd.read_csv(DATA_PATH)
promo_texts = df["tokens_joined_promo"].fillna("").tolist()
sem_texts = df["tokens_joined_semantic"].fillna("").tolist()

plt.figure(figsize=(7, 4))
plt.hist(df["token_count_promo"], bins=20, alpha=0.6, label="promo")
plt.hist(df["token_count_semantic"], bins=20, alpha=0.6, label="semantic-filtered")
plt.xlabel("Token count")
plt.ylabel("Number of videos")
plt.title("Document length before vs after filtering (US, GB, CA, AU)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "token_count_promo_vs_semantic.png", dpi=150)
plt.close()

uni_vec = SimpleVectorizer(kind="count", ngram_range=(1, 1), min_df=2)
promo_uni = uni_vec.fit_transform(promo_texts)
promo_uni_counts = promo_uni.sum(axis=0)
save_top_features_barh(
    uni_vec.vocabulary_,
    promo_uni_counts,
    "Top unigrams (promo text, English-only sample)",
    FIG_DIR / "top_unigrams_count_promo.png",
)

sem_uni = uni_vec.fit_transform(sem_texts)
sem_uni_counts = sem_uni.sum(axis=0)
save_top_features_barh(
    uni_vec.vocabulary_,
    sem_uni_counts,
    "Top unigrams (semantic-filtered text, English-only sample)",
    FIG_DIR / "top_unigrams_count_semantic.png",
)

bi_vec = SimpleVectorizer(kind="count", ngram_range=(2, 2), min_df=2)
promo_bi = bi_vec.fit_transform(promo_texts)
promo_bi_counts = promo_bi.sum(axis=0)
save_top_features_barh(
    bi_vec.vocabulary_,
    promo_bi_counts,
    "Top bigrams (promo text, English-only sample)",
    FIG_DIR / "top_bigrams_count_promo.png",
)

sem_bi = bi_vec.fit_transform(sem_texts)
sem_bi_counts = sem_bi.sum(axis=0)
save_top_features_barh(
    bi_vec.vocabulary_,
    sem_bi_counts,
    "Top bigrams (semantic-filtered text, English-only sample)",
    FIG_DIR / "top_bigrams_count_semantic.png",
)

tfidf_vec = SimpleVectorizer(kind="tfidf", ngram_range=(1, 1), min_df=2)
promo_tfidf = tfidf_vec.fit_transform(promo_texts)
sem_tfidf = tfidf_vec.fit_transform(sem_texts)

n = min(200, len(df))
promo_sub = promo_tfidf[:n]
sem_sub = sem_tfidf[:n]
promo_sim = promo_sub @ promo_sub.T
sem_sim = sem_sub @ sem_sub.T
tri = np.triu_indices(n, k=1)

plt.figure(figsize=(7, 4))
plt.hist(promo_sim[tri], bins=30, alpha=0.6, label="promo TF-IDF")
plt.hist(sem_sim[tri], bins=30, alpha=0.6, label="semantic TF-IDF")
plt.xlabel("Cosine similarity")
plt.ylabel("Number of pairs")
plt.title("Document similarity after filtering (US, GB, CA, AU)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "similarity_promo_vs_semantic_tfidf.png", dpi=150)
plt.close()

print("=== SUMMARY ===")
print(f"Videos: {len(df)}")
print("Regions:", ", ".join(sorted(df["region"].unique().tolist())))
print("Mean token_count (promo):", round(df["token_count_promo"].mean(), 2))
print("Mean token_count (semantic):", round(df["token_count_semantic"].mean(), 2))
print("Mean lexical diversity (promo):", round(df["lexdiv_promo"].mean(), 3))
print("Mean lexical diversity (semantic):", round(df["lexdiv_semantic"].mean(), 3))
print("Figures saved to:", FIG_DIR)
