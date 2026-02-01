import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "trending_music_processed.csv"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

promo_texts = df["tokens_joined_promo"].fillna("").tolist()
sem_texts = df["tokens_joined_semantic"].fillna("").tolist()

# -----------------------------
# FIGURE 1: TOKEN COUNT SHIFT (PROMO vs SEMANTIC)
# -----------------------------
plt.figure()
plt.hist(df["token_count_promo"], bins=20, alpha=0.6, label="promo")
plt.hist(df["token_count_semantic"], bins=20, alpha=0.6, label="semantic-filtered")
plt.xlabel("Token count")
plt.ylabel("Number of videos")
plt.title("Document length before vs after filtering")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "token_count_promo_vs_semantic.png")
plt.close()

# -----------------------------
# HELPER: TOP N FEATURES BARH
# -----------------------------
def save_top_features_barh(feature_names, values, title, out_path, top_n=20):
    idx = np.argsort(values)[-top_n:][::-1]
    top_feats = np.array(feature_names)[idx][::-1]
    top_vals = np.array(values)[idx][::-1]
    plt.figure()
    plt.barh(top_feats, top_vals)
    plt.xlabel("Count" if "Count" in title else "Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -----------------------------
# FIGURE 2-3: TOP UNIGRAMS (COUNT) promo vs semantic
# -----------------------------
cv_uni = CountVectorizer(ngram_range=(1, 1), min_df=2)

X_promo_uni = cv_uni.fit_transform(promo_texts)
counts_promo_uni = np.asarray(X_promo_uni.sum(axis=0)).ravel()
feats_uni = cv_uni.get_feature_names_out()
save_top_features_barh(
    feats_uni, counts_promo_uni,
    "Top unigrams (Count, promo text)",
    FIG_DIR / "top_unigrams_count_promo.png"
)

X_sem_uni = cv_uni.fit_transform(sem_texts)
counts_sem_uni = np.asarray(X_sem_uni.sum(axis=0)).ravel()
feats_uni_sem = cv_uni.get_feature_names_out()
save_top_features_barh(
    feats_uni_sem, counts_sem_uni,
    "Top unigrams (Count, semantic-filtered text)",
    FIG_DIR / "top_unigrams_count_semantic.png"
)

# -----------------------------
# FIGURE 4-5: TOP BIGRAMS (COUNT) promo vs semantic
# -----------------------------
cv_bi = CountVectorizer(ngram_range=(2, 2), min_df=2)

X_promo_bi = cv_bi.fit_transform(promo_texts)
counts_promo_bi = np.asarray(X_promo_bi.sum(axis=0)).ravel()
feats_bi = cv_bi.get_feature_names_out()
save_top_features_barh(
    feats_bi, counts_promo_bi,
    "Top bigrams (Count, promo text)",
    FIG_DIR / "top_bigrams_count_promo.png"
)

X_sem_bi = cv_bi.fit_transform(sem_texts)
counts_sem_bi = np.asarray(X_sem_bi.sum(axis=0)).ravel()
feats_bi_sem = cv_bi.get_feature_names_out()
save_top_features_barh(
    feats_bi_sem, counts_sem_bi,
    "Top bigrams (Count, semantic-filtered text)",
    FIG_DIR / "top_bigrams_count_semantic.png"
)

# -----------------------------
# FIGURE 6: SIMILARITY DISTRIBUTION promo vs semantic (TF-IDF unigrams)
# -----------------------------
tfidf = TfidfVectorizer(ngram_range=(1, 1), min_df=2)

P = tfidf.fit_transform(promo_texts)
S = tfidf.fit_transform(sem_texts)

# Pairwise cosine similarity (sample to keep fast)
n = min(200, len(df))
P_sub = P[:n]
S_sub = S[:n]

sim_promo = cosine_similarity(P_sub)
sim_sem = cosine_similarity(S_sub)

# Take upper triangle (excluding diagonal)
tri = np.triu_indices(n, k=1)
vals_promo = sim_promo[tri]
vals_sem = sim_sem[tri]

plt.figure()
plt.hist(vals_promo, bins=30, alpha=0.6, label="promo TF-IDF")
plt.hist(vals_sem, bins=30, alpha=0.6, label="semantic TF-IDF")
plt.xlabel("Cosine similarity")
plt.ylabel("Number of pairs")
plt.title("Document similarity shifts after filtering (TF-IDF unigrams)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "similarity_promo_vs_semantic_tfidf.png")
plt.close()

# -----------------------------
# QUICK SUMMARY PRINTS (for report)
# -----------------------------
print("=== SUMMARY ===")
print(f"Videos: {len(df)}")
print("Mean token_count (promo):", df["token_count_promo"].mean())
print("Mean token_count (semantic):", df["token_count_semantic"].mean())
print("Mean lexical diversity (promo):", df["lexdiv_promo"].mean())
print("Mean lexical diversity (semantic):", df["lexdiv_semantic"].mean())
print("Figures saved to:", FIG_DIR)