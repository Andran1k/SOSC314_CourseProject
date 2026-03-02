import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from text_model_utils import SimpleVectorizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "trending_music_processed.csv"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH)

plt.figure(figsize=(7, 4))
plt.hist(np.log10(df["view_count"] + 1), bins=20, color="steelblue", edgecolor="black")
plt.xlabel("log10(view_count + 1)")
plt.ylabel("Number of videos")
plt.title("Distribution of views (log scale, US/GB/CA/AU)")
plt.tight_layout()
plt.savefig(FIG_DIR / "view_count_distribution_log.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
plt.hist(df["token_count_promo"], bins=20, alpha=0.6, label="promo")
plt.hist(df["token_count_semantic"], bins=20, alpha=0.6, label="semantic-filtered")
plt.xlabel("Token count")
plt.ylabel("Number of videos")
plt.title("Text length before vs after filtering")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "token_count_promo_vs_semantic.png", dpi=150)
plt.close()

texts = df["tokens_joined_promo"].fillna("").tolist()

vec1 = SimpleVectorizer(kind="count", ngram_range=(1, 1), min_df=2)
X1 = vec1.fit_transform(texts)
counts1 = X1.sum(axis=0)
vocab1 = np.array(vec1.vocabulary_)
top_idx1 = counts1.argsort()[-20:][::-1]

plt.figure(figsize=(7, 5))
plt.barh(vocab1[top_idx1][::-1], counts1[top_idx1][::-1], color="steelblue")
plt.xlabel("Count")
plt.title("Top unigrams (promo text, min_df=2)")
plt.tight_layout()
plt.savefig(FIG_DIR / "top_unigrams_count_promo.png", dpi=150)
plt.close()

vec2 = SimpleVectorizer(kind="count", ngram_range=(2, 2), min_df=2)
X2 = vec2.fit_transform(texts)
counts2 = X2.sum(axis=0)
vocab2 = np.array(vec2.vocabulary_)
top_idx2 = counts2.argsort()[-20:][::-1]

plt.figure(figsize=(7, 5))
plt.barh(vocab2[top_idx2][::-1], counts2[top_idx2][::-1], color="steelblue")
plt.xlabel("Count")
plt.title("Top bigrams (promo text, min_df=2)")
plt.tight_layout()
plt.savefig(FIG_DIR / "top_bigrams_count_promo.png", dpi=150)
plt.close()

print("Saved figures to figures/")
