import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "music_processed.csv"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# 1) View count distribution (log)
plt.figure()
plt.hist(np.log10(df["view_count"] + 1), bins=20)
plt.xlabel("log10(view_count + 1)")
plt.ylabel("Number of videos")
plt.title("Distribution of views (log scale)")
plt.tight_layout()
plt.savefig(FIG_DIR / "view_count_distribution_log.png")
plt.close()

# 2) Token count distribution
plt.figure()
plt.hist(df["token_count"], bins=20)
plt.xlabel("Token count (document_clean)")
plt.ylabel("Number of videos")
plt.title("Distribution of text length")
plt.tight_layout()
plt.savefig(FIG_DIR / "text_length_distribution.png")
plt.close()

# Bag of words: top unigrams and bigrams
texts = df["tokens_joined"].fillna("").tolist()

# 3) Top unigrams
vec1 = CountVectorizer(ngram_range=(1, 1), min_df=2)
X1 = vec1.fit_transform(texts)
counts1 = np.asarray(X1.sum(axis=0)).ravel()
vocab1 = np.array(vec1.get_feature_names_out())
top_idx1 = counts1.argsort()[-20:][::-1]

plt.figure()
plt.barh(vocab1[top_idx1][::-1], counts1[top_idx1][::-1])
plt.xlabel("Count")
plt.title("Top unigrams (min_df=2)")
plt.tight_layout()
plt.savefig(FIG_DIR / "top_unigrams.png")
plt.close()

# 4) Top bigrams
vec2 = CountVectorizer(ngram_range=(2, 2), min_df=2)
X2 = vec2.fit_transform(texts)
counts2 = np.asarray(X2.sum(axis=0)).ravel()
vocab2 = np.array(vec2.get_feature_names_out())
top_idx2 = counts2.argsort()[-20:][::-1]

plt.figure()
plt.barh(vocab2[top_idx2][::-1], counts2[top_idx2][::-1])
plt.xlabel("Count")
plt.title("Top bigrams (min_df=2)")
plt.tight_layout()
plt.savefig(FIG_DIR / "top_bigrams.png")
plt.close()

print("Saved figures to figures/")
