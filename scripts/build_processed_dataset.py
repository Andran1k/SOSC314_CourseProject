import re
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "sample_trending_videos.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = PROCESSED_DIR / "music_processed.csv"

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
WS_PATTERN = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(URL_PATTERN, "", s)
    s = re.sub(WS_PATTERN, " ", s).strip()
    return s

def tokenize(s: str):
    return re.findall(r"[A-Za-z0-9']+", s.lower())

def count_hashtags(s: str) -> int:
    return len(re.findall(r"#\w+", str(s)))

df = pd.read_csv(RAW_PATH)

# Keep Music only (category_id=10)
df = df[df["category_id"].astype(str) == "10"].copy()

# Clean text
df["title_clean"] = df["title"].apply(clean_text)
df["description_clean"] = df["description"].apply(clean_text)
df["document_clean"] = (df["title_clean"] + " " + df["description_clean"]).str.strip()

# Tokens and measurements
df["tokens"] = df["document_clean"].apply(tokenize)
df["token_count"] = df["tokens"].apply(len)
df["unique_token_count"] = df["tokens"].apply(lambda t: len(set(t)))
df["lexical_diversity"] = df.apply(
    lambda r: (r["unique_token_count"] / r["token_count"]) if r["token_count"] > 0 else 0.0,
    axis=1
)
df["tokens_joined"] = df["tokens"].apply(lambda t: " ".join(t))

# Simple punctuation/style features (title-based)
df["exclamation_count"] = df["title"].astype(str).str.count("!")
df["question_count"] = df["title"].astype(str).str.count(r"\?")
df["title_hashtag_count"] = df["title"].apply(count_hashtags)

# Caps ratio in title
def caps_ratio(s: str) -> float:
    s = str(s)
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    caps = sum(1 for c in letters if c.isupper())
    return caps / len(letters)

df["caps_ratio"] = df["title"].apply(caps_ratio)

# Label for ML: high views (median split)
median_views = df["view_count"].median()
df["high_views"] = (df["view_count"] > median_views).astype(int)

keep_cols = [
    "video_id", "published_at", "region", "category_id",
    "view_count", "like_count", "comment_count",
    "title", "description", "tags",
    "title_clean", "description_clean", "document_clean",
    "token_count", "unique_token_count", "lexical_diversity",
    "exclamation_count", "question_count", "title_hashtag_count", "caps_ratio",
    "high_views", "tokens_joined"
]

df[keep_cols].to_csv(OUT_PATH, index=False)
print(f"Saved processed dataset: {OUT_PATH} (rows={len(df)})")
print(f"Median view_count used for high_views split: {median_views}")
