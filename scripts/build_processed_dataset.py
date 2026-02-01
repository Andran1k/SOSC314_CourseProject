import re
import pandas as pd
from pathlib import Path

# -----------------------------
# PATHS
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ASSETS_DIR = PROJECT_ROOT / "assets"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Use the most recent raw CSV automatically
raw_files = sorted(RAW_DIR.glob("*.csv"))
if not raw_files:
    raise FileNotFoundError("No raw CSV files found in data/raw/. Run fetch_youtube_trending.py first.")

RAW_PATH = raw_files[-1]
OUT_PATH = PROCESSED_DIR / "trending_music_processed.csv"
PHRASES_PATH = ASSETS_DIR / "promo_phrases.txt"

# -----------------------------
# REGEX HELPERS
# -----------------------------
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
WS_PATTERN = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(URL_PATTERN, " ", s)          # remove URLs
    s = re.sub(WS_PATTERN, " ", s).strip()   # normalize spaces
    return s

def tokenize(s: str):
    # Simple tokenizer: keep words/numbers/apostrophes
    return re.findall(r"[A-Za-z0-9']+", s.lower())

def load_phrases(path: Path):
    phrases = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().lower()
            if line and not line.startswith("#"):
                phrases.append(line)
    return phrases

def remove_phrases(text: str, phrases):
    # Remove phrases as whole substrings (simple + transparent)
    t = text.lower()
    for p in phrases:
        t = t.replace(p, " ")
    t = re.sub(WS_PATTERN, " ", t).strip()
    return t

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(RAW_PATH)

# Filter to Music category only
df = df[df["category_id"].astype(str) == "10"].copy()

# Basic cleaning
df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")
df["title_clean"] = df["title"].apply(clean_text)
df["description_clean"] = df["description"].apply(clean_text)

# Operationalization 1 (promo-inclusive document)
df["document_promo"] = (df["title_clean"] + " " + df["description_clean"]).str.strip()

# Operationalization 2 (semantic-filtered document)
phrases = load_phrases(PHRASES_PATH)
df["document_semantic"] = df["document_promo"].apply(lambda t: remove_phrases(t, phrases))

# Tokenize both documents
df["tokens_promo"] = df["document_promo"].apply(tokenize)
df["tokens_semantic"] = df["document_semantic"].apply(tokenize)

# Measurements (for description in report + diagnostics)
df["token_count_promo"] = df["tokens_promo"].apply(len)
df["token_count_semantic"] = df["tokens_semantic"].apply(len)

df["unique_token_count_promo"] = df["tokens_promo"].apply(lambda t: len(set(t)))
df["unique_token_count_semantic"] = df["tokens_semantic"].apply(lambda t: len(set(t)))

def lexical_diversity(unique_count, token_count):
    return (unique_count / token_count) if token_count > 0 else 0.0

df["lexdiv_promo"] = df.apply(lambda r: lexical_diversity(r["unique_token_count_promo"], r["token_count_promo"]), axis=1)
df["lexdiv_semantic"] = df.apply(lambda r: lexical_diversity(r["unique_token_count_semantic"], r["token_count_semantic"]), axis=1)

# A simple label (for later), median split
median_views = df["view_count"].median()
df["high_views"] = (df["view_count"] > median_views).astype(int)

# Join tokens for vectorizers
df["tokens_joined_promo"] = df["tokens_promo"].apply(lambda t: " ".join(t))
df["tokens_joined_semantic"] = df["tokens_semantic"].apply(lambda t: " ".join(t))

# Save
keep_cols = [
    "video_id", "published_at", "region", "category_id",
    "view_count", "like_count", "comment_count",
    "title", "description", "tags",
    "document_promo", "document_semantic",
    "token_count_promo", "token_count_semantic",
    "unique_token_count_promo", "unique_token_count_semantic",
    "lexdiv_promo", "lexdiv_semantic",
    "high_views",
    "tokens_joined_promo", "tokens_joined_semantic",
]

df[keep_cols].to_csv(OUT_PATH, index=False)

print(f"Loaded raw: {RAW_PATH.name}")
print(f"Saved processed: {OUT_PATH} (rows={len(df)})")
print(f"Median views used for high_views: {median_views}")
print(f"Promo phrases loaded: {len(phrases)} from {PHRASES_PATH}")