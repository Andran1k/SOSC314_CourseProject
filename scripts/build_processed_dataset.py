from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ASSETS_DIR = PROJECT_ROOT / "assets"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUT_ALL = PROCESSED_DIR / "trending_music_processed_all_regions.csv"
RAW_OUT_ENGLISH = PROCESSED_DIR / "trending_music_processed.csv"
PHRASES_PATH = ASSETS_DIR / "promo_phrases.txt"
ENGLISH_REGIONS = {"US", "GB", "CA", "AU"}

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
WS_PATTERN = re.compile(r"\s+")


def clean_text(value: str) -> str:
    if pd.isna(value):
        return ""
    value = str(value)
    value = re.sub(URL_PATTERN, " ", value)
    value = re.sub(WS_PATTERN, " ", value).strip()
    return value


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())


def load_phrases(path: Path) -> list[str]:
    phrases: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().lower()
            if line and not line.startswith("#"):
                phrases.append(line)
    return phrases


def remove_phrases(text: str, phrases: list[str]) -> str:
    lowered = str(text).lower()
    for phrase in phrases:
        lowered = lowered.replace(phrase, " ")
    return re.sub(WS_PATTERN, " ", lowered).strip()


def lexical_diversity(unique_count: int, token_count: int) -> float:
    return float(unique_count / token_count) if token_count > 0 else 0.0


def parse_collection_timestamp(path: Path) -> pd.Timestamp:
    match = re.search(r"(\d{8}_\d{6})", path.name)
    if not match:
        raise ValueError(f"Could not parse timestamp from {path.name}")
    return pd.to_datetime(match.group(1), format="%Y%m%d_%H%M%S", utc=True)


def add_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    title_lower = df["title"].fillna("").str.lower()
    df["has_official_video"] = title_lower.str.contains(r"official video", regex=True).astype(int)
    df["has_lyrics"] = title_lower.str.contains(r"lyrics|lyric video", regex=True).astype(int)
    df["has_feat_collab"] = title_lower.str.contains(r"\bfeat\b|\bft\b|\bfeaturing\b", regex=True).astype(int)
    df["title_has_brackets"] = title_lower.str.contains(r"[\[\]\(\)]", regex=True).astype(int)
    df["title_has_numbers"] = title_lower.str.contains(r"\d", regex=True).astype(int)
    return df


def prepare_dataset(source_df: pd.DataFrame, phrases: list[str]) -> pd.DataFrame:
    df = source_df.copy()
    df = df.sort_values(["collected_at_first", "video_id"]).drop_duplicates(subset=["video_id"], keep="first")

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["tags"] = df["tags"].fillna("")

    df["title_clean"] = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)
    df["title_only_clean"] = df["title_clean"]
    df["description_only_clean"] = df["description_clean"]

    df["document_promo"] = (df["title_clean"] + " " + df["description_clean"]).str.strip()
    df["document_semantic"] = df["document_promo"].apply(lambda text: remove_phrases(text, phrases))

    df["tokens_title"] = df["title_only_clean"].apply(tokenize)
    df["tokens_description"] = df["description_only_clean"].apply(tokenize)
    df["tokens_promo"] = df["document_promo"].apply(tokenize)
    df["tokens_semantic"] = df["document_semantic"].apply(tokenize)

    df["token_count_promo"] = df["tokens_promo"].apply(len)
    df["token_count_semantic"] = df["tokens_semantic"].apply(len)
    df["unique_token_count_promo"] = df["tokens_promo"].apply(lambda tokens: len(set(tokens)))
    df["unique_token_count_semantic"] = df["tokens_semantic"].apply(lambda tokens: len(set(tokens)))

    df["lexdiv_promo"] = df.apply(
        lambda row: lexical_diversity(row["unique_token_count_promo"], row["token_count_promo"]),
        axis=1,
    )
    df["lexdiv_semantic"] = df.apply(
        lambda row: lexical_diversity(row["unique_token_count_semantic"], row["token_count_semantic"]),
        axis=1,
    )

    promo_removed = df["token_count_promo"] - df["token_count_semantic"]
    df["promo_share"] = (promo_removed / df["token_count_promo"].replace(0, pd.NA)).fillna(0.0).astype(float)

    df["tokens_joined_title"] = df["tokens_title"].apply(" ".join)
    df["tokens_joined_description"] = df["tokens_description"].apply(" ".join)
    df["tokens_joined_promo"] = df["tokens_promo"].apply(" ".join)
    df["tokens_joined_semantic"] = df["tokens_semantic"].apply(" ".join)

    df["published_at_dt"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["upload_hour_utc"] = df["published_at_dt"].dt.hour.astype(int)
    df["upload_weekday_utc"] = df["published_at_dt"].dt.weekday.astype(int)
    df["upload_weekday_name_utc"] = df["published_at_dt"].dt.day_name()
    df["is_weekend_utc"] = df["upload_weekday_utc"].isin([5, 6]).astype(int)

    df["lag_hours_first_seen"] = (
        (df["collected_at_first"] - df["published_at_dt"]).dt.total_seconds() / 3600.0
    ).astype(float)

    df["title_char_len"] = df["title"].str.len().astype(int)
    df["description_char_len"] = df["description"].str.len().astype(int)
    df["is_english_region"] = df["region"].isin(ENGLISH_REGIONS).astype(int)

    df = add_rule_features(df)

    median_views = df["view_count"].median()
    df["high_views"] = (df["view_count"] > median_views).astype(int)
    df["collected_at_first"] = df["collected_at_first"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["published_at_dt"] = df["published_at_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    keep_cols = [
        "video_id",
        "published_at",
        "published_at_dt",
        "collected_at_first",
        "region",
        "is_english_region",
        "category_id",
        "view_count",
        "like_count",
        "comment_count",
        "title",
        "description",
        "tags",
        "title_clean",
        "description_clean",
        "title_only_clean",
        "description_only_clean",
        "document_promo",
        "document_semantic",
        "token_count_promo",
        "token_count_semantic",
        "unique_token_count_promo",
        "unique_token_count_semantic",
        "lexdiv_promo",
        "lexdiv_semantic",
        "promo_share",
        "upload_hour_utc",
        "upload_weekday_utc",
        "upload_weekday_name_utc",
        "is_weekend_utc",
        "lag_hours_first_seen",
        "title_char_len",
        "description_char_len",
        "title_has_brackets",
        "title_has_numbers",
        "has_official_video",
        "has_lyrics",
        "has_feat_collab",
        "high_views",
        "tokens_joined_title",
        "tokens_joined_description",
        "tokens_joined_promo",
        "tokens_joined_semantic",
    ]
    return df[keep_cols].copy()


raw_files = sorted(RAW_DIR.glob("trending_raw_*.csv"))
if not raw_files:
    raise FileNotFoundError("No raw CSV files found in data/raw/. Run fetch_youtube_trending.py first.")

frames: list[pd.DataFrame] = []
for path in raw_files:
    current = pd.read_csv(path)
    current["collected_at_first"] = parse_collection_timestamp(path)
    frames.append(current)

raw_df = pd.concat(frames, ignore_index=True)
raw_df["category_id"] = raw_df["category_id"].astype(str)
music_df = raw_df[raw_df["category_id"] == "10"].copy()

phrases = load_phrases(PHRASES_PATH)

all_regions_dataset = prepare_dataset(music_df, phrases)
english_scope_dataset = prepare_dataset(music_df[music_df["region"].isin(ENGLISH_REGIONS)].copy(), phrases)

all_regions_dataset.to_csv(RAW_OUT_ALL, index=False)
english_scope_dataset.to_csv(RAW_OUT_ENGLISH, index=False)

print(f"Loaded raw: {len(raw_files)} file(s)")
print(f"Saved all-region archive: {RAW_OUT_ALL} (rows={len(all_regions_dataset)})")
print(f"Saved English-only main dataset: {RAW_OUT_ENGLISH} (rows={len(english_scope_dataset)})")
print("English-only regions:", sorted(english_scope_dataset["region"].unique().tolist()))
