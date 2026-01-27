"""
Fetch trending YouTube videos and save raw CSV to data/raw/.

Week 3 upgrade:
- Project-root-safe paths
- Pagination support (collect >50 videos)
- Filter to one category (default: Music category_id=10)
- Timestamped output file for reproducibility
"""

from googleapiclient.discovery import build
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
from datetime import datetime

# -----------------------------
# PATH SETUP (PROJECT ROOT SAFE)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in .env file")

# -----------------------------
# CONFIGURATION
# -----------------------------
REGION_CODE = "US"
MAX_RESULTS_PER_PAGE = 50
TOTAL_TARGET = 200          # collect this many videos before filtering
CATEGORY_ID = "10"          # Music category on YouTube

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = DATA_RAW_DIR / f"trending_raw_{REGION_CODE}_{timestamp}.csv"

# -----------------------------
# BUILD API CLIENT
# -----------------------------
youtube = build("youtube", "v3", developerKey=API_KEY)

# -----------------------------
# FETCH WITH PAGINATION
# -----------------------------
# records = []
# next_page_token = None
#
# while len(records) < TOTAL_TARGET:
#     request = youtube.videos().list(
#         part="snippet,statistics",
#         chart="mostPopular",
#         regionCode=REGION_CODE,
#         maxResults=MAX_RESULTS_PER_PAGE,
#         pageToken=next_page_token
#     )
#     response = request.execute()
#
#     for item in response.get("items", []):
#         snippet = item.get("snippet", {})
#         stats = item.get("statistics", {})
#
#         records.append({
#             "video_id": item.get("id", ""),
#             "title": snippet.get("title", ""),
#             "description": snippet.get("description", ""),
#             "tags": " ".join(snippet.get("tags", [])),
#             "category_id": snippet.get("categoryId", ""),
#             "published_at": snippet.get("publishedAt", ""),
#             "view_count": int(stats.get("viewCount", 0)),
#             "like_count": int(stats.get("likeCount", 0)),
#             "comment_count": int(stats.get("commentCount", 0)),
#             "region": REGION_CODE
#         })
#
#     next_page_token = response.get("nextPageToken")
#     if not next_page_token:
#         break
#
# df = pd.DataFrame(records).drop_duplicates(subset=["video_id"])

REGION_CODES = ["US", "GB", "CA", "AU"]
TOTAL_TARGET_PER_REGION = 200

all_records = []

for REGION_CODE in REGION_CODES:
    records = []
    next_page_token = None

    while len(records) < TOTAL_TARGET_PER_REGION:
        request = youtube.videos().list(
            part="snippet,statistics",
            chart="mostPopular",
            regionCode=REGION_CODE,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})

            records.append({
                "video_id": item.get("id", ""),
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "tags": " ".join(snippet.get("tags", [])),
                "category_id": snippet.get("categoryId", ""),
                "published_at": snippet.get("publishedAt", ""),
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "region": REGION_CODE
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    all_records.extend(records)
    print(f"{REGION_CODE}: collected {len(records)} trending videos")

df = pd.DataFrame(all_records).drop_duplicates(subset=["video_id"])
music_df = df[df["category_id"].astype(str) == "10"].copy()

print(f"Total unique videos: {len(df)}")
print(f"Music videos: {len(music_df)}")

# -----------------------------
# FILTER TO MUSIC CATEGORY
# -----------------------------
music_df = df[df["category_id"] == CATEGORY_ID].copy()

music_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(music_df)} music videos to: {OUTPUT_PATH}")
print(f"(Collected {len(df)} total trending videos before filtering.)")
