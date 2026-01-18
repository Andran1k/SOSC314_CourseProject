"""
Fetch trending YouTube videos (Week 2 feasibility sample).
Saves a CSV to data/raw/.
"""

from googleapiclient.discovery import build
import pandas as pd
from pathlib import Path


# -----------------------------
# CONFIGURATION
# -----------------------------

API_KEY = "AIzaSyAu7BpqODKIIAUlzhmAS3WOfchU7LZJESk"
REGION_CODE = "US"
MAX_RESULTS = 50

OUTPUT_PATH = Path("data/raw/sample_trending_videos.csv")

# -----------------------------
# BUILD YOUTUBE API CLIENT
# -----------------------------

youtube = build(
    serviceName="youtube",
    version="v3",
    developerKey=API_KEY
)

# -----------------------------
# FETCH TRENDING VIDEOS
# -----------------------------

request = youtube.videos().list(
    part="snippet,statistics",
    chart="mostPopular",
    regionCode=REGION_CODE,
    maxResults=MAX_RESULTS
)

response = request.execute()

# -----------------------------
# PARSE RESPONSE
# -----------------------------

records = []

for item in response["items"]:
    snippet = item["snippet"]
    stats = item["statistics"]

    records.append({
        "video_id": item["id"],
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

# -----------------------------
# SAVE TO CSV
# -----------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} videos to {OUTPUT_PATH}")
