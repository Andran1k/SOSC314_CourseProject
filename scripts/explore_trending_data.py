"""
explore_trending_data.py

Initial exploratory analysis of YouTube trending video metadata.
Used for Week 2 to assess data feasibility and basic properties.

This script:
- Loads the raw sample dataset
- Prints basic counts and summaries
- Shows example titles and descriptions
- Computes category counts
"""

import pandas as pd
from pathlib import Path

DATA_RAW_PATH = Path("data/raw/sample_trending_videos.csv")

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(DATA_RAW_PATH)

print("=== BASIC DATA INFO ===")
print(f"Number of videos: {len(df)}")
print()

# -----------------------------
# MISSING VALUE CHECK
# -----------------------------

print("=== MISSING VALUES ===")
print(df[["title", "description"]].isna().sum())
print()

# -----------------------------
# TEXT LENGTH CHECK
# -----------------------------

df["title_length"] = df["title"].astype(str).apply(len)
df["description_length"] = df["description"].astype(str).apply(len)

print("=== TEXT LENGTH SUMMARY ===")
print(df[["title_length", "description_length"]].describe())
print()

# -----------------------------
# SAMPLE EXAMPLES
# -----------------------------

print("=== SAMPLE TITLES ===")
for i, title in enumerate(df["title"].head(5), start=1):
    print(f"{i}. {title}")
print()

print("=== SAMPLE DESCRIPTIONS ===")
for i, desc in enumerate(df["description"].head(3), start=1):
    print(f"{i}. {desc[:200]}...")
print()

# -----------------------------
# CATEGORY COUNTS
# -----------------------------

print("=== CATEGORY COUNTS ===")
category_counts = df["category_id"].value_counts()
print(category_counts)
print()

print("Exploratory analysis complete.")
