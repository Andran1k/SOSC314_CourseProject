"""
explore_trending_data.py

Week 3: Exploratory analysis of the processed dataset.
Prints basic summaries that you can cite in the Week 3 report.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "trending_music_processed.csv"

df = pd.read_csv(PROCESSED_PATH)

print("=== BASIC INFO ===")
print(f"Rows (videos): {len(df)}")
print()

print("=== TEXT AVAILABILITY ===")
print("Empty titles:", (df["title"].fillna("").str.strip() == "").sum())
print("Empty descriptions:", (df["description"].fillna("").str.strip() == "").sum())
print()

print("=== LENGTH SUMMARIES ===")
print(df[["title_len", "description_len", "document_len", "token_count"]].describe())
print()

print("=== EXAMPLE DOCUMENTS (CLEAN) ===")
for i, txt in enumerate(df["document_clean"].head(3), start=1):
    print(f"\n--- Example {i} ---\n{txt[:400]}...")
