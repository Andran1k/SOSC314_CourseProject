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
print(df[["token_count_promo", "token_count_semantic", "lexdiv_promo", "lexdiv_semantic"]].describe())
print()

print("=== EXAMPLE DOCUMENTS (CLEAN) ===")
for i, txt in enumerate(df["document_promo"].head(3), start=1):
    safe_txt = str(txt[:400]).encode("cp1252", errors="replace").decode("cp1252")
    print(f"\n--- Example {i} ---\n{safe_txt}...")
