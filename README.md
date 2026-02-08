## YouTube Trending Language Project

### Research Question
**How does the language used in YouTube trending video metadata differ across content categories (with a focus on music)?**

### Data Source
- **API**: YouTube Data API v3 (`videos.list`, `chart=mostPopular`)
- **Regions**: Multiple English-speaking regions (e.g., US, GB, CA, AU)
- **Category filter**: Primarily `category_id=10` (Music)

### Unit of Analysis
- **One row = one YouTube trending video** (with metadata and basic engagement metrics).

### Repository Structure
- **`data/raw/`**: Raw CSV files fetched from the YouTube API (e.g., `trending_raw_US_*.csv`).
- **`data/processed/`**: Cleaned and feature-engineered datasets (e.g., `music_processed.csv`).
- **`scripts/`**:
  - **`fetch_youtube_trending.py`**: Fetches trending videos via the API and saves timestamped raw CSV files for several regions, then filters to music videos.
  - **`build_processed_dataset.py`**: Cleans titles/descriptions, tokenizes text, creates lexical/formatting features, and produces `music_processed.csv`.
  - **`explore_trending_data.py`**: Prints descriptive summaries of the processed dataset for use in weekly reports.
  - **`make_figures_week3.py`**: Generates basic EDA plots (view-count distribution, text length distribution, top unigrams/bigrams) and saves them to `figures/`.
  - **`train_bow_model.py`**: Trains a bag-of-words logistic regression model to predict whether a video is above-median views (`high_views`) and saves evaluation figures (confusion matrix, ROC curve).
  - **`compare_operationalizations_week4.py`**: Compares promo vs semantic text (token counts, top n-grams, TF-IDF cosine similarity); reads `trending_music_processed.csv`.
  - **`diagnostics_week5.py`**: Week 5 diagnostics—sensitivity to train/test split, text representation (promo vs semantic), and preprocessing (`min_df`). Prints a summary table, saves `data/processed/diagnostics_week5_results.csv` and `figures/diagnostics_week5_auc.png`.
- **`figures/`**: Output plots used in reports (e.g., `view_count_distribution_log.png`, `top_unigrams.png`, `confusion_matrix.png`, `diagnostics_week5_auc.png`).

### Requirements
- **Python**: 3.9+ recommended.
- **Install for analysis/diagnostics (no API needed):**

```bash
py -m pip install -r requirements.txt
```

- **Install for API fetching (only needed to run `fetch_youtube_trending.py`):**

```bash
py -m pip install -r requirements_api.txt
```

You will also need a YouTube Data API key stored in a `.env` file at the project root:

```bash
YOUTUBE_API_KEY=your_api_key_here
```

### Typical Workflow
1. **Fetch raw trending data**
   - Run `fetch_youtube_trending.py` to collect trending videos for **US, GB, CA, AU** and save timestamped CSV files under `data/raw/`. Run it periodically (e.g. daily) to grow the raw data; the API returns up to ~50 per region per run.
2. **Build the processed dataset**
   - Run `build_processed_dataset.py` to merge **all** raw `trending_raw_*.csv` files (deduplicated by `video_id`), filter to music, and write `data/processed/trending_music_processed.csv`. More raw files → larger sample for diagnostics.
3. **Explore the processed data**
   - Run `explore_trending_data.py` to print summary statistics and example documents you can cite in your Week 3 write-up.
4. **Make descriptive figures**
   - Run `make_figures_week3.py` to generate descriptive figures in `figures/` (distribution plots and top n-grams).
5. **Train and evaluate the BoW model**
   - Run `train_bow_model.py` to fit a bag-of-words logistic regression model predicting `high_views`, and to save confusion-matrix and ROC-curve figures to `figures/`.
6. **Week 5 diagnostics**
   - Run `diagnostics_week5.py` (requires `trending_music_processed.csv`) to run cross-validated diagnostics and sanity checks; outputs:
     - `data/processed/diagnostics_week5_results_cv.csv`
     - `data/processed/diagnostics_week5_top_terms.csv`
     - `figures/diagnostics_week5_auc_cv.png`
   - A short explanation of what the diagnostics are doing is in `reports/week5_code_explanation.md`.

All scripts use project-root-safe paths (`Path(__file__).resolve().parent.parent`), so you can run them from the `scripts/` directory or the project root.
