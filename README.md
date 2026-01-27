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
- **`figures/`**: Output plots used in reports (e.g., `view_count_distribution_log.png`, `top_unigrams.png`, `confusion_matrix.png`).

### Requirements
- **Python**: 3.9+ recommended.
- **Key packages** (non-exhaustive):
  - `pandas`, `numpy`, `matplotlib`
  - `scikit-learn`
  - `google-api-python-client`
  - `python-dotenv`

You will also need a YouTube Data API key stored in a `.env` file at the project root:

```bash
YOUTUBE_API_KEY=your_api_key_here
```

### Typical Workflow
1. **Fetch raw trending data**
   - Run `fetch_youtube_trending.py` to collect trending videos for multiple regions and save timestamped CSV files under `data/raw/`.
2. **Build the processed dataset**
   - Run `build_processed_dataset.py` to create `data/processed/music_processed.csv` with cleaned text and derived features.
3. **Explore the processed data**
   - Run `explore_trending_data.py` to print summary statistics and example documents you can cite in your Week 3 write-up.
4. **Make descriptive figures**
   - Run `make_figures_week3.py` to generate descriptive figures in `figures/` (distribution plots and top n-grams).
5. **Train and evaluate the BoW model**
   - Run `train_bow_model.py` to fit a bag-of-words logistic regression model predicting `high_views`, and to save confusion-matrix and ROC-curve figures to `figures/`.

All scripts use project-root-safe paths (`Path(__file__).resolve().parent.parent`), so you can run them from the `scripts/` directory or the project root.
