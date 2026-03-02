## YouTube Trending Music Metadata and Relative Engagement

This project studies whether the language used in YouTube trending music-video titles and descriptions helps predict which videos receive relatively higher view counts within the trending set. The final report is now scoped to the four English-speaking regions `US`, `GB`, `CA`, and `AU` so that language-based interpretation is not mixed across different languages.

- GitHub repository: `https://github.com/Andran1k/SOSC314_CourseProject`
- GitHub Pages report: `https://andran1k.github.io/SOSC314_CourseProject/`

## Project Summary

The repository contains a full course-project pipeline for collecting YouTube trending data, building processed text datasets, running descriptive analysis, fitting lightweight text-based models, and reporting the final findings in a public-facing notebook and HTML page. The final report focuses on a careful and modest claim: metadata language contains some predictive information about relative engagement among trending music videos, but timing, packaging, and simple structural cues matter as well.

## Research Question

**Does language in YouTube trending music-video metadata help predict above-median view counts within the trending set?**

## Data Source and Unit of Analysis

- **Source:** YouTube Data API v3, using `videos.list` with `chart=mostPopular`
- **Category filter:** Music (`category_id = 10`)
- **Unit of analysis:** One row = one trending YouTube video
- **Main processed dataset:** `data/processed/trending_music_processed.csv` (English-only: `US`, `GB`, `CA`, `AU`)
- **Archive dataset:** `data/processed/trending_music_processed_all_regions.csv`

The processed dataset included in the repository is sufficient to reproduce the final report. Fetching new API data is optional and only needed if you want to extend or refresh the sample.

## Repository Structure

- `data/raw/`: timestamped raw CSV files collected from the YouTube API
- `data/processed/`: cleaned and feature-engineered datasets plus diagnostics tables
- `assets/`: project assets such as the promotional phrase list
- `figures/`: saved plots used in weekly reports and the final project
- `scripts/`: reproducible scripts for fetching, preprocessing, diagnostics, and modeling
- `notebooks/`: the final notebook report source
- `docs/`: rendered site assets for later GitHub Pages publication
- `reports/`: earlier weekly write-ups and supporting course deliverables

## Requirements

- Python 3.8 or newer
- Core project dependencies listed in `requirements.txt`
- Optional API dependencies listed in `requirements_api.txt`

Install the analysis and report-building dependencies with:

```bash
py -m pip install -r requirements.txt
```

If you want to collect new YouTube API data as well:

```bash
py -m pip install -r requirements_api.txt
```

For API collection, create a `.env` file in the project root:

```bash
YOUTUBE_API_KEY=your_api_key_here
```

## Reproducing the Project

### 1. Create and activate an environment

Example on Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
py -m pip install -r requirements.txt
```

### 3. Optionally fetch new raw data

This step is optional because the processed dataset is already included.

```bash
py scripts/fetch_youtube_trending.py
```

### 4. Build the processed dataset

```bash
py scripts/build_processed_dataset.py
```

### 5. Run analysis and diagnostics scripts

```bash
py scripts/explore_trending_data.py
py scripts/make_figures_week3.py
py scripts/train_bow_model.py
py scripts/compare_operationalizations_week4.py
py scripts/diagnostics_week5.py
py scripts/final_report_extensions.py
```

### 6. Execute and export the final notebook

```bash
py -m jupyter nbconvert --to html --execute --no-input notebooks/final_report.ipynb --output index.html --output-dir docs
```

Or use the helper script:

```bash
powershell -ExecutionPolicy Bypass -File scripts/publish_report.ps1
```

## Main Scripts

- `scripts/fetch_youtube_trending.py`: collects raw trending-video metadata from the YouTube API
- `scripts/build_processed_dataset.py`: builds the English-only main dataset plus an all-region archive and adds timing, packaging, and structure features
- `scripts/explore_trending_data.py`: prints descriptive summaries
- `scripts/make_figures_week3.py`: creates descriptive figures
- `scripts/train_bow_model.py`: fits a basic bag-of-words logistic model and saves evaluation figures
- `scripts/compare_operationalizations_week4.py`: compares promo-inclusive and semantic-filtered text
- `scripts/diagnostics_week5.py`: runs cross-validated robustness checks and saves summary outputs
- `scripts/final_report_extensions.py`: runs the final title/description, timing, packaging, and promo-structure analyses
- `scripts/publish_report.ps1`: exports the final notebook to `docs/index.html`, copies the source notebook into `docs/notebooks/`, and copies figures into `docs/figures/` for GitHub Pages

## Final Report Files

- Source notebook: `notebooks/final_report.ipynb`
- Render target for publication: `docs/index.html`
- GitHub Pages notebook copy: `docs/notebooks/final_report.ipynb`
- GitHub Pages figure copy: `docs/figures/`
- GitHub Pages helper file: `docs/.nojekyll`

## Publishing Later

The repo is prepared for GitHub Pages publication from `main` / `docs`, but nothing is published automatically. When you are ready to publish later:

1. Push the repository to GitHub.
2. Open the repository settings page.
3. Go to `Settings` > `Pages`.
4. Set the source to `Deploy from a branch`.
5. Choose branch `main` and folder `/docs`.
6. Save and wait for the site URL to appear.

The expected report URL format will be:

```text
https://andran1k.github.io/SOSC314_CourseProject/
```
