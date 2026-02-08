## Week 5 — How the diagnostics code works (simple + academic)

This note explains what `scripts/diagnostics_week5.py` is doing and why the outputs are meaningful.

### What the script takes as input

- **Dataset**: `data/processed/trending_music_processed.csv`
- **Text features**:
  - **promo**: `tokens_joined_promo` (title + description after basic cleaning)
  - **semantic**: `tokens_joined_semantic` (promo text with “promo phrases” removed)
- **Main label**: `high_views` (1 if `view_count` > sample median, else 0)

### What model is being evaluated

The model is a standard supervised text classifier:

- **Vectorizer**: converts text into numeric features using either
  - **Count** (bag-of-words counts), or
  - **TF–IDF** (downweights very common words, upweights more distinctive words)
- **Classifier**: **logistic regression**, which learns a weight for each word/bigram.  
  - Positive weights push predictions toward `high_views = 1`
  - Negative weights push predictions toward `high_views = 0`

### Why we use cross-validation (CV)

If we evaluate on only one train/test split, results can be misleading—especially with a moderate dataset size—because the test set might be unusually “easy” or “hard.”

So we use **repeated stratified cross-validation**:

- **5 folds × 3 repeats = 15 test folds**
- “Stratified” means each fold keeps roughly the same fraction of 0/1 labels as the full dataset.

This produces 15 AUC values, and we summarize them with **mean ± standard deviation (sd)**. The sd is a direct indicator of how much results vary across different splits.

### What AUC means (in plain language)

**AUC** is a ranking metric. It answers:

> If we randomly pick one high-view video and one low-view video, how often does the model assign a higher score to the high-view one?

- **AUC = 0.50**: chance-level ranking (no useful signal)
- **AUC > 0.50**: some predictive signal
- Values around **0.65–0.75** are often described as “modest” in noisy social data contexts (stronger claims require more evidence and larger samples).

### Why we include sanity checks

The script runs two sanity checks:

- **Majority baseline (dummy classifier)**: predicts the most common class every time. This should not rank well; it gives **AUC ≈ 0.50**.
- **Label shuffle**: randomly permutes `high_views` and reruns the same modeling pipeline. If the evaluation is correct and there is no leakage, performance should return to **chance** (AUC near 0.50, with some random variation).

These checks help confirm that “good” performance is not coming from an error like leaking the label into the features.

### Why `min_df` matters

`min_df` removes very rare terms:

- Low `min_df` keeps rare words (often names/titles).
- High `min_df` removes them, leaving only common vocabulary.

If AUC drops as `min_df` increases, it suggests that rare or specific terms carry useful signal. That affects interpretation: the model may be learning **identifiers** (names/titles) more than broad “style.”

### What the outputs mean

The script saves:

- `data/processed/diagnostics_week5_results_cv.csv`  
  A table of mean±sd AUC/accuracy for each diagnostic configuration.
- `figures/diagnostics_week5_auc_cv.png`  
  A report-friendly figure summarizing fold stability and key comparisons.
- `data/processed/diagnostics_week5_top_terms.csv`  
  The highest-weight positive and negative terms from a TF–IDF logistic regression fit. These are used for **interpretation**, not causal claims.

