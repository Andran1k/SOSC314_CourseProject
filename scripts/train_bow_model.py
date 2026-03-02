from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from text_model_utils import (
    SimpleLogisticRegression,
    SimpleVectorizer,
    auc_score,
    train_test_split_stratified,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "trending_music_processed.csv"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        cm[truth, pred] += 1
    return cm


def roc_curve_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.unique(np.sort(scores))[::-1]
    thresholds = np.r_[np.inf, thresholds, -np.inf]
    fpr = []
    tpr = []
    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fp = np.sum((y_true == 0) & (pred == 1))
        tn = np.sum((y_true == 0) & (pred == 0))
        fn = np.sum((y_true == 1) & (pred == 0))
        tpr.append(tp / (tp + fn) if (tp + fn) else 0.0)
        fpr.append(fp / (fp + tn) if (fp + tn) else 0.0)
    return np.array(fpr), np.array(tpr)


df = pd.read_csv(DATA_PATH)
X_text = df["tokens_joined_promo"].fillna("").astype(str).tolist()
y = df["high_views"].astype(int).to_numpy()

train_idx, test_idx = train_test_split_stratified(y, test_size=0.25, random_state=42)
X_train = [X_text[i] for i in train_idx]
X_test = [X_text[i] for i in test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

vectorizer = SimpleVectorizer(kind="count", ngram_range=(1, 2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = SimpleLogisticRegression(c=1.0, max_iter=1500, learning_rate=0.35)
model.fit(X_train_vec, y_train)

y_prob = model.predict_proba(X_test_vec)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
holdout_auc = auc_score(y_test, y_prob)

cm = confusion_matrix_counts(y_test, y_pred)
plt.figure(figsize=(4.5, 4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
plt.title("Confusion matrix (high_views)")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr = roc_curve_points(y_test, y_prob)
plt.figure(figsize=(4.5, 4))
plt.plot(fpr, tpr, label=f"AUC = {holdout_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve (high_views)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_curve.png", dpi=150)
plt.close()

print(f"Holdout AUC: {holdout_auc:.3f}")
print("Saved ML figures to figures/")
