from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return text.split()


def _generate_ngrams(tokens: list[str], ngram_range: tuple[int, int]) -> list[str]:
    lo, hi = ngram_range
    grams: list[str] = []
    for n in range(lo, hi + 1):
        if len(tokens) < n:
            continue
        grams.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return grams


@dataclass
class SimpleVectorizer:
    kind: str = "count"
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int = 1

    def fit(self, texts: Iterable[str]) -> "SimpleVectorizer":
        docs = list(texts)
        doc_freq: Counter[str] = Counter()
        for text in docs:
            grams = set(_generate_ngrams(_tokenize(text), self.ngram_range))
            doc_freq.update(grams)

        vocab = sorted(term for term, freq in doc_freq.items() if freq >= self.min_df)
        self.vocabulary_ = vocab
        self.term_to_idx_ = {term: idx for idx, term in enumerate(vocab)}
        self.doc_freq_ = np.array([doc_freq[term] for term in vocab], dtype=float)
        self.n_docs_ = len(docs)
        self.idf_ = np.log((1.0 + self.n_docs_) / (1.0 + self.doc_freq_)) + 1.0
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        docs = list(texts)
        X = np.zeros((len(docs), len(self.vocabulary_)), dtype=float)
        for row_idx, text in enumerate(docs):
            grams = _generate_ngrams(_tokenize(text), self.ngram_range)
            if not grams:
                continue
            counts = Counter(grams)
            for gram, count in counts.items():
                col_idx = self.term_to_idx_.get(gram)
                if col_idx is not None:
                    X[row_idx, col_idx] = float(count)

        if self.kind == "tfidf":
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            X = (X / row_sums) * self.idf_
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms

        return X

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        docs = list(texts)
        self.fit(docs)
        return self.transform(docs)


class SimpleLogisticRegression:
    def __init__(self, c: float = 1.0, max_iter: int = 1200, learning_rate: float = 0.4):
        self.c = c
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleLogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0
        reg_strength = 1.0 / max(self.c, 1e-6)

        for _ in range(self.max_iter):
            logits = X @ self.coef_ + self.intercept_
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -25, 25)))
            error = probs - y

            grad_w = (X.T @ error) / n_samples + (reg_strength / n_samples) * self.coef_
            grad_b = float(np.mean(error))

            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        logits = X @ self.coef_ + self.intercept_
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -25, 25)))
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    pos_rank_sum = np.sum(ranks[y_true == 1])
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def stratified_folds(
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(random_state)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(n_repeats):
        idx0_shuf = rng.permutation(idx0)
        idx1_shuf = rng.permutation(idx1)
        split0 = np.array_split(idx0_shuf, n_splits)
        split1 = np.array_split(idx1_shuf, n_splits)

        for fold_idx in range(n_splits):
            test_idx = np.concatenate([split0[fold_idx], split1[fold_idx]])
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=False)
            folds.append((train_idx, test_idx))

    return folds


def train_test_split_stratified(
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in [0, 1]:
        idx = np.where(y == label)[0]
        idx = rng.permutation(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    return train_idx, test_idx


def standardize_train_test(
    train_numeric: np.ndarray,
    test_numeric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_numeric, axis=0, keepdims=True)
    std = np.std(train_numeric, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train_numeric - mean) / std, (test_numeric - mean) / std


def _weekday_one_hot(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).astype(int)
    out = np.zeros((len(values), 7), dtype=float)
    out[np.arange(len(values)), values] = 1.0
    return out


def build_numeric_matrix(df, columns: list[str]) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for col in columns:
        values = df[col].to_numpy()
        if col == "upload_weekday_utc":
            blocks.append(_weekday_one_hot(values.astype(int)))
        else:
            blocks.append(values.astype(float).reshape(-1, 1))
    return np.hstack(blocks) if blocks else np.zeros((len(df), 0), dtype=float)


def evaluate_binary_model_cv(
    texts: list[str] | None,
    y: np.ndarray,
    *,
    vectorizer_kind: str = "tfidf",
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    numeric_features: np.ndarray | None = None,
    c: float = 1.0,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> dict:
    y = np.asarray(y).astype(int)
    folds = stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    aucs: list[float] = []
    accs: list[float] = []

    for train_idx, test_idx in folds:
        X_train_parts: list[np.ndarray] = []
        X_test_parts: list[np.ndarray] = []

        if texts is not None:
            train_texts = [texts[i] for i in train_idx]
            test_texts = [texts[i] for i in test_idx]
            vec = SimpleVectorizer(kind=vectorizer_kind, ngram_range=ngram_range, min_df=min_df)
            X_train_text = vec.fit_transform(train_texts)
            X_test_text = vec.transform(test_texts)
            X_train_parts.append(X_train_text)
            X_test_parts.append(X_test_text)

        if numeric_features is not None:
            X_train_num = numeric_features[train_idx]
            X_test_num = numeric_features[test_idx]
            X_train_num, X_test_num = standardize_train_test(X_train_num, X_test_num)
            X_train_parts.append(X_train_num)
            X_test_parts.append(X_test_num)

        X_train = np.hstack(X_train_parts) if X_train_parts else np.zeros((len(train_idx), 0), dtype=float)
        X_test = np.hstack(X_test_parts) if X_test_parts else np.zeros((len(test_idx), 0), dtype=float)

        model = SimpleLogisticRegression(c=c)
        model.fit(X_train, y[train_idx])
        scores = model.predict_proba(X_test)[:, 1]
        preds = (scores >= 0.5).astype(int)

        aucs.append(auc_score(y[test_idx], scores))
        accs.append(accuracy_score(y[test_idx], preds))

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "acc_mean": float(np.mean(accs)),
        "acc_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "n_folds": len(aucs),
        "fold_aucs": aucs,
        "fold_accs": accs,
    }


def evaluate_dummy_cv(
    y: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> dict:
    y = np.asarray(y).astype(int)
    folds = stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    aucs: list[float] = []
    accs: list[float] = []

    for _, test_idx in folds:
        majority = int(np.mean(y) >= 0.5)
        preds = np.full(len(test_idx), majority, dtype=int)
        scores = preds.astype(float)
        aucs.append(0.5)
        accs.append(accuracy_score(y[test_idx], preds))

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "acc_mean": float(np.mean(accs)),
        "acc_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "n_folds": len(aucs),
        "fold_aucs": aucs,
        "fold_accs": accs,
    }


def fit_text_model_with_vocab(
    texts: list[str],
    y: np.ndarray,
    *,
    vectorizer_kind: str = "tfidf",
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    c: float = 1.0,
) -> tuple[SimpleVectorizer, SimpleLogisticRegression]:
    vec = SimpleVectorizer(kind=vectorizer_kind, ngram_range=ngram_range, min_df=min_df)
    X = vec.fit_transform(texts)
    model = SimpleLogisticRegression(c=c)
    model.fit(X, np.asarray(y).astype(int))
    return vec, model
