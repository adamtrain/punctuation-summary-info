"""Linear probes: standardized features + L2-regularized logistic regression (BRIEF.md section 4.5)."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROBE_C = 0.1  # inverse L2 strength; fixed for now, to be tuned on training data in milestone 3


def fit_eval(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
             C: float = PROBE_C) -> float:
    """Test accuracy of a probe trained on (X_train, y_train)."""
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    probe.fit(X_train, y_train)
    return float(probe.score(X_test, y_test))


def fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, C: float = PROBE_C) -> np.ndarray:
    """Predicted test labels of a probe trained on (X_train, y_train)."""
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    probe.fit(X_train, y_train)
    return probe.predict(X_test)


def pair_flip_null(correct: np.ndarray, pair_ids: np.ndarray, n_perm: int = 10000, seed: int = 0) -> np.ndarray:
    """Permutation null for held-out accuracy, at the pair level.

    Swapping a test pair's two labels turns each of its members' correct predictions into wrong ones and vice versa.
    Under the null of no association between features and labels, every swap pattern is equally likely, so the null
    distribution of accuracy is obtained by flipping the correctness of random pairs. No retraining is involved, so
    the null cannot be inflated by a probe amplifying a chance correlation between permuted and true labels (the
    failure seen with retrained shuffles; BRIEF.md section 10.4).
    """
    uniq, inv = np.unique(pair_ids, return_inverse=True)
    per_pair = np.bincount(inv, weights=correct.astype(float), minlength=len(uniq))  # correct members per pair
    size = np.bincount(inv, minlength=len(uniq))
    flips = np.random.default_rng(seed).random((n_perm, len(uniq))) < 0.5
    flipped = np.where(flips, size - per_pair, per_pair)  # a swap turns c correct out of s into s - c correct
    return flipped.sum(1) / size.sum()


def chance_band(n_test: int, z: float = 1.96) -> tuple[float, float]:
    """Range a no-information probe's accuracy falls in 95% of the time (normal approximation)."""
    half = z * np.sqrt(0.25 / n_test)
    return 0.5 - half, 0.5 + half
