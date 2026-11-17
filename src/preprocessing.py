"""
LinkShield AI — Preprocessing Module
======================================

Person 2 deliverable: reusable scikit-learn preprocessing utilities that
prepare the extracted URL features for a Logistic Regression classifier.

This module builds a ``ColumnTransformer`` that:
- Scales numerical features with ``StandardScaler``.
- One-hot-encodes categorical features (``tld``) with
  ``handle_unknown="ignore"`` so unseen TLDs do not crash the pipeline.
- Imputes missing values where necessary.

The preprocessor is **not fitted** in this module.  Person 3 will fit
it on the training data only, inside a scikit-learn Pipeline.

Author: Person 2 (Feature Engineering & Preprocessing)
"""

from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .url_features import FEATURE_NAMES

# ---------------------------------------------------------------------------
# Feature classification
# ---------------------------------------------------------------------------

#: Numerical features — all features except ``tld``.
NUMERICAL_FEATURES: List[str] = [f for f in FEATURE_NAMES if f != "tld"]

#: Categorical features — currently only ``tld``.
CATEGORICAL_FEATURES: List[str] = ["tld"]


def build_preprocessor() -> ColumnTransformer:
    """
    Build and return a scikit-learn ``ColumnTransformer`` for preprocessing
    the features produced by ``URLFeatureExtractor``.

    Numerical pipeline:
        1. ``SimpleImputer(strategy="constant", fill_value=0)`` — fills any
           remaining NaN with zero (defensive; the extractor should not
           produce NaN for valid inputs).
        2. ``StandardScaler()`` — zero-mean, unit-variance scaling for
           Logistic Regression.

    Categorical pipeline:
        1. ``SimpleImputer(strategy="constant", fill_value="unknown")`` —
           fills missing TLD with the string ``"unknown"``.
        2. ``OneHotEncoder(handle_unknown="ignore", sparse_output=False)`` —
           one-hot-encodes the TLD.  Unknown TLDs seen only at prediction
           time produce an all-zero row for the TLD columns, which is safe.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        An **unfitted** transformer ready to be placed in a Pipeline.

    Example
    -------
    >>> from src.url_features import URLFeatureExtractor
    >>> from src.preprocessing import build_preprocessor
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> pipe = Pipeline([
    ...     ("url_features", URLFeatureExtractor()),
    ...     ("preprocessor", build_preprocessor()),
    ...     ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ... ])
    >>> # pipe.fit(X_train_urls, y_train)
    >>> # pipe.predict(X_test_urls)
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=int,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Drop any unexpected columns
    )

    return preprocessor


def get_feature_lists() -> Tuple[List[str], List[str]]:
    """
    Return the lists of numerical and categorical feature names.

    Useful for downstream code that needs to know which features are which.
    """
    return NUMERICAL_FEATURES, CATEGORICAL_FEATURES
