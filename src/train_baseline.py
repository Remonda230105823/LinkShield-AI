"""
LinkShield AI — Phase 1 Baseline Training & Evaluation
=======================================================

Person 3 deliverable: the first working machine-learning model for the
project — a Logistic Regression baseline that is trained and evaluated
through Person 2's leak-free interfaces.

What this module does
---------------------
1. Confirms the modelling contract handed over by Person 2 (label
   convention, split reproducibility, zero train/test URL overlap).
2. Builds ONE scikit-learn ``Pipeline``:
   ``URLFeatureExtractor -> build_preprocessor() -> LogisticRegression``.
3. Fits the complete Pipeline on ``data/processed/train.csv`` only.
4. Evaluates once on ``data/processed/test.csv`` at threshold 0.50.
5. Saves metrics JSON, confusion-matrix image, baseline notes and the
   fitted Pipeline artifact.

What this module deliberately does NOT do (Phase 1 rules)
---------------------------------------------------------
- No hyperparameter tuning and no threshold tuning (Phase 2 work).
- No SHAP / LIME, no extra models.
- No new split and no change to Person 2's features.
- No network access: URLs are treated as inert text.

Run from the project root::

    python -m src.train_baseline

Author: Person 3 (Baseline Model Development)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Make ``src`` importable when the file is run directly
# (``python src/train_baseline.py``) as well as with ``-m``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from src.data_cleaning import normalize_url  # noqa: E402
from src.url_features import FEATURE_NAMES, URLFeatureExtractor, _safe_parse  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed Phase 1 configuration (agreed in the team handbook)
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
THRESHOLD: float = 0.50          # baseline convention, NOT a tuned value
MAX_ITER: int = 1000             # raise only if a ConvergenceWarning appears
CLASS_WEIGHT: str = "balanced"   # class-imbalance handling for Phase 1
TEST_SIZE: float = 0.20
LABELS = [0, 1]                  # confusion-matrix order: legitimate, phishing
POSITIVE_LABEL: int = 1          # phishing
CV_FOLDS: int = 5

TRAIN_CSV = Path("data/processed/train.csv")
TEST_CSV = Path("data/processed/test.csv")
CLEANED_CSV = Path("data/processed/cleaned_urls.csv")
MANIFEST_JSON = Path("data/manifest.json")
SPLIT_SUMMARY_JSON = Path("artifacts/split_manifest.json")

METRICS_JSON = Path("reports/phase1_baseline_metrics.json")
CONFUSION_PNG = Path("reports/phase1_confusion_matrix.png")
NOTES_MD = Path("reports/phase1_baseline_notes.md")
MODEL_ARTIFACT = Path("artifacts/linkshield_logreg_baseline.joblib")

#: Inert example strings used only to smoke-test the saved artifact.
#: They are never opened, resolved or requested.
SAMPLE_URLS = [
    "https://www.example.com",
    "https://www.example.org/about/team",
    "http://secure-login-verify.example.com/account/update?id=12345&session=abcdef",
    "http://192.0.2.10/bank/signin/confirm.php",
]


# ---------------------------------------------------------------------------
# Data loading and contract checks
# ---------------------------------------------------------------------------

def load_split(
    project_root: Path = PROJECT_ROOT,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load Person 2's saved train/test split exactly as handed over."""
    train_df = pd.read_csv(project_root / TRAIN_CSV)
    test_df = pd.read_csv(project_root / TEST_CSV)

    for name, df in (("train", train_df), ("test", test_df)):
        if list(df.columns) != ["url", "is_phishing"]:
            raise ValueError(
                f"{name}.csv must contain exactly ['url', 'is_phishing'], "
                f"got {list(df.columns)}"
            )
        if df["url"].isna().any():
            raise ValueError(f"{name}.csv contains missing URLs.")
        if not set(df["is_phishing"].unique()) <= set(LABELS):
            raise ValueError(f"{name}.csv contains labels outside {{0, 1}}.")

    return (
        train_df["url"],
        train_df["is_phishing"].astype(int),
        test_df["url"],
        test_df["is_phishing"].astype(int),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hostname(url: str) -> str:
    """Lower-cased hostname using Person 2's parser (no network access)."""
    parsed, _ = _safe_parse(url)
    return (parsed.hostname or "").lower()


def verify_contract(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    project_root: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    """
    Re-check the modelling contract before any training happens
    (handbook Section 9.3, Task 1 and the Person 2 acceptance checklist).

    Raises ``AssertionError`` if a hard requirement fails.
    """
    checks: Dict[str, Any] = {}

    # 1. Label convention: binary, phishing = 1.
    checks["labels_binary"] = bool(
        set(y_train.unique()) == {0, 1} and set(y_test.unique()) == {0, 1}
    )
    assert checks["labels_binary"], "Labels are not binary {0, 1}."

    # 2. Zero URL overlap between train and test (exact and normalized).
    exact_overlap = len(set(X_train) & set(X_test))
    norm_overlap = len(
        set(X_train.map(normalize_url)) & set(X_test.map(normalize_url))
    )
    checks["exact_url_overlap"] = exact_overlap
    checks["normalized_url_overlap"] = norm_overlap
    assert exact_overlap == 0 and norm_overlap == 0, "Train/test URL overlap!"

    # 3. Split sizes match Person 2's recorded split summary.
    summary_path = project_root / SPLIT_SUMMARY_JSON
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checks["matches_person2_split_summary"] = bool(
            summary["train_rows"] == len(X_train)
            and summary["test_rows"] == len(X_test)
            and summary["random_seed"] == RANDOM_STATE
            and summary["train_class_counts"]["phishing (1)"] == int(y_train.sum())
            and summary["test_class_counts"]["phishing (1)"] == int(y_test.sum())
        )
        assert checks["matches_person2_split_summary"], (
            "Loaded split does not match artifacts/split_manifest.json."
        )

    # 4. The saved split is reproducible from the cleaned data with the
    #    agreed procedure (stratified 80/20, seed 42). This proves Person 3
    #    is using Person 2's split and has not created a different one.
    cleaned_path = project_root / CLEANED_CSV
    if cleaned_path.exists():
        clean_df = pd.read_csv(cleaned_path)
        re_train, re_test = train_test_split(
            clean_df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=clean_df["is_phishing"],
        )
        checks["split_reproducible_from_cleaned_csv"] = bool(
            re_train["url"].reset_index(drop=True).equals(X_train.reset_index(drop=True))
            and re_test["url"].reset_index(drop=True).equals(X_test.reset_index(drop=True))
            and (re_test["is_phishing"].to_numpy() == y_test.to_numpy()).all()
        )
        assert checks["split_reproducible_from_cleaned_csv"], (
            "train.csv/test.csv cannot be reproduced from cleaned_urls.csv."
        )

    # 5. Feature schema is the 25 features agreed with Person 2, and the
    #    target is not one of them.
    checks["feature_count"] = len(FEATURE_NAMES)
    checks["target_not_in_features"] = "is_phishing" not in FEATURE_NAMES
    assert checks["target_not_in_features"]

    # 6. Hostname overlap (known limitation, measured not hidden).
    train_hosts = set(X_train.map(_hostname))
    test_hosts = X_test.map(_hostname)
    shared_mask = test_hosts.isin(train_hosts)
    checks["test_rows_with_hostname_seen_in_train"] = int(shared_mask.sum())
    checks["test_rows_with_hostname_seen_in_train_pct"] = float(shared_mask.mean())

    return checks


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_baseline_pipeline(max_iter: int = MAX_ITER) -> Pipeline:
    """
    Build the complete, unfitted baseline Pipeline.

    Every learned step (imputer, scaler, one-hot encoder, classifier) lives
    inside this single object, so calling ``.fit`` on training data is the
    only place anything is learned.
    """
    return Pipeline(
        steps=[
            ("url_features", URLFeatureExtractor()),
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=CLASS_WEIGHT,
                    max_iter=max_iter,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_baseline(
    X_train: pd.Series, y_train: pd.Series, max_iter: int = MAX_ITER
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Fit the full Pipeline on TRAINING data only.

    Convergence warnings are captured and reported, never suppressed.
    """
    pipeline = build_baseline_pipeline(max_iter=max_iter)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)

    convergence = [str(w.message) for w in caught
                   if issubclass(w.category, ConvergenceWarning)]
    clf: LogisticRegression = pipeline.named_steps["classifier"]

    info = {
        "train_rows": int(len(X_train)),
        "max_iter": int(max_iter),
        "n_iter_used": int(np.max(clf.n_iter_)),
        "converged": len(convergence) == 0,
        "convergence_warnings": convergence,
        "n_model_input_columns": int(clf.coef_.shape[1]),
    }
    return pipeline, info


def predict_at_threshold(
    pipeline: Pipeline, X: pd.Series, threshold: float = THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (phishing probability, predicted class at ``threshold``)."""
    classes = list(pipeline.named_steps["classifier"].classes_)
    proba = pipeline.predict_proba(X)[:, classes.index(POSITIVE_LABEL)]
    return proba, (proba >= threshold).astype(int)


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Confusion-matrix counts and phishing-class metrics (pos_label=1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=LABELS).ravel()
    return {
        "precision_phishing": float(
            precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall_phishing": float(
            recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1_phishing": float(
            f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    threshold: float = THRESHOLD,
    train_hostnames: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Single evaluation on the held-out test set at the fixed threshold.

    If ``train_hostnames`` is supplied, the same metrics are also reported
    for the subset of test rows whose hostname never appears in training
    (a diagnostic for the hostname-overlap limitation — not a new split).
    """
    y_true = y_test.to_numpy()
    proba, y_pred = predict_at_threshold(pipeline, X_test, threshold)

    result = _score(y_true, y_pred)
    result["roc_auc"] = float(roc_auc_score(y_true, proba))
    result["average_precision"] = float(average_precision_score(y_true, proba))
    result["false_positive_rate"] = result["fp"] / (result["fp"] + result["tn"])
    result["false_negative_rate"] = result["fn"] / (result["fn"] + result["tp"])
    result["test_rows"] = int(len(y_true))
    result["test_class_counts"] = {
        "legitimate_0": int((y_true == 0).sum()),
        "phishing_1": int((y_true == 1).sum()),
    }

    if train_hostnames is not None:
        unseen = ~X_test.map(_hostname).isin(train_hostnames).to_numpy()
        sub = _score(y_true[unseen], y_pred[unseen])
        sub["rows"] = int(unseen.sum())
        sub["phishing_rows"] = int(y_true[unseen].sum())
        result["unseen_hostname_subset"] = sub

    return result


def cross_validate_on_train(
    X_train: pd.Series, y_train: pd.Series, max_iter: int = MAX_ITER
) -> Dict[str, Any]:
    """
    5-fold stratified cross-validation on TRAINING data only, with the
    fixed baseline configuration.

    This is a stability check (does the score depend on which rows the
    model saw?). Nothing is selected or tuned from it, and the test set is
    not touched.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        build_baseline_pipeline(max_iter=max_iter),
        X_train,
        y_train,
        cv=cv,
        scoring={"precision": "precision", "recall": "recall", "f1": "f1"},
    )
    out: Dict[str, Any] = {"folds": CV_FOLDS, "shuffle_random_state": RANDOM_STATE}
    for name in ("precision", "recall", "f1"):
        values = scores[f"test_{name}"]
        out[f"{name}_phishing_mean"] = float(values.mean())
        out[f"{name}_phishing_std"] = float(values.std())
        out[f"{name}_phishing_per_fold"] = [float(v) for v in values]
    return out


def top_coefficients(pipeline: Pipeline, k: int = 8) -> Dict[str, Any]:
    """
    Largest positive / negative Logistic Regression coefficients.

    Numeric inputs are standardised, so their coefficients are comparable
    with each other. This is a simple sanity view of the linear model, not
    a SHAP/LIME explanation (that is Phase 2).
    """
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["classifier"].coef_.ravel()
    order = np.argsort(coefs)
    as_list = lambda idx: [  # noqa: E731
        {"feature": str(names[i]), "coefficient": float(coefs[i])} for i in idx
    ]
    numeric = [i for i in order if str(names[i]).startswith("num__")]
    return {
        "numeric_pushing_towards_phishing": as_list(numeric[::-1][:k]),
        "numeric_pushing_towards_legitimate": as_list(numeric[:k]),
    }


def dataset_artifact_check(
    X_train: pd.Series, y_train: pd.Series, X_test: pd.Series, y_test: pd.Series
) -> Dict[str, Any]:
    """
    Measure how easily the two classes separate in THIS dataset.

    In PhiUSIIL every legitimate URL is a bare ``https://www.<host>`` home
    page. This function records that fact from the training data and scores
    a one-line, no-machine-learning rule ("flag anything that is not
    https + www") on the test set as a reference point. If a trivial rule
    scores almost as well as the model, the model's score says more about
    how the dataset was collected than about phishing detection in general.
    """
    extractor = URLFeatureExtractor()
    f_train = extractor.transform(X_train)
    f_test = extractor.transform(X_test)
    legit = (y_train == 0).to_numpy()
    phish = ~legit

    def share(mask: pd.Series) -> Dict[str, float]:
        values = mask.to_numpy()
        return {"legitimate": float(values[legit].mean()),
                "phishing": float(values[phish].mean())}

    rule_pred = ((f_test["has_https"] == 0) | (f_test["has_www"] == 0)).astype(int)
    rule = _score(y_test.to_numpy(), rule_pred.to_numpy())
    return {
        "train_share_of_urls_where": {
            "scheme_is_https": share(f_train["has_https"] == 1),
            "hostname_starts_with_www": share(f_train["has_www"] == 1),
            "path_longer_than_slash": share(f_train["path_length"] > 1),
            "has_query_string": share(f_train["has_query"] == 1),
        },
        "trivial_rule": "predict phishing unless the URL is https AND starts with www.",
        "trivial_rule_test_scores": rule,
        "note": "Reference point only. The rule was written from training-data "
                "statistics and is not part of the model.",
    }


def business_metric_inputs(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs for the Phase 1 business proxy (handbook Section 3.8):

        review_hours = (TP + FP) * assumed_review_minutes_per_alert / 60

    scaled to 10,000 screened URLs at the TEST-SET class mix. The minutes
    per alert are ILLUSTRATIVE ASSUMPTIONS, not measured bank figures.
    Person 4 owns the final business-metric write-up.
    """
    scale = 10_000 / metrics["test_rows"]
    alerts = (metrics["tp"] + metrics["fp"]) * scale
    return {
        "basis": "per 10,000 screened URLs at the test-set class mix "
                 "(42.7% phishing — a benchmark mix, not real bank traffic)",
        "alerts_raised_tp_plus_fp": alerts,
        "of_which_false_alarms_fp": metrics["fp"] * scale,
        "missed_phishing_fn": metrics["fn"] * scale,
        "review_hours_by_assumed_minutes_per_alert": {
            f"{m}_min": alerts * m / 60 for m in (1, 3, 5)
        },
        "assumption_note": "1/3/5 minutes per alert are illustrative scenarios "
                           "only; no bank interview or cited source backs them.",
    }


# ---------------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------------

def save_confusion_matrix_png(metrics: Dict[str, Any], path: Path) -> None:
    """Save the confusion matrix with the class order written on the axes."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array([[metrics["tn"], metrics["fp"]],
                   [metrics["fn"], metrics["tp"]]])
    cell_names = np.array([["TN", "FP"], ["FN", "TP"]])
    tick_labels = ["Legitimate (0)", "Phishing (1)"]

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=tick_labels)
    ax.set_yticks([0, 1], labels=tick_labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(
        "LinkShield AI — Phase 1 Logistic Regression baseline\n"
        f"Held-out test set (n = {metrics['test_rows']:,}), threshold = {THRESHOLD:.2f}",
        fontsize=10,
    )
    limit = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cell_names[i, j]}\n{cm[i, j]:,}",
                ha="center", va="center", fontsize=13,
                color="white" if cm[i, j] > limit else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_baseline_notes(report: Dict[str, Any], path: Path) -> None:
    """Write the interpretation notes directly from the measured values."""
    m = report
    cv = m.get("cross_validation_on_train")
    sub = m.get("unseen_hostname_subset")
    biz = m["business_metric_inputs_for_person4"]
    hours = biz["review_hours_by_assumed_minutes_per_alert"]
    n_legit = m["test_class_counts"]["legitimate_0"]
    n_phish = m["test_class_counts"]["phishing_1"]
    coef = m["top_coefficients"]
    art = m["dataset_artifact_check"]
    shares = art["train_share_of_urls_where"]
    rule = art["trivial_rule_test_scores"]

    def coef_rows(items):
        return "\n".join(
            f"| `{c['feature'].replace('num__', '')}` | {c['coefficient']:+.3f} |"
            for c in items
        )

    lines = f"""# Phase 1 Baseline Notes — Logistic Regression

**Author:** Person 3 (Baseline Model Development)
**Generated by:** `src/train_baseline.py` on {m['run_timestamp_utc']}
**Source of every number below:** `reports/phase1_baseline_metrics.json` (same run)

> This is a Phase 1 baseline on a public benchmark dataset. It is a reference
> point for Phase 2, not a production-ready detector, and no claim is made
> that it would perform the same on real bank traffic.

---

## 1. What was run

| Item | Value |
|---|---|
| Model | scikit-learn `LogisticRegression` (final step of one `Pipeline`) |
| Pipeline | `URLFeatureExtractor` → `build_preprocessor()` (impute + `StandardScaler` + `OneHotEncoder`) → `LogisticRegression` |
| Class imbalance handling | `class_weight="balanced"` (no SMOTE, no resampling) |
| `max_iter` | {m['model_params']['max_iter']} (solver used {m['training']['n_iter_used']} iterations; converged: {m['training']['converged']}) |
| Solver / penalty / C | `{m['model_params']['solver']}` / scikit-learn default L2 / {m['model_params']['C']} (defaults, not tuned) |
| Random seed | {m['random_state']} |
| Decision threshold | {m['threshold']:.2f} (baseline convention, not tuned) |
| Positive class | phishing (`is_phishing = 1`) |
| Training rows | {m['training']['train_rows']:,} (fit on these only) |
| Test rows | {m['test_rows']:,} ({n_legit:,} legitimate, {n_phish:,} phishing) |
| Raw features | {m['feature_schema']['raw_feature_count']} lexical URL features from Person 2 |
| Model input columns | {m['training']['n_model_input_columns']} after one-hot encoding `tld` |

## 2. Held-out test results (threshold {m['threshold']:.2f})

Confusion matrix, label order `[0, 1]` = `[legitimate, phishing]`:

| | Predicted legitimate (0) | Predicted phishing (1) |
|---|---|---|
| **Actual legitimate (0)** | TN = {m['tn']:,} | FP = {m['fp']:,} |
| **Actual phishing (1)** | FN = {m['fn']:,} | TP = {m['tp']:,} |

| Metric (phishing = positive class) | Value |
|---|---|
| Precision | {m['precision_phishing']:.4f} |
| Recall | {m['recall_phishing']:.4f} |
| F1 | {m['f1_phishing']:.4f} |
| Accuracy (supplementary only) | {m['accuracy']:.4f} |
| ROC-AUC (supplementary, threshold-free) | {m['roc_auc']:.4f} |
| Average precision (supplementary, threshold-free) | {m['average_precision']:.4f} |

Image: `reports/phase1_confusion_matrix.png`

## 3. What the numbers mean for a retail bank security team

- **Precision {m['precision_phishing']:.4f}** — of the {m['tp'] + m['fp']:,} test URLs the model flagged as
  phishing, {m['tp']:,} really were phishing and {m['fp']:,} were legitimate. Precision tells
  the analyst team how much of their alert queue is real work.
- **Recall {m['recall_phishing']:.4f}** — of the {n_phish:,} phishing URLs in the test set, the model
  caught {m['tp']:,} and missed {m['fn']:,}. Recall tells the bank how much phishing gets
  through this control.
- **False negatives (FN = {m['fn']:,})** — phishing URLs scored as legitimate
  ({m['false_negative_rate']:.2%} of all phishing in the test set). For a bank this is the costly error:
  a customer or employee may reach a credential-harvesting page.
- **False positives (FP = {m['fp']:,})** — legitimate URLs flagged as phishing
  ({m['false_positive_rate']:.2%} of all legitimate URLs in the test set). These cost analyst review
  time and can block or delay genuine customer links.
- **Why not accuracy alone** — accuracy mixes both classes into one number and
  can look high while the phishing class is detected poorly. Precision, recall
  and F1 for the phishing class are the primary metrics; accuracy is reported
  only as a supplement.

## 4. Stability check (training data only)

{_cv_section(cv)}

## 5. Hostname-overlap diagnostic

Person 2's leakage audit found that some hostnames appear in both train and
test (URL-level stratified split). To see whether that inflates the result, the
same test predictions were re-scored on only the test rows whose hostname never
appears in training. This is a diagnostic on the existing test set — the split
was not changed.

{_subset_section(m, sub)}

## 6. Which features the linear model leans on

Largest coefficients among the standardised numeric features (a sanity view of
the fitted linear model — not a SHAP/LIME explanation, which is Phase 2 work).

| Pushes towards **phishing** | Coefficient |
|---|---|
{coef_rows(coef['numeric_pushing_towards_phishing'])}

| Pushes towards **legitimate** | Coefficient |
|---|---|
{coef_rows(coef['numeric_pushing_towards_legitimate'])}

Correlated features (for example `url_length` with the individual character
counts) share weight between them, so single coefficients should not be
over-interpreted.

## 7. Inputs for the Phase 1 business metric (for Person 4)

Formula from the handbook: `review_hours = (TP + FP) × assumed_minutes_per_alert / 60`.
Scaled to {biz['basis']}:

| Quantity per 10,000 screened URLs | Value |
|---|---|
| Alerts raised (TP + FP) | {biz['alerts_raised_tp_plus_fp']:.1f} |
| of which false alarms (FP) | {biz['of_which_false_alarms_fp']:.1f} |
| Missed phishing (FN), reported separately | {biz['missed_phishing_fn']:.1f} |
| Review hours at an **assumed** 1 min/alert | {hours['1_min']:.1f} |
| Review hours at an **assumed** 3 min/alert | {hours['3_min']:.1f} |
| Review hours at an **assumed** 5 min/alert | {hours['5_min']:.1f} |

The minutes-per-alert values are illustrative scenarios, not measured facts.
The benchmark is 42.7% phishing; real bank traffic contains far less phishing,
so the alert volume and the precision would both be different in deployment.

## 8. Limitations

1. **Benchmark, not bank traffic.** PhiUSIIL is a public research dataset with
   a 42.7% phishing share. Real traffic has a much lower phishing rate, which
   lowers precision for the same model.
2. **The dataset is close to trivially separable — the most important
   caveat.** In the training data {shares['scheme_is_https']['legitimate']:.1%} of legitimate URLs are `https`,
   {shares['hostname_starts_with_www']['legitimate']:.1%} start with `www.`, and {shares['path_longer_than_slash']['legitimate']:.1%} have any path beyond `/`
   (phishing: {shares['scheme_is_https']['phishing']:.1%}, {shares['hostname_starts_with_www']['phishing']:.1%} and {shares['path_longer_than_slash']['phishing']:.1%}). A one-line rule with no
   machine learning — "{art['trivial_rule']}" —
   scores precision {rule['precision_phishing']:.4f}, recall {rule['recall_phishing']:.4f}, F1 {rule['f1_phishing']:.4f} on the same test set.
   The baseline's F1 of {m['f1_phishing']:.4f} is therefore only a small step above a
   trivial rule and mostly reflects how the benchmark was collected (legitimate
   entries are bare home pages). Consequence: the model flags ordinary
   legitimate deep links — in the artifact smoke test an inert
   `https://www.example.org/about/team` string is scored as phishing. The
   near-perfect score must not be presented as real-world detection ability.
3. **Hostname overlap.** {m['contract_checks']['test_rows_with_hostname_seen_in_train']:,} test rows ({m['contract_checks']['test_rows_with_hostname_seen_in_train_pct']:.2%}) share a hostname with
   training rows; see Section 5 for the measured effect.
4. **URL text only.** No page content, DNS, WHOIS or domain age is used, so a
   phishing page on a clean-looking URL cannot be detected by this model.
5. **Threshold not optimised.** 0.50 is a convention. Choosing a threshold for
   a stated business cost is Phase 2 work and must use training data /
   cross-validation, never the test set.
6. **No tuning, no robustness test, no explainability yet.** `C`, the penalty
   and the solver are scikit-learn defaults. Evasion/drift tests and SHAP/LIME
   are Phase 2.
7. **Approximate TLD parsing.** `tld` is the last hostname label, so
   multi-part suffixes such as `.co.uk` are represented as `uk`.
8. **Single test evaluation.** The test set was scored once with this fixed
   configuration. It must not be reused to select Phase 2 models or thresholds.

## 9. Reproduce

```bash
pip install -r requirements.txt
python -m src.train_baseline
```

Outputs: `reports/phase1_baseline_metrics.json`, `reports/phase1_confusion_matrix.png`,
`reports/phase1_baseline_notes.md`, `artifacts/linkshield_logreg_baseline.joblib`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lines, encoding="utf-8")


def _cv_section(cv: Optional[Dict[str, Any]]) -> str:
    if not cv:
        return "Cross-validation was skipped for this run (`--skip-cv`)."
    return (
        f"{cv['folds']}-fold stratified cross-validation of the same fixed configuration on the\n"
        "training set only (nothing was tuned or selected from it):\n\n"
        "| Metric (phishing) | Mean | Std across folds |\n|---|---|---|\n"
        f"| Precision | {cv['precision_phishing_mean']:.4f} | {cv['precision_phishing_std']:.4f} |\n"
        f"| Recall | {cv['recall_phishing_mean']:.4f} | {cv['recall_phishing_std']:.4f} |\n"
        f"| F1 | {cv['f1_phishing_mean']:.4f} | {cv['f1_phishing_std']:.4f} |\n\n"
        "A small spread across folds means the baseline score does not depend on\n"
        "which training rows the model happened to see."
    )


def _subset_section(m: Dict[str, Any], sub: Optional[Dict[str, Any]]) -> str:
    if not sub:
        return "Not computed for this run."
    return (
        "| Test rows scored | Rows | Precision | Recall | F1 | FP | FN |\n"
        "|---|---|---|---|---|---|---|\n"
        f"| All test rows | {m['test_rows']:,} | {m['precision_phishing']:.4f} | "
        f"{m['recall_phishing']:.4f} | {m['f1_phishing']:.4f} | {m['fp']:,} | {m['fn']:,} |\n"
        f"| Hostname unseen in training | {sub['rows']:,} | {sub['precision_phishing']:.4f} | "
        f"{sub['recall_phishing']:.4f} | {sub['f1_phishing']:.4f} | {sub['fp']:,} | {sub['fn']:,} |"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_baseline(
    project_root: Path = PROJECT_ROOT,
    run_cv: bool = True,
    save_model: bool = True,
    verbose: bool = True,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Run the complete Phase 1 baseline and save every output."""
    log = print if verbose else (lambda *a, **k: None)

    log("[1/6] Loading Person 2's split ...")
    X_train, y_train, X_test, y_test = load_split(project_root)

    log("[2/6] Verifying the modelling contract ...")
    checks = verify_contract(X_train, y_train, X_test, y_test, project_root)

    cv_result = None
    if run_cv:
        log(f"[3/6] {CV_FOLDS}-fold cross-validation on TRAINING data only ...")
        cv_result = cross_validate_on_train(X_train, y_train)
    else:
        log("[3/6] Cross-validation skipped.")

    log("[4/6] Fitting the full Pipeline on TRAINING data only ...")
    pipeline, train_info = train_baseline(X_train, y_train)
    if not train_info["converged"]:
        log("      WARNING: ConvergenceWarning raised — see metrics JSON.")

    log(f"[5/6] Evaluating once on the held-out test set at threshold {THRESHOLD} ...")
    metrics = evaluate(
        pipeline, X_test, y_test, THRESHOLD,
        train_hostnames=set(X_train.map(_hostname)),
    )

    clf_params = pipeline.named_steps["classifier"].get_params()
    manifest_path = project_root / MANIFEST_JSON
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})

    report: Dict[str, Any] = {
        "model": "LogisticRegression",
        "phase": "Phase 1 baseline",
        "positive_class": "phishing",
        "positive_label": POSITIVE_LABEL,
        "threshold": THRESHOLD,
        "threshold_note": "Baseline convention; not tuned on any data.",
        "random_state": RANDOM_STATE,
        "precision_phishing": metrics["precision_phishing"],
        "recall_phishing": metrics["recall_phishing"],
        "f1_phishing": metrics["f1_phishing"],
        "accuracy": metrics["accuracy"],
        "accuracy_note": "Supplementary only.",
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "confusion_matrix_label_order": ["legitimate_0", "phishing_1"],
        "confusion_matrix": [[metrics["tn"], metrics["fp"]],
                             [metrics["fn"], metrics["tp"]]],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "false_positive_rate": metrics["false_positive_rate"],
        "false_negative_rate": metrics["false_negative_rate"],
        "test_rows": metrics["test_rows"],
        "test_class_counts": metrics["test_class_counts"],
        "train_class_counts": {
            "legitimate_0": int((y_train == 0).sum()),
            "phishing_1": int((y_train == 1).sum()),
        },
        "unseen_hostname_subset": metrics.get("unseen_hostname_subset"),
        "cross_validation_on_train": cv_result,
        "training": train_info,
        "model_params": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                             else str(v)) for k, v in clf_params.items()},
        "pipeline_steps": [name for name, _ in pipeline.steps],
        "pipeline_params": {
            k: (v if isinstance(v, (int, float, str, bool, type(None)))
                else " ".join(repr(v).split())[:200])
            for k, v in sorted(pipeline.get_params(deep=True).items())
            if k not in ("steps", "preprocessor__transformers")
        },
        "feature_schema": {
            "raw_feature_count": len(FEATURE_NAMES),
            "raw_feature_names_in_order": list(FEATURE_NAMES),
        },
        "top_coefficients": top_coefficients(pipeline),
        "contract_checks": checks,
        "split": {
            "method": "stratified 80/20 by is_phishing (Person 2's saved split)",
            "seed": RANDOM_STATE,
            "train_file": TRAIN_CSV.as_posix(),
            "test_file": TEST_CSV.as_posix(),
            "train_sha256": _sha256(project_root / TRAIN_CSV),
            "test_sha256": _sha256(project_root / TEST_CSV),
        },
        "dataset": {
            "name": manifest.get("dataset_name"),
            "uci_dataset_id": manifest.get("uci_dataset_id"),
            "license": manifest.get("license"),
            "doi": manifest.get("doi"),
            "raw_export_sha256": manifest.get("exported_csv_sha256"),
        },
        "package_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    report["business_metric_inputs_for_person4"] = business_metric_inputs(report)
    report["dataset_artifact_check"] = dataset_artifact_check(
        X_train, y_train, X_test, y_test)

    log("[6/6] Saving outputs ...")
    if save_model:
        artifact = project_root / MODEL_ARTIFACT
        artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, artifact, compress=3)
        report["model_artifact"] = {
            "path": MODEL_ARTIFACT.as_posix(),
            "bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
            "load_note": "Load with joblib from the project root so that "
                         "'src.url_features' can be imported.",
        }

    metrics_path = project_root / METRICS_JSON
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_confusion_matrix_png(report, project_root / CONFUSION_PNG)
    write_baseline_notes(report, project_root / NOTES_MD)

    log("")
    log("Held-out test set, threshold 0.50, positive class = phishing (1)")
    log(f"  Confusion matrix [[TN, FP], [FN, TP]] = "
        f"[[{report['tn']}, {report['fp']}], [{report['fn']}, {report['tp']}]]")
    log(f"  Precision (phishing): {report['precision_phishing']:.4f}")
    log(f"  Recall    (phishing): {report['recall_phishing']:.4f}")
    log(f"  F1        (phishing): {report['f1_phishing']:.4f}")
    log(f"  Accuracy  (supplementary): {report['accuracy']:.4f}")
    return pipeline, report


def smoke_test_artifact(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Reload the saved Pipeline and score a few inert sample strings."""
    pipeline = joblib.load(project_root / MODEL_ARTIFACT)
    proba, pred = predict_at_threshold(pipeline, pd.Series(SAMPLE_URLS))
    return pd.DataFrame({
        "url_text_inert": SAMPLE_URLS,
        "phishing_score": np.round(proba, 4),
        "predicted_class": pred,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip the 5-fold stability check on training data.")
    parser.add_argument("--no-save-model", action="store_true",
                        help="Do not write the joblib artifact.")
    args = parser.parse_args()

    run_baseline(run_cv=not args.skip_cv, save_model=not args.no_save_model)
    if not args.no_save_model:
        print("\nArtifact reload check on inert sample strings:")
        print(smoke_test_artifact().to_string(index=False))


if __name__ == "__main__":
    main()
