"""
Unit tests for the Phase 1 baseline (Person 3).

These tests use small synthetic, inert URL strings so they run in seconds
and never touch the real train/test files or the network.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import train_baseline as tb


@pytest.fixture(scope="module")
def toy_data():
    legit = [f"https://www.site{i}.example.com" for i in range(40)]
    phish = [f"http://login-verify{i}.example.net/account/update?id={i}" for i in range(40)]
    X = pd.Series(legit + phish)
    y = pd.Series([0] * 40 + [1] * 40)
    return X, y


@pytest.fixture(scope="module")
def fitted(toy_data):
    X, y = toy_data
    return tb.train_baseline(X, y)


def test_fixed_phase1_configuration():
    assert tb.RANDOM_STATE == 42
    assert tb.THRESHOLD == 0.50
    assert tb.LABELS == [0, 1]
    assert tb.POSITIVE_LABEL == 1


def test_pipeline_structure_and_model_settings():
    pipe = tb.build_baseline_pipeline()
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["url_features", "preprocessor", "classifier"]
    clf = pipe.named_steps["classifier"]
    assert isinstance(clf, LogisticRegression)
    assert clf.class_weight == "balanced"
    assert clf.max_iter == 1000
    assert clf.random_state == 42


def test_pipeline_is_unfitted_until_fit_is_called():
    pipe = tb.build_baseline_pipeline()
    assert not hasattr(pipe.named_steps["classifier"], "coef_")


def test_training_reports_convergence(fitted):
    _, info = fitted
    assert info["train_rows"] == 80
    assert info["converged"] is True
    assert info["convergence_warnings"] == []
    assert info["n_iter_used"] <= info["max_iter"]


def test_predict_at_threshold_uses_phishing_probability(fitted, toy_data):
    pipe, _ = fitted
    X, _ = toy_data
    proba, pred = tb.predict_at_threshold(pipe, X, 0.5)
    assert proba.shape == (80,) and ((proba >= 0) & (proba <= 1)).all()
    assert np.array_equal(pred, (proba >= 0.5).astype(int))
    # Column used must be the probability of class 1 (phishing).
    assert np.allclose(proba, pipe.predict_proba(X)[:, 1])


def test_score_confusion_matrix_order_and_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0])
    s = tb._score(y_true, y_pred)
    assert (s["tn"], s["fp"], s["fn"], s["tp"]) == (2, 1, 2, 2)
    assert s["precision_phishing"] == pytest.approx(2 / 3)
    assert s["recall_phishing"] == pytest.approx(2 / 4)
    assert s["f1_phishing"] == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))
    assert s["accuracy"] == pytest.approx(4 / 7)


def test_evaluate_counts_are_consistent(fitted, toy_data):
    pipe, _ = fitted
    X, y = toy_data
    m = tb.evaluate(pipe, X, y)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == m["test_rows"] == 80
    assert m["test_class_counts"] == {"legitimate_0": 40, "phishing_1": 40}
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_business_metric_formula():
    out = tb.business_metric_inputs({"test_rows": 20_000, "tp": 900, "fp": 100, "fn": 50})
    assert out["alerts_raised_tp_plus_fp"] == pytest.approx(500)
    assert out["missed_phishing_fn"] == pytest.approx(25)
    hours = out["review_hours_by_assumed_minutes_per_alert"]
    assert hours["3_min"] == pytest.approx(500 * 3 / 60)


def test_verify_contract_rejects_train_test_overlap(toy_data):
    X, y = toy_data
    with pytest.raises(AssertionError):
        tb.verify_contract(X, y, X, y, project_root=Path("__no_such_dir__"))


def test_saved_metrics_json_is_internally_consistent():
    """If the real run has been executed, its JSON must add up."""
    path = tb.PROJECT_ROOT / tb.METRICS_JSON
    if not path.exists():
        pytest.skip("Baseline has not been run yet.")
    m = json.loads(path.read_text(encoding="utf-8"))
    assert m["positive_label"] == 1 and m["threshold"] == 0.5
    assert m["confusion_matrix_label_order"] == ["legitimate_0", "phishing_1"]
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == m["test_rows"]
    assert m["precision_phishing"] == pytest.approx(m["tp"] / (m["tp"] + m["fp"]))
    assert m["recall_phishing"] == pytest.approx(m["tp"] / (m["tp"] + m["fn"]))
    assert "FILL_AFTER_RUNNING" not in path.read_text(encoding="utf-8")
