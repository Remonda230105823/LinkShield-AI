# Phase 1 Requirement Traceability

## 1. Purpose

This document maps the Phase 1 project requirements to their corresponding implementation files and evidence in the LinkShield-AI repository.

## 2. Requirement Mapping

| Phase 1 Requirement | Project Evidence | Status |
|---|---|---|
| Problem definition | `docs/problem_market_case.md` | Completed |
| Market case | `docs/problem_market_case.md` | Completed |
| Dataset selection and description | `docs/dataset_card.md` | Completed |
| Data audit and exploration | `notebooks/01_data_audit_eda.ipynb`, `reports/eda_summary.json`, `reports/eda_observations.md` | Completed |
| Data preprocessing | `src/data_cleaning.py`, `src/preprocessing.py`, `notebooks/02_feature_pipeline.ipynb` | Completed |
| Leak-free data splitting | `artifacts/split_manifest.json`, `reports/leakage_check.md` | Completed |
| Baseline model | `src/train_baseline.py`, `notebooks/03_baseline_model.ipynb` | Completed |
| Baseline evaluation | `reports/phase1_baseline_metrics.json`, `reports/phase1_confusion_matrix.png` | Completed |
| Business metric | `docs/business_metric_assumptions.md` | Completed |
| Documentation | `README.md`, `docs/`, `reports/` | Completed |

## 3. Evaluation Evidence

The baseline model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Confusion Matrix

The reported results are available in `reports/phase1_baseline_metrics.json`.

## 4. Scope and Limitations

Phase 1 delivers a dataset preparation pipeline, a baseline Logistic Regression model, and its initial evaluation.

Advanced model development, hyperparameter tuning, explainability improvements, and further system integration are outside the current Phase 1 baseline scope.

The dataset results should not be interpreted as proof of production readiness in real banking environments.

## 5. Conclusion

The repository contains evidence for the main Phase 1 requirements, including the problem definition, dataset, preprocessing pipeline, baseline model, evaluation, and supporting documentation.

This traceability document provides a reference for reviewing the delivered work.