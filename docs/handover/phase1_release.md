# LinkShield-AI — Phase 1 Release Handover

## 1. Handover Overview

This document summarizes the Phase 1 deliverables prepared by Person 4
and records the current release status.

**Project:** LinkShield-AI  
**Phase:** Phase 1 — Baseline Model  
**Responsible role:** Person 4 — Integration, Business Metric, Report and Presentation

## 2. Main Deliverables

### Documentation
- `README.md`
- `docs/business_metric_assumptions.md`
- `docs/phase1_requirement_traceability.md`
- `docs/dataset_card.md`
- `docs/feature_dictionary.md`
- `docs/problem_market_case.md`

### End-to-End Integration
- `notebooks/LinkShield_Phase1_End_to_End.ipynb`

The notebook downloads the official UCI dataset, cleans the data using
existing project modules, creates the agreed train/test split, verifies
the modelling contract, trains the baseline model, evaluates its
performance, calculates the business metric and saves the results.

### Report and Presentation
- `reports/LinkShield_Phase1_Proposal_and_Baseline.md`
- `presentations/LinkShield_Phase1.pptx`

## 3. End-to-End Verification Results

The notebook was executed successfully in Google Colab.

A fresh runtime was also tested by Person 4 after disconnecting and
deleting the previous runtime and running the notebook again from
beginning to end.

### Dataset
- Raw rows: 235,795
- Clean rows: 235,356
- Removed duplicate rows: 439
- Quarantined rows: 0
- Training rows: 188,284
- Test rows: 47,072

### Model
- Algorithm: Logistic Regression
- Random state: 42
- Cross-validation: 5 folds on training data only
- Training converged successfully.
- No convergence warnings.

### Held-out Test Results

| Metric | Result |
|---|---:|
| Precision | 99.9950% |
| Recall | 99.5324% |
| F1-score | 99.7632% |
| Accuracy | 99.7982% |
| ROC-AUC | 99.8864% |

### Confusion Matrix

| | Predicted Legitimate | Predicted Phishing |
|---|---:|---:|
| Actual Legitimate | 26,969 | 1 |
| Actual Phishing | 94 | 20,008 |

## 4. Business Metric

The business metric estimates analyst review time per 10,000 screened
URLs using the test-set class distribution.

- Estimated alerts: 4,250.7
- False alarms: 0.2
- Missed phishing URLs: 20.0
- Estimated review time:
  - 1 minute per alert: 70.85 hours
  - 3 minutes per alert: 212.54 hours
  - 5 minutes per alert: 354.23 hours

The review-time assumptions are illustrative scenarios only.
They are not based on bank interviews or a cited operational study.
The test-set class distribution is a benchmark mix, not real bank traffic.

## 5. Limitations

- Evaluation uses the selected UCI dataset and its benchmark distribution.
- The unseen-hostname subset is part of the same held-out test set,
  not an independent external dataset.
- The model uses lexical URL features and does not visit the destination.
- Test performance does not guarantee equivalent performance in
  real banking or production environments.
- Business review-time estimates are illustrative, not measured savings.
- Further external validation is required before any production use.

## 6. Remaining Release Actions

The following actions are still required:

1. Ask another teammate to reproduce the notebook in a fresh runtime.
2. Review the deliverables and obtain team approval.
3. Commit any approved final changes.
4. Push the repository to GitHub.
5. Add the GitHub repository URL to the final handover.

## 7. Current Git Status

- Branch: `master`
- Latest notebook commit: `d40f7ef`
- Commit message: `Add Phase 1 end-to-end notebook`
- Working tree was clean after the commit.

No GitHub push is claimed in this handover.

## 8. Release Decision

**Status: Prepared — Pending Final Release Verification**

The Phase 1 report, presentation, documentation and end-to-end
notebook have been prepared. Person 4's fresh-runtime test passed.

Final release remains subject to independent teammate reproduction,
team approval and GitHub publication.
