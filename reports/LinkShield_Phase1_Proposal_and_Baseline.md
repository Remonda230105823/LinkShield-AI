# LinkShield-AI
## Phase 1: Proposal and Baseline Report

## 1. Project Overview

LinkShield-AI is an AI-based phishing URL detection project designed to identify potentially malicious URLs and support safer online banking experiences.

The project uses URL-based features and a Machine Learning classification model to distinguish between legitimate and phishing URLs.

## 2. Problem Statement

Phishing attacks use deceptive URLs to trick users into visiting malicious websites and revealing sensitive information.

In banking environments, identifying suspicious URLs is important for reducing exposure to phishing threats. Manual inspection of large numbers of URLs can also create additional workload for security teams.

LinkShield-AI aims to provide an initial automated approach for classifying URLs.

## 3. Dataset

The project uses the PhiUSIIL Phishing URL Dataset from the UCI Machine Learning Repository (Dataset ID: 967).

The dataset is used to train and evaluate a binary classification model:

- 0: Legitimate URL
- 1: Phishing URL

The dataset is licensed under CC BY 4.0.

## 4. Data Preparation

The project includes:

- Data cleaning and validation.
- URL feature extraction.
- Preprocessing of numerical and categorical features.
- Train/test separation.
- Leakage checks to reduce the risk of information leakage.

The preprocessing and feature pipeline are documented in the project notebooks and source files.

## 5. Baseline Model

The Phase 1 baseline uses Logistic Regression.

The pipeline consists of:

1. URL feature extraction.
2. Feature preprocessing.
3. Logistic Regression classification.

The model uses balanced class weights, random seed 42, and a classification threshold of 0.5.

No hyperparameter tuning or SMOTE was applied in this baseline.

## 6. Baseline Evaluation

The model was evaluated on 47,072 test URLs.

| Metric | Result |
|---|---:|
| Accuracy | 99.798% |
| Precision | 99.995% |
| Recall | 99.532% |
| F1-score | 99.763% |
| ROC-AUC | 99.886% |
| Average Precision | 99.910% |

### Confusion Matrix

| | Predicted Legitimate | Predicted Phishing |
|---|---:|---:|
| Actual Legitimate | 26,969 | 1 |
| Actual Phishing | 94 | 20,008 |

The results show strong performance on the available test dataset.

## 7. Business Metric

The selected business metric is Estimated Analyst Review Time.

The estimated workload per 10,000 screened URLs is calculated using:

`Review Hours = ((TP + FP) * M) / 60`

Where M represents the assumed review time per alert in minutes.

| Review Time per Alert | Estimated Hours per 10,000 URLs |
|---|---:|
| 1 minute | 70.8 |
| 3 minutes | 212.5 |
| 5 minutes | 354.2 |

These are illustrative estimates, not measured results from a real bank.

## 8. Limitations and Risks

The high baseline metrics must be interpreted carefully.

The dataset contains noticeable differences between legitimate and phishing URLs. For example, all legitimate URLs in the test set use HTTPS and start with www, while the phishing class has different distributions.

A simple rule based on these characteristics already achieves strong performance. Therefore, the baseline results may partly reflect dataset-specific patterns rather than general phishing detection ability.

Other limitations include:

- The dataset does not represent actual bank traffic.
- The model has not been validated in a production banking environment.
- Analyst review times are assumptions.
- False negatives remain possible.
- The baseline threshold has not been optimized.

## 9. Future Work

Potential Phase 2 improvements include:

- Evaluating additional Machine Learning models.
- Improving generalization using more representative datasets.
- Performing controlled hyperparameter tuning.
- Adding explainability methods.
- Developing an integrated demonstration system.
- Evaluating operational performance under realistic conditions.

## 10. Conclusion

LinkShield-AI Phase 1 establishes the problem, dataset, preprocessing pipeline, and initial Logistic Regression baseline.

The evaluation provides a starting point for further development. However, additional validation is necessary before considering real-world banking deployment.

## 11. Supporting Project Files

- `docs/problem_market_case.md`
- `docs/dataset_card.md`
- `docs/feature_dictionary.md`
- `docs/business_metric_assumptions.md`
- `docs/phase1_requirement_traceability.md`
- `reports/phase1_baseline_metrics.json`
- `reports/phase1_baseline_notes.md`
- `reports/phase1_confusion_matrix.png`