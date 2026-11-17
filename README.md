# LinkShield AI — AI Phishing & Malicious URL Detector for Banks

> [!NOTE]
> **Academic Prototype Disclaimer:** LinkShield AI is an academic research and engineering prototype developed as part of a university graduation project. It is designed to assist bank security operations (SecOps) and email-security analysts with pre-click triage. It is **not** a production-ready bank security system, nor does it replace existing enterprise defence-in-depth controls, email filters, or active sandbox inspection.

---

## 1. Project Overview

LinkShield AI classifies URLs as legitimate or phishing based **exclusively on lexical features** extracted from the URL text string.

### Core Architecture Principles:
- **Zero-Network Policy:** URLs are treated strictly as inert text strings. The system never opens, resolves, pings, scans, or sends HTTP requests to any submitted link.
- **Purely Lexical Signals:** Feature extraction avoids dependence on volatile external lookups (WHOIS, DNS, domain age, reputation APIs).
- **Security Operations Context:** Aimed at retail bank security operations centres (SOC) to prioritize suspicious alerts from customer submissions or automated mail filters.

---

## 2. Dataset Source & Attribution

- **Dataset Name:** PhiUSIIL Phishing URL Dataset (Website)
- **Repository:** UCI Machine Learning Repository (Dataset ID: 967)
- **Authors:** Arvind Prasad and Shalini Chandra (2024)
- **DOI:** [10.1016/j.cose.2023.103545](https://doi.org/10.1016/j.cose.2023.103545)
- **License:** CC BY 4.0
- **Benchmark Size:** 235,795 raw records (134,850 legitimate, 100,945 phishing).

---

## 3. Project Roadmap & Team Roles

| Stage | Role | Focus Area | Status | Deliverables |
|---|---|---|---|---|
| **Phase 1: Step 1** | **Person 1** | Dataset Research, EDA & Manifest | **Completed** | Problem statement, dataset card, manifest, initial EDA notebook and reports. |
| **Phase 1: Step 2** | **Person 2** | Feature Engineering & Preprocessing | **Completed** | Cleaning pipeline, deduplication, train/test split, 25-feature extractor, preprocessor, 63 unit tests, leakage audit, feature dictionary, notebook 02. |
| **Phase 1: Step 3** | **Person 3** | Baseline Model Development | **Completed** | Logistic Regression baseline (`src/train_baseline.py`), notebook 03, metrics JSON, confusion matrix, baseline notes, fitted Pipeline artifact, 10 unit tests, handover to Person 4. |
| **Phase 2** | **Person 4** | Advanced Models & Ensemble | *Upcoming* | Tree ensembles, neural architectures, error analysis. |

---

## 4. Completed Work

### Person 1 Completed:
- Formalized retail bank market case and operational problem statement.
- Curated and validated the PhiUSIIL dataset via official `ucimlrepo` retrieval.
- Conducted exploratory data analysis on URL lengths, character profiles, and class distributions.
- Authored `docs/problem_market_case.md`, `docs/dataset_card.md`, and `reports/eda_observations.md`.

### Person 2 Completed:
- **Data Validation & Cleaning (`src/data_cleaning.py`):** Schema validation, whitespace stripping, and conservative RFC-compliant normalization.
- **Deduplication:** Removed 439 duplicate instances across 439 normalized duplicate groups (all uniformly phishing), while ensuring zero label conflicts. Final clean dataset: **235,356 rows**.
- **Stratified Partitioning:** Executed deterministic 80/20 train/test split (`random_state=42`) stratified by `is_phishing`.
  - Train: 188,284 URLs (107,880 legitimate, 80,404 phishing).
  - Test: 47,072 URLs (26,970 legitimate, 20,102 phishing).
- **Feature Engineering (`src/url_features.py`):** Implemented `URLFeatureExtractor`, extracting 25 lexical features (16 structural, 4 hostname, 4 lexical/entropy/keyword, 1 categorical TLD) with zero external network requests.
- **Preprocessing Utilities (`src/preprocessing.py`):** Built `build_preprocessor()` producing a scikit-learn `ColumnTransformer` with `StandardScaler` and `OneHotEncoder(handle_unknown="ignore")`.
- **Quality Assurance & Testing (`tests/`):** Implemented and verified **63 unit tests** across edge cases, IPs, malformed URLs, determinism, unseen TLDs, and split isolation.
- **Leakage Audit (`reports/leakage_check.md`):** Confirmed zero URL and normalized URL leakage across splits, verified zero target/preprocessing leakage, and documented hostname overlap limitations.
- **Handover Documentation (`docs/handover/person2_to_person3.md`):** Complete technical specifications and code templates for Person 3.

### Person 3 Completed:
- **Baseline training (`src/train_baseline.py`):** one scikit-learn `Pipeline` (`URLFeatureExtractor` → `build_preprocessor()` → `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)`), fitted on `train.csv` only. No hyperparameter or threshold tuning (Phase 2).
- **Contract checks before training:** label convention, zero train/test URL overlap, and proof that `train.csv`/`test.csv` are exactly reproduced from `cleaned_urls.csv` with the agreed stratified 80/20 split (seed 42).
- **Single held-out evaluation at threshold 0.50** (positive class = phishing, label order `[0, 1]`): TN 26,969 / FP 1 / FN 94 / TP 20,008 — precision 0.99995, recall 0.99532, F1 0.99763, accuracy 0.99798 (supplementary).
- **Important caveat:** every legitimate URL in this benchmark is a bare `https://www.<host>` home page, so a one-line rule with no machine learning already reaches F1 0.9885 on the same test set. The baseline score reflects the dataset and is **not** evidence of real-world performance. See `reports/phase1_baseline_notes.md`.
- **Outputs:** `reports/phase1_baseline_metrics.json`, `reports/phase1_confusion_matrix.png`, `reports/phase1_baseline_notes.md`, `artifacts/linkshield_logreg_baseline.joblib`, `notebooks/03_baseline_model.ipynb`, `tests/test_train_baseline.py`, `docs/handover/person3_to_person4.md`.

---

## 5. Project Directory Structure

```text
LinkShield-AI/
│
├── data/
│   ├── raw/
│   │   └── person1_raw_labeled_urls.csv    # Raw benchmark dataset (Git-excluded)
│   ├── processed/
│   │   ├── cleaned_urls.csv                # 235,356 cleaned & deduplicated URLs
│   │   ├── train.csv                       # 188,284 training records (80%)
│   │   └── test.csv                        # 47,072 testing records (20%)
│   ├── README.md                           # How to obtain and rebuild the data
│   └── manifest.json                       # Person 1 dataset manifest
│
├── artifacts/
│   ├── split_manifest.json                 # Person 2 split sizes, seed, class counts
│   └── linkshield_logreg_baseline.joblib   # Fitted baseline Pipeline (Person 3)
│
├── docs/
│   ├── dataset_card.md                     # Dataset documentation & provenance
│   ├── problem_market_case.md              # SecOps business & problem context
│   ├── feature_dictionary.md               # 25-feature dictionary & math formulas
│   └── handover/
│       ├── person1_to_person2.md           # Person 1 handover notes
│       ├── person2_to_person3.md           # Person 2 handover instructions
│       └── person3_to_person4.md           # Person 3 handover to Person 4
│
├── notebooks/
│   ├── 01_data_audit_eda.ipynb             # Person 1 exploratory data analysis
│   ├── 02_feature_pipeline.ipynb        # Person 2 feature engineering & verification
│   └── 03_baseline_model.ipynb             # Person 3 Logistic Regression baseline
│
├── reports/
│   ├── eda_class_distribution.png          # Person 1 EDA chart
│   ├── eda_url_length_distribution.png     # Person 1 EDA chart
│   ├── eda_summary.json                    # Person 1 EDA numbers
│   ├── eda_observations.md                 # Person 1 EDA observations
│   ├── data_quality_summary.json           # Person 2 cleaning row-count report
│   ├── leakage_check.md                    # Person 2 leakage audit
│   ├── phase1_baseline_metrics.json        # Person 3 measured baseline metrics
│   ├── phase1_confusion_matrix.png         # Person 3 confusion matrix image
│   └── phase1_baseline_notes.md            # Person 3 interpretation & limitations
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py                    # Cleaning & normalization engine
│   ├── features.py                         # URLFeatureExtractor transformer
│   ├── preprocessing.py                    # build_preprocessor() ColumnTransformer
│   └── train_baseline.py                   # Person 3 baseline training & evaluation
│
├── tests/
│   ├── __init__.py
│   ├── test_data_cleaning.py               # Cleaning & normalization unit tests
│   ├── test_url_features.py                    # Feature extractor & transformer unit tests
│   └── test_train_baseline.py              # Baseline training & metric unit tests
│
├── requirements.txt                        # Open-source Python dependencies
├── .gitignore                              # Git exclusion rules
└── README.md                               # Project documentation (this file)
```

---

## 6. Installation & Environment Setup

1. **Clone repository and create a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 7. Running Unit Tests

Execute the automated test suite using `pytest`:
```bash
pytest tests/ -v
```
All **76 tests** should pass (66 for cleaning and features, 10 for the baseline), verifying schema compliance, normalization invariants, URL edge cases (empty strings, IP hosts, missing schemes, long payloads), and unknown TLD handling.

---

## 8. Usage: Feature Extraction & Preprocessing Pipeline

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.url_features import URLFeatureExtractor
from src.preprocessing import build_preprocessor

# 1. Load data
train_df = pd.read_csv("data/processed/train.csv")
X_train, y_train = train_df["url"], train_df["is_phishing"]

# 2. Build full pipeline
pipeline = Pipeline([
    ("url_features", URLFeatureExtractor()),
    ("preprocessor", build_preprocessor()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# 3. Fit pipeline on training data
pipeline.fit(X_train, y_train)

# 4. Predict on arbitrary URLs
new_urls = pd.Series([
    "https://login-verify-account.bank-security.com/update",
    "https://www.wikipedia.org"
])
predictions = pipeline.predict(new_urls)
print(predictions)  # 1 = Phishing, 0 = Legitimate
```

---

## 9. Running the Phase 1 Baseline (Person 3)

From the project root:
```bash
python -m src.train_baseline
```
This verifies the modelling contract, fits the Pipeline on training data only, evaluates once on the held-out test set at threshold 0.50, and rewrites the metrics JSON, confusion-matrix image, baseline notes and the joblib artifact. `notebooks/03_baseline_model.ipynb` shows the same run step by step.

## 10. Next Steps for Person 4

See `docs/handover/person3_to_person4.md` for results, run instructions, limitations and the open items for integration, the business metric, the report and the Phase 1 talk.
