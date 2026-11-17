# Person 2 → Person 3 Handover

**Project:** LinkShield AI — AI Phishing and Malicious URL Detector for Banks  
**Phase:** Phase 1 — Feature Engineering & Preprocessing Completion  
**Author:** Person 2 (Feature Engineering & Preprocessing Engineer)  
**Recipient:** Person 3 (Baseline Model Development Engineer)  
**Date:** 2026-11-16  

---

## 1. Executive Summary & Purpose

This handover transfers the cleaned, deduplicated, and audited datasets, as well as production-ready, scikit-learn-compatible feature extraction and preprocessing pipelines from **Person 2** to **Person 3**.

Person 3's core responsibility is to develop, train, tune, and evaluate the **Logistic Regression baseline classifier** strictly on the held-out test set using the standardized pipeline interfaces provided here.

---

## 2. Dataset Handover & Partition Specifications

All processed datasets are saved in `data/processed/` and contain **only two columns**: `url` (raw text string) and `is_phishing` (integer label in `{0, 1}`).

### A. Dataset Paths and Record Counts

| Dataset File | File Path | Record Count | Legitimate (`0`) | Phishing (`1`) | Phishing Ratio |
|---|---|---|---|---|---|
| **Raw Labeled Dataset** | `data/raw/person1_raw_labeled_urls.csv` | 235,795 | 134,850 | 100,945 | 42.81% |
| **Cleaned & Deduplicated** | `data/processed/cleaned_urls.csv` | 235,356 | 134,850 | 100,506 | 42.70% |
| **Training Split (80%)** | `data/processed/train.csv` | 188,284 | 107,880 | 80,404 | 42.70% |
| **Testing Split (20%)** | `data/processed/test.csv` | 47,072 | 26,970 | 20,102 | 42.70% |
| **Quarantined Rows** | *(None)* | 0 | 0 | 0 | N/A |

### B. Label Mapping Convention
The dataset follows the project convention:
- `is_phishing = 1`: **Phishing / Malicious** URL.
- `is_phishing = 0`: **Legitimate** URL.

> [!WARNING]
> The original UCI dataset used `label=1` for legitimate and `label=0` for phishing. The files provided to you have already been inverted (`is_phishing = 1 - original_label`). **DO NOT** invert the labels again.

### C. Data Cleaning & Deduplication Decisions
1. **Schema Validation:** Strict verification confirmed that all records contain non-null string URLs and valid binary labels `{0, 1}`.
2. **Whitespace Handling:** URLs underwent surrounding whitespace trimming. No internal characters, query parameters, or paths were modified.
3. **Conservative URL Normalization:** A deterministic normalization function lowercased schemes and hostnames, stripped default ports (80 for HTTP, 443 for HTTPS), and preserved case-sensitive paths, parameters, and fragments.
4. **Duplicate Analysis:**
   - 425 exact string duplicates were detected.
   - 439 normalized duplicate groups (totaling 878 rows) were detected.
   - All 439 duplicate occurrences had identical labels (`is_phishing=1`).
   - One deterministic representative (first occurrence) was retained for each group, removing 439 duplicate rows.
   - **0 conflicting-label groups** were detected (hence 0 quarantined rows).

### D. Train/Test Split Methodology
- **Strategy:** Stratified sampling preserving exact class distribution (57.30% legitimate, 42.70% phishing).
- **Proportion:** 80% train (188,284 rows) / 20% test (47,072 rows).
- **Random State:** `random_state=42`.
- **Integrity Guarantee:** Split was performed **strictly after** cleaning and deduplication, and **before** fitting any feature scalers or encoders.

---

## 3. Leakage Audit Findings & Modelling Limitations

The leakage audit (`reports/leakage_check.md`) established:
1. **Exact URL Overlap:** Exactly **0** URLs appear in both train and test sets.
2. **Normalized URL Overlap:** Exactly **0** normalized URLs appear in both train and test sets.
3. **No Target Leakage:** Feature extractors have zero access to the target column `is_phishing`.
4. **No Preprocessing Leakage:** No transformer, scaler, or imputer was fitted prior to the split.
5. **Hostname Overlap Limitation (Important):**
   - 2,046 hostnames appear in both training and test sets (4.53% of test hostnames).
   - 3,745 test rows (7.96% of the test set) originate from these overlapping hostnames.
   - *Implication for Person 3:* Standard stratified splitting samples at the URL level, reflecting benchmark conditions. However, test set accuracy may slightly overestimate performance on completely unseen, novel domain infrastructure. Person 3 is advised to document this limitation in their evaluation report.

---

## 4. Reusable Component Interfaces

### A. `URLFeatureExtractor` (`src/url_features.py`)
A scikit-learn-compatible `BaseEstimator` / `TransformerMixin` that parses raw URL strings and extracts 25 deterministic lexical features:
- **16 URL Structure Features:** `url_length`, `domain_length`, `path_length`, `query_length`, `fragment_length`, `num_dots`, `num_hyphens`, `num_underscores`, `num_slashes`, `num_at_symbols`, `num_question_marks`, `num_ampersands`, `num_equal_signs`, `num_percent_signs`, `num_digits`, `num_special_chars`.
- **4 Hostname Features:** `num_subdomains`, `has_ip`, `has_https`, `has_www`.
- **4 Other Lexical Features:** `has_fragment`, `has_query`, `url_entropy`, `suspicious_keyword_count`.
- **1 Categorical Feature:** `tld`.

```python
from src.url_features import URLFeatureExtractor
import pandas as pd

extractor = URLFeatureExtractor()
sample_urls = pd.Series([
    "https://secure-bank-login.example.com/auth?session=1",
    "http://192.168.1.1/admin"
])

features_df = extractor.transform(sample_urls)
print(features_df.shape)  # (2, 25)
```

### B. `build_preprocessor()` (`src/preprocessing.py`)
Returns an unfitted `ColumnTransformer` configured as:
- **Numerical Pipeline:** `SimpleImputer(strategy="constant", fill_value=0)` → `StandardScaler()` applied to the 24 numerical features.
- **Categorical Pipeline:** `SimpleImputer(strategy="constant", fill_value="unknown")` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` applied to `tld`.

```python
from src.preprocessing import build_preprocessor

preprocessor = build_preprocessor()
# Unfitted transformer ready to be placed inside a pipeline
```

---

## 5. Verified End-to-End Pipeline Integration Example

Person 3 can directly use this verified template to train and evaluate the baseline Logistic Regression model:

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from src.url_features import URLFeatureExtractor
from src.preprocessing import build_preprocessor

# 1. Load datasets
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train = train_df["url"]
y_train = train_df["is_phishing"]

X_test = test_df["url"]
y_test = test_df["is_phishing"]

# 2. Assemble unified pipeline
baseline_pipeline = Pipeline([
    ("url_features", URLFeatureExtractor()),
    ("preprocessor", build_preprocessor()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"  # Optional, but recommended for bank security triage
    ))
])

# 3. Fit ONLY on training data
baseline_pipeline.fit(X_train, y_train)

# 4. Predict on held-out test data
y_pred = baseline_pipeline.predict(X_test)
y_prob = baseline_pipeline.predict_proba(X_test)[:, 1]

# 5. Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
```

---

## 6. Manifest of Created Files

| File Path | Description |
|---|---|
| `src/__init__.py` | Package marker for `src`. |
| `src/data_cleaning.py` | Schema validation, whitespace stripping, URL normalization, deduplication, and quarantine pipeline. |
| `src/url_features.py` | Scikit-learn-compatible `URLFeatureExtractor` (25 lexical features). |
| `src/preprocessing.py` | `build_preprocessor()` returning `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`. |
| `tests/__init__.py` | Package marker for `tests`. |
| `tests/test_url_features.py` | 38 unit tests covering all URL edge cases, IPs, keywords, determinism, and unseen TLDs. |
| `tests/test_data_cleaning.py` | 25 unit tests covering validation, whitespace, normalization, deduplication, and leakage. |
| `docs/feature_dictionary.md` | Comprehensive catalog documenting definitions, math formulas, types, phishing rationale, and limitations for all 25 features. |
| `docs/handover/person2_to_person3.md` | This handover document. |
| `data/processed/cleaned_urls.csv` | Cleaned and deduplicated dataset (235,356 rows). |
| `data/processed/train.csv` | Stratified training split (188,284 rows). |
| `data/processed/test.csv` | Stratified testing split (47,072 rows). |
| `reports/data_quality_summary.json` | Detailed metrics and row counts for each cleaning stage. |
| `artifacts/split_manifest.json` | Split proportions, seed, row counts, and class balances. |
| `reports/leakage_check.md` | Comprehensive 5-point leakage audit. |
| `notebooks/02_feature_pipeline.ipynb` | Executed, reproducible Jupyter notebook walking through cleaning, splitting, feature extraction, and pipeline verification. |
| `requirements.txt` | Minimal open-source dependencies. |
| `.gitignore` | Exclusion rules protecting raw data, caches, environments, and secrets. |
| `README.md` | Updated root documentation reflecting Phase 2 completion. |

---

## 7. Execution & Reproduction Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Unit Tests
```bash
pytest tests/ -v
```

### Run Feature Engineering Notebook
```bash
jupyter notebook notebooks/02_feature_pipeline.ipynb
```

---

## 8. Person 3's Next Responsibilities

Person 3 should proceed with the following deliverables:
1. **Model Selection:** Train the baseline `LogisticRegression` model using the provided Pipeline architecture.
2. **Hyperparameter Tuning:** Tune regularizer parameter `C` and penalty options using 5-fold cross-validation on `train.csv` **only**.
3. **Single Held-Out Test Evaluation:** Evaluate the tuned model **once** on `test.csv`.
4. **Metrics Reporting:** Report Phishing Recall (primary), Precision, F1-Score, and ROC-AUC.
5. **Operational Bank Triage Simulation:** Compute analyst workload reduction metrics based on the confusion matrix at threshold `0.50` and alternative thresholds (e.g. high recall vs. high precision operating points).
