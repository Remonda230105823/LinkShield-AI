# Leakage Audit Report — LinkShield AI

**Date:** 2026-11-16  
**Auditor:** Person 2 (Feature Engineering & Preprocessing)  
**Dataset:** PhiUSIIL Phishing URL Dataset (UCI ID 967)  
**Split Seed:** 42 (80% Train, 20% Test, Stratified by `is_phishing`)

---

## 1. Executive Summary

| Audit Dimension | Result | Status | Notes |
|---|---|---|---|
| **1. Exact URL Overlap** | **0** URLs | **PASS (No Leakage)** | Zero identical URL strings appear in both train and test |
| **2. Normalized URL Overlap** | **0** URLs | **PASS (No Leakage)** | Zero normalized duplicate URLs cross the train/test boundary |
| **3. Hostname Overlap** | 2046 hostnames (4.53% of test hostnames) | **REPORTED (Expected Limitation)** | Expected in standard stratified URL splits; documented below |
| **4. Feature Extraction Leakage** | **None** | **PASS (No Leakage)** | All 25 features are purely lexical functions of inert URL strings without target/label access |
| **5. Preprocessing Leakage** | **None** | **PASS (No Leakage)** | Zero scalers, imputers, or encoders were fitted before splitting. `build_preprocessor()` produces an unfitted transformer for Person 3 |

---

## 2. Detailed Verification

### Check 1: Exact URL Overlap
- **Definition:** Exact string equality `url_train == url_test`.
- **Finding:** Exactly **0** overlapping URLs out of 188,284 training URLs and 47,072 testing URLs.
- **Conclusion:** Perfect partition at the raw string level.

### Check 2: Normalized URL Overlap
- **Definition:** Functional equality under conservative RFC-compliant URL normalization (scheme casing, hostname casing, default port removal, whitespace trimming, and empty parameter handling).
- **Finding:** Exactly **0** overlapping normalized URLs.
- **Conclusion:** No semantic or functional URL duplicates leaked across splits.

### Check 3: Hostname Overlap Analysis (Distributional Limitation)
- **Train unique hostnames:** 176,923
- **Test unique hostnames:** 45,200
- **Overlapping hostnames:** 2,046 (4.53% of test hostnames)
- **Test rows with shared hostnames:** 3,745 / 47,072 (7.96%)

#### Why Hostname Overlap Occurs and Why It Is Expected:
1. In the PhiUSIIL benchmark dataset, multiple URLs originate from major legitimate platforms (e.g., `google.com`, `wikipedia.org`, `github.com`) and specific phishing campaigns with multiple targeted landing paths.
2. The project specification strictly mandates a **primary stratified split** (80% train / 20% test, `random_state=42`, stratified by `is_phishing`).
3. Standard stratified splitting samples at the URL record level, not by group (domain). As instructed, we do not silently change the required primary split into a domain-grouped split.

#### Operational Impact & Risk for Model Evaluation:
- **Lexical Generalization vs. Memorization:** High hostname overlap means that if a model learns features strongly tied to specific hostnames (or TLDs), test evaluation may slightly overestimate performance on brand-new, never-before-seen domains (zero-day phishing infrastructure).
- **Recommendation for Person 3 & Person 4:** Person 3 should evaluate on the primary held-out test set as required, but should also consider an optional diagnostic split (e.g., GroupKFold on registered domain) to measure the domain-generalization gap.

### Check 4: Feature Extraction Leakage
- `URLFeatureExtractor` is strictly stateless (`fit()` is a no-op).
- Features rely solely on character counts, string lengths, Shannon entropy, regex patterns, and standard-library parsing.
- The feature calculation has zero access to `y` (the target column `is_phishing`) or global corpus statistics.

### Check 5: Learned Preprocessing Leakage
- No `StandardScaler`, `SimpleImputer`, or `OneHotEncoder` was fitted during Phase 2.
- Preprocessing utilities in `src/preprocessing.py` are exported as an unfitted `ColumnTransformer` via `build_preprocessor()`.
- Person 3 will fit the full pipeline exclusively on `train.csv`.

---

## 3. Handbook checklist — explicit answers

Added during the final Phase 1 check so that every question in handbook
Section 8.3 (Task 6) has a direct answer. Items 3 and 6 were verified against
the code and against Person 3's baseline run.

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Were train/test rows split before any learned transformation was fitted? | **Yes** | The split is made in notebook 02 on cleaned rows; the first `.fit` happens later, inside the Pipeline. |
| 2 | Is exact normalized URL overlap zero? | **Yes, 0** | Check 2 above; re-checked on every run by `src/train_baseline.py::verify_contract`. |
| 3 | Were the dataset's precomputed features excluded? | **Yes** | Only `URL` and `label` are taken from UCI (notebook 01). Every file in `data/` has exactly the columns `url` and `is_phishing`; the other 53 UCI feature columns and `FILENAME` never enter the project. |
| 4 | Is the target excluded from features? | **Yes** | `URLFeatureExtractor` receives URL strings only; `is_phishing` is not in `FEATURE_NAMES` (asserted in `verify_contract`). |
| 5 | Was any scaler/encoder/imputer fitted on all rows? | **No** | `build_preprocessor()` returns an unfitted `ColumnTransformer`; it is fitted only on `train.csv` inside the Pipeline. |
| 6 | Was the test set used in any fitting or selection? | **No** | Notebook 02 fits its smoke-test Pipeline on a 5,000-row sample of the training split only. Person 3's baseline is fitted on `train.csv` only, with fixed settings and a fixed 0.50 threshold; the test set was used for evaluation only. |
| 7 | Are train/test class proportions recorded? | **Yes** | `artifacts/split_manifest.json` (42.70% phishing in both). |
| 8 | Was hostname overlap checked? | **Yes — measured, not zero** | Check 3 above: 2,046 shared hostnames, 3,745 test rows (7.96%). Reported as a limitation; zero domain overlap is **not** claimed. |
