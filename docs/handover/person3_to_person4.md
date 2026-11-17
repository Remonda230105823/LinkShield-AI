# Person 3 → Person 4 Handover

**Project:** LinkShield AI — AI Phishing and Malicious URL Detector for Banks
**Phase:** Phase 1 — Logistic Regression baseline and evaluation
**Author:** Person 3 (Baseline Model Development)
**Recipient:** Person 4 (Integration, business metric, report and presentation)
**Date:** 2026-11-16

---

## 1. Summary

The Phase 1 baseline is trained, evaluated once on the held-out test set, saved
and reproducible. This package contains the complete work of Persons 1, 2 and 3
in one project folder. Nothing from Person 1 or Person 2 was modified except
the three shared files listed in Section 7.

**Read Section 5 before writing the report or slides.** The score is very high,
and the reason is a property of the dataset that the team must state openly.

---

## 2. What Person 3 delivers

| File | What it is |
|---|---|
| `src/train_baseline.py` | Reusable training / evaluation logic (contract checks, Pipeline, fit, evaluate, save). |
| `notebooks/03_baseline_model.ipynb` | Executed notebook showing the full baseline run step by step. |
| `reports/phase1_baseline_metrics.json` | Every measured number, model parameters, package versions, split hashes. |
| `reports/phase1_confusion_matrix.png` | Confusion matrix image with class order on the axes. |
| `reports/phase1_baseline_notes.md` | Interpretation for a bank security team, limitations, business-metric inputs. |
| `artifacts/linkshield_logreg_baseline.joblib` | Fitted Pipeline (about 10 KB, safe to commit). |
| `tests/test_train_baseline.py` | 10 unit tests for the baseline code. |
| `docs/handover/person3_to_person4.md` | This file. |

---

## 3. Model and settings

| Item | Value |
|---|---|
| Model | scikit-learn `LogisticRegression`, final step of one `Pipeline` |
| Pipeline | `URLFeatureExtractor` → `build_preprocessor()` → `LogisticRegression` |
| Class imbalance | `class_weight="balanced"` (no SMOTE / resampling) |
| `max_iter` | 1000 — solver converged in 17 iterations, no `ConvergenceWarning` |
| Other parameters | scikit-learn defaults (`C=1.0`, L2, `lbfgs`) — **not tuned** |
| Random seed | 42 |
| Threshold | 0.50 (baseline convention, **not tuned**) |
| Positive class | phishing, `is_phishing = 1` |
| Confusion-matrix label order | `[0, 1]` = `[legitimate, phishing]` |
| Train rows (fit on these only) | 188,284 (107,880 legitimate / 80,404 phishing) |
| Test rows (scored once) | 47,072 (26,970 legitimate / 20,102 phishing) |
| Features | Person 2's 25 lexical URL features, unchanged (580 model columns after one-hot `tld`) |
| Environment of the recorded run | Python 3.12.10, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4, joblib 1.5.3 |

---

## 4. Results on the held-out test set (threshold 0.50)

| | Predicted legitimate (0) | Predicted phishing (1) |
|---|---|---|
| **Actual legitimate (0)** | TN = 26,969 | FP = 1 |
| **Actual phishing (1)** | FN = 94 | TP = 20,008 |

| Metric (phishing = positive) | Value |
|---|---|
| Precision | 0.99995 |
| Recall | 0.99532 |
| F1 | 0.99763 |
| Accuracy (supplementary) | 0.99798 |
| ROC-AUC (supplementary) | 0.99886 |

- 5-fold cross-validation on the **training set only**, same fixed settings,
  nothing tuned: F1 mean 0.9977 (std 0.0002), recall mean 0.9954, precision
  mean 0.9999.
- Test rows whose hostname never appears in training (43,327 rows):
  precision 0.9999, recall 0.9988, F1 0.9994. Hostname overlap is not what
  drives the score.

Exact, unrounded values are in `reports/phase1_baseline_metrics.json`. Quote
from that file, not from this table.

---

## 5. Important caveat — why the score is so high

The result is leak-free, but the benchmark is close to trivially separable:

- In the training data **100% of legitimate URLs are `https`, 100% start with
  `www.`, and 0% have any path beyond `/`** — every legitimate entry is a bare
  home page. Phishing URLs: 48.5% `https`, 41.5% `www.`, 27.2% with a path.
- A one-line rule with no machine learning ("flag as phishing unless the URL is
  `https` and starts with `www.`") scores **precision 1.0000, recall 0.9772,
  F1 0.9885** on the same test set.
- So the baseline (F1 0.9976) is only a small step above a trivial rule. The
  number mostly reflects how the dataset was collected.
- Practical consequence: the saved model scores an ordinary legitimate deep
  link such as the inert string `https://www.example.org/about/team` as
  phishing, because it never saw a legitimate URL with a path.

What this means for Person 4:

1. Report the metrics exactly as measured, together with this caveat. Do not
   describe the model as accurate on real bank traffic or production-ready.
2. Use the trivial-rule comparison in the talk — it is a strong, honest answer
   to "why is your baseline almost perfect?".
3. Put it in the **Phase 2 risks**: Random Forest and boosting will also score
   ~0.99 here, so "measurable gain over the baseline" will be tiny on this
   dataset. The team should discuss adding legitimate URLs that have paths and
   queries (for example from another permitted public source) or evaluating on
   a second dataset, and should plan the robustness test around this weakness.
   That decision belongs to the whole team, so I have not changed the data.

---

## 6. How to reproduce

From the project root (`LinkShield-AI/`):

```bash
pip install -r requirements.txt
```

```bash
python -m pytest tests -q
```

```bash
python -m src.train_baseline
```

- Tests: 76 pass (66 for cleaning and features, 10 for the baseline).
- The training command takes about one minute on a laptop. It rewrites the
  metrics JSON, confusion-matrix image, notes and the joblib artifact, then
  reloads the artifact and scores four inert sample strings.
- Notebook: open `notebooks/03_baseline_model.ipynb` and run all cells. In
  Colab, upload or clone the project folder, `%cd` into it, and install
  `requirements.txt` first.
- Loading the artifact: run from the project root so `src.url_features` can be
  imported, then `joblib.load("artifacts/linkshield_logreg_baseline.joblib")`.
  The Pipeline takes a pandas Series of raw URL strings.

**Not yet done:** the notebook was executed locally (Python 3.12, Windows), not
in a fresh Google Colab runtime. Person 4 should run it once in Colab before
pinning versions in `requirements.txt`. A joblib file saved under one
scikit-learn version may warn or fail under another; if that happens, re-run
`python -m src.train_baseline` to regenerate it.

---

## 7. Shared files Person 3 touched

| File | Change |
|---|---|
| `README.md` | Person 3 status, baseline results, run command, directory tree. |
| `requirements.txt` | Added `joblib`. |
| `.gitignore` | No change. `data/raw/*.csv` stays excluded from Git as Person 2 set it. |
| `data/README.md` | New: how to obtain the dataset and rebuild the processed files. |
| Other Person 1 / Person 2 files | See "Changes made to Person 1 and Person 2 files in the final check" in Section 8. |
| Person 2's files | Renamed to the handbook names (see Section 8, item 2). Names and paths only. |

---

## 8. Things Person 4 should know

1. **Scope difference with Person 2's note.** `person2_to_person3.md` asks
   Person 3 to tune `C` with cross-validation and try alternative thresholds.
   The team handbook (Section 9.4) forbids tuning and threshold selection in
   Phase 1, so I followed the handbook: fixed settings, threshold 0.50. The
   5-fold CV in my run is a stability check only. Tuning is Phase 2.
2. **File names now follow the handbook's folder plan (Section 6.2).** Person 2's
   files were renamed to the handbook names on 2026-11-16, with every import and
   reference updated and the model artifact regenerated:
   `src/features.py` → `src/url_features.py`,
   `tests/test_features.py` → `tests/test_url_features.py`,
   `notebooks/02_feature_engineering.ipynb` → `notebooks/02_feature_pipeline.ipynb`,
   `reports/person2/cleaning_summary.json` → `reports/data_quality_summary.json`,
   `reports/person2/leakage_audit.md` → `reports/leakage_check.md`,
   `reports/person2/split_summary.json` → `artifacts/split_manifest.json`.
   Only names and paths changed — no logic, data, split or feature was altered.
   `data/README.md` was added. Person 2 should be told, because their own copy
   still uses the old names. Extra files the handbook does not list were kept:
   `docs/feature_dictionary.md`, `tests/test_data_cleaning.py`,
   `tests/test_train_baseline.py`, `reports/eda_observations.md`.
3. **Feature list differs slightly from handbook Section 3.4.** Person 2
   delivers 25 features (adds `fragment_length`, `num_underscores`,
   `num_special_chars`, `has_www`, `has_fragment`, `has_query`; keyword feature
   is a count, not a flag). The report should describe the 25 features in
   `docs/feature_dictionary.md`.
4. **README roadmap.** The README written earlier lists Person 4 as "Phase 2:
   advanced models". The handbook gives Person 4 integration, business metric,
   report and presentation. I left that row for you to correct.
5. **Business metric.** Inputs are in `reports/phase1_baseline_notes.md`
   Section 7 and in the JSON under `business_metric_inputs_for_person4`: per
   10,000 screened URLs at the test-set mix, 4,250.7 alerts (0.2 false alarms),
   20.0 missed phishing, and 70.8 / 212.5 / 354.2 review hours at an **assumed**
   1 / 3 / 5 minutes per alert. These minutes are illustrative scenarios, not
   facts, and the 42.7% phishing share is a benchmark mix, not bank traffic.
6. **Test set is now used.** It was scored once for this baseline. Do not use
   it to choose Phase 2 models, hyperparameters or thresholds.
7. **Licence — confirmed.** The official UCI page for dataset 967 states that the
   dataset is licensed under CC BY 4.0 (checked 2026-11-16). The manifest, dataset
   card and `data/README.md` now say so.
8. **Processed CSVs and Git.** `.gitignore` excludes `data/raw/*.csv` only.
   `data/processed/*.csv` (about 18 MB, essentially the full dataset) would be
   committed. The handbook says not to commit the raw dataset by default, so
   decide whether to ignore them too and rebuild them with notebook 02.

### Changes made to Person 1 and Person 2 files in the final check

Made so that every handbook item is met and each part runs from a fresh
runtime. No data, split, feature or model logic was changed, and the baseline
results are identical before and after. Persons 1 and 2 should review these,
since they will be asked about their own parts.

| File | Change | Why |
|---|---|---|
| `notebooks/01_data_audit_eda.ipynb` | Removed the Google Drive backup cell and the call to `/content/finish_person1.py`; added a title, a project-root cell, the raw two-column export with SHA-256 check, inert URL samples and an observations cell. | The notebook depended on a script that is not in the project, so it could not run from a fresh runtime. Handbook steps 11–12 (inert samples, observations) were missing. |
| `reports/person1/` | Deleted. | Four files byte-identical to the copies in `reports/`. |
| `data/manifest.json`, `docs/dataset_card.md` | Licence wording now "confirmed"; raw CSV path corrected to `data/raw/…`. | The licence check was still open; the path did not match the file. |
| `src/data_cleaning.py` | Added the input contract to the module docstring and a small `convert_original_label()` function. | Handbook Section 8.3, Tasks 1 and 2. |
| `tests/test_data_cleaning.py` | Added 3 label-conversion tests. | Handbook Task 7 lists "original label conversion is correct". |
| `artifacts/split_manifest.json` | Added feature schema version, feature list and dataset-manifest reference. | Handbook Task 5. |
| `reports/leakage_check.md` | Added a table answering the handbook's eight leakage questions. | Two of them (precomputed features, test-set use) had no explicit answer. |
| `docs/feature_dictionary.md` | Added "Why domain age is not a feature". | Handbook Section 3.4 asks for the reason to be stated. |
| `notebooks/02_feature_pipeline.ipynb` | Added a section that saves the data-quality summary and split manifest and asserts the split equals the saved files. | Handbook Task 8, step 9. |
| `requirements.txt` | Added `ucimlrepo`. | Needed by notebook 01. |

---

## 9. Acceptance checklist (handbook Section 9.5)

| Check | Status | Evidence |
|---|---|---|
| The model is Logistic Regression | Yes | `model` and `model_params` in metrics JSON |
| `class_weight="balanced"` used and documented | Yes | metrics JSON, notes Section 1 |
| Full Pipeline fitted on training data only | Yes | `train_baseline()`; notebook Section 4 |
| Test split unchanged | Yes | `split_reproducible_from_cleaned_csv: true`, file hashes in JSON |
| Positive class is phishing = 1 | Yes | `positive_label: 1`, `pos_label=1` in code |
| Precision, recall, F1 and confusion matrix exist | Yes | JSON, PNG, notes |
| Confusion-matrix order documented | Yes | `confusion_matrix_label_order`, axis labels on PNG |
| Metrics JSON matches notebook output | Yes | assertion in notebook Section 9 |
| Threshold 0.50 stated | Yes | JSON, notes, PNG title |
| No test-set tuning | Yes | fixed settings; CV on train only |
| Notebook runs from a fresh runtime | Local fresh kernel: yes. Colab: **not yet run** | Section 6 |
| No unsupported performance claims | Yes | caveat in Section 5 and notes Section 8 |
