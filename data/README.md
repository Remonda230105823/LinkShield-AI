# Data — LinkShield AI

All URLs in this folder are **inert text**. Nothing here is ever opened,
resolved or requested over the network.

## Source

| Item | Value |
|---|---|
| Dataset | PhiUSIIL Phishing URL (Website) |
| Repository | UCI Machine Learning Repository, dataset ID 967 |
| Page | https://archive.ics.uci.edu/dataset/967/phiusil+phishing+url+dataset |
| Authors | Arvind Prasad and Shalini Chandra (2024) |
| DOI | 10.1016/j.cose.2023.103545 |
| Licence | CC BY 4.0 — confirmed on the official UCI dataset page on 2026-11-16 |

Full provenance, counts and the SHA-256 of the raw export are in
`data/manifest.json`; the description is in `docs/dataset_card.md`.

## Files

| File | Rows | Made by | Contents |
|---|---|---|---|
| `raw/person1_raw_labeled_urls.csv` | 235,795 | Person 1 | `url`, `is_phishing` exported from UCI; not cleaned. Excluded from Git by `.gitignore`. |
| `processed/cleaned_urls.csv` | 235,356 | Person 2 | After validation and normalized de-duplication (439 duplicates removed). |
| `processed/train.csv` | 188,284 | Person 2 | Stratified 80% split, seed 42. |
| `processed/test.csv` | 47,072 | Person 2 | Stratified 20% split, seed 42. Held out; used once for the Phase 1 baseline. |

Label convention in every file: `is_phishing = 1` is phishing, `is_phishing = 0`
is legitimate. The original UCI `label` column is the opposite (1 = legitimate)
and has already been inverted. Do not invert it again.

Only the `URL` and `label` columns of the UCI dataset are used. Its precomputed
feature columns are deliberately not used.

## How to obtain the raw data again

`notebooks/01_data_audit_eda.ipynb` downloads the dataset with the official
client and exports the two columns:

```python
from ucimlrepo import fetch_ucirepo   # pip install ucimlrepo
import pandas as pd

dataset = fetch_ucirepo(id=967)
df = pd.DataFrame({
    "url": dataset.data.features["URL"],
    "is_phishing": 1 - dataset.data.targets["label"],
})
df.to_csv("data/raw/person1_raw_labeled_urls.csv", index=False)
```

Check the result against `exported_csv_sha256` in `manifest.json`.

## How to rebuild the processed files

Run `notebooks/02_feature_pipeline.ipynb`, or from the project root:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_cleaning import run_cleaning_pipeline

raw = pd.read_csv("data/raw/person1_raw_labeled_urls.csv")
clean, quarantined, report = run_cleaning_pipeline(raw)
train, test = train_test_split(
    clean, test_size=0.20, random_state=42, stratify=clean["is_phishing"]
)
```

`src/train_baseline.py` re-checks on every run that `train.csv` and `test.csv`
are exactly what this procedure produces from `cleaned_urls.csv`.
