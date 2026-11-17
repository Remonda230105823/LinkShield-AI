# Person 1 → Person 2 Handover

## Project
LinkShield AI, Phase 1, Person 1 research/dataset/EDA. Primary customer: retail bank security operations / email-security team.

## Source and retrieval
Official UCI dataset ID 967: https://archive.ics.uci.edu/dataset/967/phiusil+phishing+url+dataset
```python
!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=967)
X, y = dataset.data.features, dataset.data.targets
```

## Input contract
- Original URL column: `URL`; original target: `label`.
- Original mapping: `1=legitimate`, `0=phishing`.
- Project mapping: `is_phishing=1` for phishing, `0` for legitimate.
- Convert with `is_phishing = 1 - original_label` after validating labels are {0,1}.
- Only two columns are allowed in modelling input: `url` (raw URL string) and `is_phishing` (integer label).
- Do not use precomputed UCI features, or a `FILENAME` field, in the model.

## Observed data quality
- Raw rows: 235,795; legitimate: 134,850; phishing: 100,945.
- Missing URL: 0; empty URL: 0.
- Exact duplicate URL strings: 425; conflicting-label URL groups: 0.
- No cleaning/deduplication performed by Person 1.

## Files
- `docs/problem_market_case.md`
- `docs/dataset_card.md`
- `data/manifest.json`
- `notebooks/01_data_audit_eda.ipynb` (must be copied from Colab using File > Save a copy in Drive / download and upload)
- `reports/eda_summary.json`, `reports/eda_class_distribution.png`, `reports/eda_url_length_distribution.png`, `reports/eda_observations.md`
- Local only: `data/raw/person1_raw_labeled_urls.csv` — **DO NOT COMMIT RAW CSV TO GITHUB**.
- SHA256 of local CSV: `fe2271b5e5f09dc9f013f669b76fbb483b1f5806b149ae409c4b09ea04b14190`

## Person 2 next actions
1. Validate schema and original/project mapping; rerun P1 notebook from fresh runtime.
2. Implement documented offline URL validation, whitespace handling and normalized duplicate detection.
3. Quarantine conflicting-label normalized groups; do not lowercase full URL blindly.
4. Extract deterministic lexical features without network requests.
5. Split before fitting scalers/encoders; audit for leakage and hostname overlap.
6. Record row counts after every cleaning stage and create a clean handover to Person 3.

## Safety and open checks
Never open or visit any URL in the dataset. Verify exact authors, citation, DOI and current license from official UCI page before final submission if metadata did not supply them. The original download timestamp and raw filename were not recorded; the export timestamp is not presented as the original download date.

## Acceptance
Person 2 confirms the UCI source, loading, label mapping, licence/citation, runnable EDA notebook, no raw CSV on GitHub, and no URL visitation.
