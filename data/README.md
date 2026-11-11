# Data

The dataset is **not stored in this repository**. Download it with:

```bash
python scripts/download_data.py      # or: make data
```

This saves `data/megafon_dataset.csv.gz` (about 270 MB compressed, 570 MB as CSV). The script first tries
`sklift.datasets.fetch_megafon()` and falls back to a public mirror.

| | |
|---|---|
| Source | MegaFon Uplift Competition, Open Data Science (ODS.ai), 2021 |
| Type | Synthetic data that imitates a telecom marketing campaign |
| Rows | 600,000 labelled customers |
| Columns | `treatment_group` (`treatment` / `control`), `X_1` … `X_50` (anonymized numeric features), `conversion` (0/1) |
| Groups | 300,368 treatment, 299,632 control |
| Missing values | none |

The file has no subscriber id column, so the code uses the row number as `customer_id`.
