"""Loading, validating and splitting the MegaFon uplift dataset."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_URLS, FEATURES, SEED, TEST_SIZE


def download_megafon(dest: Path, force: bool = False) -> Path:
    """Download ``megafon_dataset.csv.gz`` (~270 MB) to ``dest`` if it is missing.

    Tries the scikit-uplift loader first, then the public mirror listed in
    :data:`uplift_telecom.config.DATA_URLS`.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from sklift.datasets import fetch_megafon

        bunch = fetch_megafon()
        df = pd.concat([bunch.data, bunch.treatment, bunch.target], axis=1)
        df.to_csv(dest, index=False)
        return dest
    except Exception as err:  # network or API differences: fall back to the mirror
        last_err: Exception = err
    for url in DATA_URLS:
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as err:
            last_err = err
    raise RuntimeError(f"Could not download the MegaFon dataset: {last_err}")


def load_megafon(path: Path) -> pd.DataFrame:
    """Read the dataset and add ``customer_id`` (row index) and the 0/1 flag ``T``.

    The distributed file has no subscriber id column, so the row number is used.
    """
    df = pd.read_csv(path)
    if "id" in df.columns:
        df = df.drop(columns="id")
    df.insert(0, "customer_id", np.arange(len(df)))
    expected = {"treatment", "control"}
    found = set(df["treatment_group"].unique())
    if found != expected:
        raise ValueError(f"treatment_group must contain {expected}, found {found}")
    df["T"] = (df["treatment_group"] == "treatment").astype(int)
    df["conversion"] = df["conversion"].astype(int)
    return df


def validate(df: pd.DataFrame) -> dict:
    """Basic data-quality report used in the notebook (section 2)."""
    return {
        "rows": int(len(df)),
        "features": int(sum(c in df.columns for c in FEATURES)),
        "missing_values": int(df[FEATURES + ["conversion"]].isna().sum().sum()),
        "duplicate_rows": int(df.drop(columns="customer_id").duplicated().sum()),
        "treatment_share": float(df["T"].mean()),
        "conversion_rate": float(df["conversion"].mean()),
    }


def average_treatment_effect(y: pd.Series, t: pd.Series) -> dict:
    """Difference in conversion rates with a normal-approximation 95% CI."""
    y, t = np.asarray(y), np.asarray(t)
    p1, p0 = y[t == 1].mean(), y[t == 0].mean()
    n1, n0 = (t == 1).sum(), (t == 0).sum()
    ate = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return {"p_treatment": float(p1), "p_control": float(p0), "ate": float(ate),
            "ci_low": float(ate - 1.96 * se), "ci_high": float(ate + 1.96 * se),
            "relative_lift": float(ate / p0)}


def standardized_mean_difference(X: pd.DataFrame, t: pd.Series) -> pd.Series:
    """SMD = (mean_T - mean_C) / sqrt((var_T + var_C) / 2) for every column of ``X``."""
    t = np.asarray(t)
    xt, xc = X[t == 1], X[t == 0]
    return ((xt.mean() - xc.mean()) / np.sqrt((xt.var() + xc.var()) / 2)).rename("SMD")


def split_indices(t: pd.Series, y: pd.Series, seed: int = SEED, test_size: float = TEST_SIZE):
    """Stratified train/test split on the four (treatment, conversion) combinations."""
    strata = np.asarray(t) * 2 + np.asarray(y)
    return train_test_split(np.arange(len(strata)), test_size=test_size,
                            stratify=strata, random_state=seed)
