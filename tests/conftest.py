"""Synthetic uplift data with a known heterogeneous treatment effect (no download needed)."""

import numpy as np
import pandas as pd
import pytest

from uplift_telecom.config import FEATURES


def make_synthetic(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    t = rng.integers(0, 2, size=n)
    # baseline depends on X_1; the offer helps when X_2 > 0 and hurts otherwise
    logit = -1.0 + 0.8 * X[:, 0] + t * np.where(X[:, 1] > 0, 1.5, -1.0)
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame(X, columns=FEATURES)
    df.insert(0, "treatment_group", np.where(t == 1, "treatment", "control"))
    df["conversion"] = y.astype(int)
    return df


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "megafon_dataset.csv.gz"
    make_synthetic().to_csv(path, index=False)
    return path
