"""Turn uplift scores into a ranked campaign targeting list."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import K


def targeting_list(customer_ids, uplift, budget: float = K, model=None, X=None) -> pd.DataFrame:
    """Rank customers by predicted uplift and flag the top ``budget`` share as ``target``.

    If a fitted S- or T-learner and its feature matrix are given, the two
    potential-outcome probabilities (with and without the offer) are added.
    """
    out = pd.DataFrame({"customer_id": np.asarray(customer_ids), "predicted_uplift": np.asarray(uplift)})
    if model is not None and X is not None and hasattr(model, "trmnt_preds_"):
        model.predict(X)  # refreshes the stored per-arm probabilities for X
        out["p_with_offer"] = model.trmnt_preds_
        out["p_without_offer"] = model.ctrl_preds_
    out = out.sort_values("predicted_uplift", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    n_target = int(round(budget * len(out)))
    out["decision"] = np.where(out["rank"] <= n_target, "target", "do not target")
    return out
