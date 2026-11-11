"""Tuning and evaluation with uplift metrics (Qini, AUUC, uplift@k)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklift.metrics import (
    qini_auc_score,
    qini_curve,
    uplift_at_k,
    uplift_auc_score,
    uplift_by_percentile,
)

from .config import ALL_MODELS, BASELINES, CV_FOLDS, GRID, SEED, SEEDS, UPLIFT_MODELS, K
from .data import split_indices
from .models import fit_score


def tune(X, y, t, models=UPLIFT_MODELS, grid=GRID, folds=CV_FOLDS):
    """Choose base-learner settings by mean cross-validated Qini on the training set.

    Returns the full CV table and a ``{model: params}`` dict with the best settings.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    strata = np.asarray(t) * 2 + np.asarray(y)
    rows = []
    for name in models:
        for gi, params in enumerate(grid):
            scores = []
            for f_tr, f_va in skf.split(X, strata):
                u, _ = fit_score(name, params, X.iloc[f_tr], y.iloc[f_tr], t.iloc[f_tr], X.iloc[f_va])
                scores.append(qini_auc_score(y.iloc[f_va], u, t.iloc[f_va]))
            rows.append({"model": name, "config": gi, **params,
                         "cv_qini_mean": np.mean(scores), "cv_qini_std": np.std(scores)})
    cv = pd.DataFrame(rows)
    best = {m: grid[int(cv[cv.model == m].sort_values("cv_qini_mean").iloc[-1]["config"])] for m in models}
    best["Conversion model"] = best.get("S-learner", grid[0])
    return cv, best


def uplift_metrics(y, scores, t, k: float = K) -> dict:
    return {
        "Qini coefficient": qini_auc_score(y, scores, t),
        "AUUC": uplift_auc_score(y, scores, t),
        f"Uplift@{int(k * 100)}%": uplift_at_k(y, scores, t, strategy="overall", k=k),
    }


def evaluate_on_test(X_tr, y_tr, t_tr, X_te, y_te, t_te, best: dict, models=ALL_MODELS):
    """Fit every model on the training set and score the held-out test set."""
    import time

    scores, fitted, rows = {}, {}, {}
    for name in models:
        start = time.time()
        scores[name], fitted[name] = fit_score(name, best.get(name, GRID[0]), X_tr, y_tr, t_tr, X_te)
        rows[name] = {**uplift_metrics(y_te, scores[name], t_te), "Fit+predict time (s)": time.time() - start}
    return pd.DataFrame(rows).T.loc[list(models)], scores, fitted


def repeated_splits(X, y, t, best: dict, seeds=SEEDS, models=ALL_MODELS, k: float = K):
    """Retrain every model on several stratified splits and summarize the Qini coefficient."""
    rows = []
    for sd in seeds:
        a, b = split_indices(t, y, seed=sd)
        for name in models:
            u, _ = fit_score(name, best.get(name, GRID[0]), X.iloc[a], y.iloc[a], t.iloc[a], X.iloc[b])
            rows.append({"seed": sd, "model": name,
                         "qini": qini_auc_score(y.iloc[b], u, t.iloc[b]),
                         "uplift_at_30": uplift_at_k(y.iloc[b], u, t.iloc[b], strategy="overall", k=k)})
    rep = pd.DataFrame(rows)
    summary = rep.groupby("model").agg(
        qini_mean=("qini", "mean"), qini_std=("qini", "std"), qini_min=("qini", "min"),
        positive_splits=("qini", lambda s: int((s > 0).sum())), uplift30_mean=("uplift_at_30", "mean"),
    ).loc[list(models)]
    return rep, summary


def select_final_model(summary: pd.DataFrame) -> str:
    """Uplift approach with the highest mean Qini over the repeated splits."""
    return summary.drop(BASELINES, errors="ignore")["qini_mean"].idxmax()


def deciles(y, scores, t, bins: int = 10) -> pd.DataFrame:
    dec = uplift_by_percentile(y, scores, t, strategy="overall", bins=bins, total=False)
    dec.index = [f"D{i}" for i in range(1, bins + 1)]
    return dec


def campaign_efficiency(y, t, scores: dict, final: str, shares=(0.1, 0.2, 0.3, 0.4, 0.5)) -> pd.DataFrame:
    """Estimated incremental conversions = customers contacted x observed uplift in that group."""
    y, t = np.asarray(y), np.asarray(t)
    overall = y[t == 1].mean() - y[t == 0].mean()
    rows = []
    for k in shares:
        n_sel = int(round(k * len(y)))
        row = {"contact_share": f"{int(k * 100)}%", "customers_contacted": n_sel, "random_targeting": n_sel * overall}
        for name in (final, "Conversion model"):
            row[name] = n_sel * uplift_at_k(y, scores[name], t, strategy="overall", k=k)
        rows.append(row)
    eff = pd.DataFrame(rows).set_index("contact_share")
    eff[f"gain_vs_random ({final})"] = eff[final] / eff["random_targeting"]
    return eff


def qini_curve_table(y, t, scores: dict, step: int = 2) -> pd.DataFrame:
    """Qini curves of all models interpolated on a common grid of contact shares (%)."""
    grid = np.arange(0, 101, step)
    out = {"share_pct": grid}
    last = None
    for name, s in scores.items():
        if name == "Random":
            continue
        xs, ys = qini_curve(y, s, t)
        out[name] = np.interp(grid, xs / len(y) * 100, ys)
        last = ys[-1]
    out["Random"] = grid / 100 * last
    return pd.DataFrame(out)


def negative_uplift_profile(X, y, t, scores, top: int = 8) -> dict:
    """Share of customers with negative predicted uplift and how they differ from the rest."""
    y, t, scores = np.asarray(y), np.asarray(t), np.asarray(scores)
    neg = scores < 0
    obs = {lbl: y[m & (t == 1)].mean() - y[m & (t == 0)].mean() for lbl, m in (("negative", neg), ("rest", ~neg))}
    xn, xr = X[neg], X[~neg]
    smd = (xn.mean() - xr.mean()) / np.sqrt((xn.var() + xr.var()) / 2)
    smd = smd.reindex(smd.abs().sort_values(ascending=False).index).head(top)
    return {"share": float(neg.mean()), "n_negative": int(neg.sum()),
            "observed_uplift_negative": float(obs["negative"]), "observed_uplift_rest": float(obs["rest"]),
            "top_smd": smd}


def permutation_importance_qini(model, X, y, t, n_sample: int = 60_000, repeats: int = 2) -> tuple[pd.Series, float]:
    """Drop in Qini when each feature is shuffled (on a random sample of ``X``)."""
    rs = np.random.RandomState(SEED)
    sample = rs.choice(len(X), min(n_sample, len(X)), replace=False)
    Xs = X.iloc[sample].reset_index(drop=True)
    ys, ts = np.asarray(y)[sample], np.asarray(t)[sample]
    base = qini_auc_score(ys, model.predict(Xs), ts)
    drops = {}
    for f in X.columns:
        d = []
        for r in range(repeats):
            Xp = Xs.copy()
            Xp[f] = np.random.RandomState(SEED + r).permutation(Xp[f].values)
            d.append(base - qini_auc_score(ys, model.predict(Xp), ts))
        drops[f] = np.mean(d)
    return pd.Series(drops).sort_values(ascending=False), float(base)
