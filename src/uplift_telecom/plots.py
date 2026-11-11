"""Figures saved to ``figures/`` (same style and file names as the notebook)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklift.metrics import qini_curve  # noqa: E402

from .config import COLORS  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "axes.titleweight": "bold",
})


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def group_conversion(p0: float, p1: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars = ax.bar(["Control (T=0)", "Treatment (T=1)"], [p0, p1], color=["#8a8984", "#2a78d6"], width=0.55)
    for bar, v in zip(bars, [p0, p1]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.1%}", ha="center")
    ax.set_ylabel("Conversion rate")
    ax.set_ylim(0, 0.28)
    ax.set_title("Conversion rate by group")
    return _save(fig, path)


def balance_smd(smd: pd.Series, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(range(1, len(smd) + 1), smd.abs().values, color="#2a78d6", width=0.7)
    ax.axhline(0.1, color="#d03b3b", ls="--", lw=1.2, label="threshold 0.1")
    ax.set_xlabel("Feature index (X_1 … X_50)")
    ax.set_ylabel("|SMD|")
    ax.set_title("Treatment vs control balance for all 50 features")
    ax.set_ylim(0, 0.11)
    ax.legend()
    return _save(fig, path)


def qini_curves(y, t, scores: dict, metrics: pd.DataFrame, path: Path, k: float = 0.30) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ys = None
    for name, s in scores.items():
        if name == "Random":
            continue
        xs, ys = qini_curve(y, s, t)
        ax.plot(xs / len(y) * 100, ys, lw=2, color=COLORS[name],
                label=f"{name} (Qini {metrics.loc[name, 'Qini coefficient']:.3f})")
    ax.plot([0, 100], [0, ys[-1]], ls="--", color=COLORS["Random"], lw=1.5, label="Random targeting (Qini 0)")
    ax.axvline(k * 100, color="#52514e", lw=0.8, ls=":")
    ax.text(k * 100 + 1, ax.get_ylim()[1] * 0.05, f"{int(k * 100)}% contact share", color="#52514e", fontsize=8)
    ax.set_xlabel("Share of test customers contacted, ranked by score (%)")
    ax.set_ylabel("Incremental conversions (Qini)")
    ax.set_title("Qini curves on the held-out test set")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, path)


def repeated_splits(rep: pd.DataFrame, summary: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    order = summary.sort_values("qini_mean").index
    for i, name in enumerate(order):
        vals = rep[rep.model == name]["qini"].values
        ax.scatter(vals, [i] * len(vals), color=COLORS[name], s=36, alpha=0.8, zorder=3)
        ax.plot([vals.mean()] * 2, [i - 0.3, i + 0.3], color="#0b0b0b", lw=2, zorder=4)
    ax.axvline(0, color="#8a8984", ls="--", lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Qini coefficient on the test split (dots = 5 seeds, bar = mean)")
    ax.set_title("Stability of the Qini coefficient across repeated splits")
    return _save(fig, path)


def decile_chart(dec: pd.DataFrame, final: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    xi = np.arange(len(dec))
    ax.bar(xi - 0.2, dec["response_rate_treatment"], 0.38, color="#2a78d6", label="Treatment conversion")
    ax.bar(xi + 0.2, dec["response_rate_control"], 0.38, color="#8a8984", label="Control conversion")
    for i, v in enumerate(dec["uplift"].values):
        top = max(dec["response_rate_treatment"].iloc[i], dec["response_rate_control"].iloc[i])
        ax.text(i, top + 0.01, f"{v:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(xi)
    ax.set_xticklabels(dec.index)
    ax.set_xlabel(f"Decile of predicted uplift ({final}); D1 = highest")
    ax.set_ylabel("Observed conversion rate")
    ax.legend(fontsize=8)
    ax.set_title("Observed conversion by uplift decile (numbers = observed uplift)")
    return _save(fig, path)


def campaign_efficiency(eff: pd.DataFrame, final: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    xi, w = np.arange(len(eff)), 0.27
    ax.bar(xi - w, eff["random_targeting"], w, color=COLORS["Random"], label="Random targeting")
    ax.bar(xi, eff["Conversion model"], w, color=COLORS["Conversion model"], label="Conversion model")
    ax.bar(xi + w, eff[final], w, color=COLORS[final], label=final)
    ax.set_xticks(xi)
    ax.set_xticklabels(eff.index)
    ax.set_xlabel("Share of the test customers contacted")
    ax.set_ylabel("Estimated incremental conversions")
    ax.legend(fontsize=8)
    ax.set_title("Incremental conversions by contact share")
    return _save(fig, path)


def feature_influence(imp: pd.Series, final: str, path: Path, top: int = 15) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    sel = imp.head(top)[::-1]
    ax.barh(sel.index, sel.values, color="#eb6834", height=0.65)
    ax.set_xlabel("Drop in Qini coefficient when the feature is shuffled")
    ax.set_title(f"Top {top} features driving the {final} ranking")
    return _save(fig, path)
