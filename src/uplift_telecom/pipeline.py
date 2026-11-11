"""End-to-end pipeline: data -> tuning -> evaluation -> targeting list.

Run from the repository root::

    python -m uplift_telecom.pipeline              # full run, about 15 minutes on 2 CPU cores
    python -m uplift_telecom.pipeline --sample 60000 --fast   # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import FEATURES, GRID, SEED, SEEDS, K, Paths, repo_root
from .data import (
    average_treatment_effect,
    download_megafon,
    load_megafon,
    split_indices,
    standardized_mean_difference,
    validate,
)
from .evaluation import (
    campaign_efficiency,
    deciles,
    evaluate_on_test,
    negative_uplift_profile,
    permutation_importance_qini,
    qini_curve_table,
    repeated_splits,
    select_final_model,
    tune,
)
from .targeting import targeting_list

log = logging.getLogger("uplift_telecom")


def run(paths: Paths, data_path: Path | None = None, sample: int | None = None, fast: bool = False) -> dict:
    t0 = time.time()
    paths.make_dirs()
    data_path = Path(data_path) if data_path else paths.data_file
    if not data_path.exists():
        log.info("Dataset not found, downloading to %s", data_path)
        download_megafon(data_path)

    # 1. Load and validate
    df = load_megafon(data_path)
    if sample:
        df = df.sample(n=sample, random_state=SEED).reset_index(drop=True)
    quality = validate(df)
    log.info("Loaded %s rows; missing=%s, duplicates=%s", f"{quality['rows']:,}",
             quality["missing_values"], quality["duplicate_rows"])
    X, y, t = df[FEATURES], df["conversion"], df["T"]

    # 2. Treatment effect and balance
    ate = average_treatment_effect(y, t)
    smd = standardized_mean_difference(X, t)
    log.info("ATE = %.4f (95%% CI %.4f to %.4f); max |SMD| = %.4f",
             ate["ate"], ate["ci_low"], ate["ci_high"], smd.abs().max())
    plots.group_conversion(ate["p_control"], ate["p_treatment"], paths.figures / "fig1_group_conversion.png")
    plots.balance_smd(smd, paths.figures / "fig2_balance_smd.png")

    # 3. Train/test split
    idx_tr, idx_te = split_indices(t, y, seed=SEED)
    X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_tr], y.iloc[idx_te]
    t_tr, t_te = t.iloc[idx_tr], t.iloc[idx_te]

    # 4. Tuning by cross-validated Qini (training set only)
    grid = GRID[:2] if fast else GRID
    cv, best = tune(X_tr, y_tr, t_tr, grid=grid)
    cv.to_csv(paths.results / "cv_tuning.csv", index=False)
    log.info("Tuned settings: %s", best)

    # 5. Test-set evaluation
    metrics, scores, fitted = evaluate_on_test(X_tr, y_tr, t_tr, X_te, y_te, t_te, best)
    metrics.to_csv(paths.results / "test_metrics.csv")
    qini_curve_table(y_te, t_te, scores).to_csv(paths.results / "qini_curves.csv", index=False)
    plots.qini_curves(y_te, t_te, scores, metrics, paths.figures / "fig3_qini_curves.png")
    log.info("Test metrics:\n%s", metrics.round(4).to_string())

    # 6. Repeated splits and model selection
    seeds = SEEDS[:2] if fast else SEEDS
    rep, summary = repeated_splits(X, y, t, best, seeds=seeds)
    rep.to_csv(paths.results / "repeated_splits.csv", index=False)
    final = select_final_model(summary)
    plots.repeated_splits(rep, summary, paths.figures / "fig4_repeated_splits.png")
    log.info("Selected final model: %s", final)

    # 7. Deciles and campaign efficiency
    dec = deciles(y_te, scores[final], t_te)
    dec.to_csv(paths.results / "deciles_final_model.csv")
    plots.decile_chart(dec, final, paths.figures / "fig5_deciles.png")
    eff = campaign_efficiency(y_te, t_te, scores, final)
    eff.to_csv(paths.results / "campaign_efficiency.csv")
    plots.campaign_efficiency(eff, final, paths.figures / "fig6_campaign_efficiency.png")

    # 8. Targeting list
    tl = targeting_list(df["customer_id"].iloc[idx_te].values, scores[final], budget=K,
                        model=fitted[final], X=X_te)
    tl.to_csv(paths.results / "targeting_list_top30.csv.gz", index=False)

    # 9. Negative uplift and feature influence
    neg = negative_uplift_profile(X_te, y_te, t_te, scores[final])
    imp, base_q = permutation_importance_qini(fitted[final], X_te, y_te, t_te)
    imp.to_csv(paths.results / "feature_influence.csv", header=["qini_drop"])
    plots.feature_influence(imp, final, paths.figures / "fig7_feature_influence.png")

    # 10. Success criteria from the proposal
    overall = float(y_te[t_te == 1].mean() - y_te[t_te == 0].mean())
    fin = summary.loc[final]
    share_dev = max(abs(t_tr.mean() - t.mean()), abs(t_te.mean() - t.mean()))
    runtime = (time.time() - t0) / 60
    criteria = {
        "1_data_size": bool(len(df) >= 100_000),
        "2_group_identification": bool(share_dev <= 0.01),
        "3_group_balance": bool(smd.abs().max() < 0.1),
        "4_coverage": bool(np.isfinite(scores[final]).all()),
        "5_better_than_random": bool(metrics.loc[final, "Qini coefficient"] > 0),
        "6_better_than_conversion_model": bool(metrics.loc[final, "Qini coefficient"]
                                               > metrics.loc["Conversion model", "Qini coefficient"]),
        "7_stability": bool(fin["positive_splits"] >= len(seeds) - 1),
        "8_targeting_value": bool(metrics.loc[final, "Uplift@30%"] > overall),
        "9_reproducibility": bool(runtime < 60),
    }
    summary_json = {
        "final_model": final, "best_params": best, "data_quality": quality, "ate": ate,
        "max_smd": float(smd.abs().max()), "overall_uplift_test": overall,
        "test_metrics": metrics.drop(columns="Fit+predict time (s)").to_dict(orient="index"),
        "repeated_splits": summary.to_dict(orient="index"),
        "negative_uplift": {k: v for k, v in neg.items() if k != "top_smd"},
        "permutation_base_qini": base_q, "success_criteria": criteria,
        "all_criteria_met": all(criteria.values()), "runtime_min": runtime,
        "sample": sample, "fast": fast,
    }
    with open(paths.results / "summary.json", "w") as fh:
        json.dump(summary_json, fh, indent=2, default=float)
    failed = [k for k, v in criteria.items() if not v]
    if failed:
        note = " (expected with --sample below 100,000 rows)" if failed == ["1_data_size"] and sample else ""
        log.info("Success criteria not met: %s%s. Runtime %.1f min.", ", ".join(failed), note, runtime)
    else:
        log.info("All 9 success criteria met. Runtime %.1f min.", runtime)
    return summary_json


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Uplift modeling pipeline on the MegaFon dataset.")
    parser.add_argument("--root", type=Path, default=repo_root(), help="repository root (default: auto)")
    parser.add_argument("--data", type=Path, default=None, help="path to megafon_dataset.csv.gz")
    parser.add_argument("--sample", type=int, default=None, help="use a random sample of N rows (quick run)")
    parser.add_argument("--fast", action="store_true", help="2 grid settings and 2 seeds instead of 4 and 5")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn deprecation notices from sklift
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    pd.set_option("display.width", 140)
    run(Paths(args.root), data_path=args.data, sample=args.sample, fast=args.fast)


if __name__ == "__main__":
    main()
