import json

import numpy as np
import pytest
from sklift.metrics import qini_auc_score

from uplift_telecom.config import ALL_MODELS, FEATURES, GRID, Paths
from uplift_telecom.data import load_megafon, split_indices, standardized_mean_difference, validate
from uplift_telecom.models import fit_score
from uplift_telecom.pipeline import run
from uplift_telecom.targeting import targeting_list


@pytest.fixture(scope="module")
def df(synthetic_csv):
    return load_megafon(synthetic_csv)


def test_load_adds_id_and_flag(df):
    assert df["customer_id"].is_unique
    assert set(df["T"].unique()) == {0, 1}
    q = validate(df)
    assert q["features"] == 50 and q["missing_values"] == 0


def test_split_is_stratified(df):
    tr, te = split_indices(df["T"], df["conversion"], seed=42)
    assert len(set(tr) & set(te)) == 0
    assert len(te) == pytest.approx(0.3 * len(df), abs=1)
    assert abs(df["T"].iloc[tr].mean() - df["T"].iloc[te].mean()) < 0.01
    assert abs(df["conversion"].iloc[tr].mean() - df["conversion"].iloc[te].mean()) < 0.01


def test_random_assignment_is_balanced(df):
    assert standardized_mean_difference(df[FEATURES], df["T"]).abs().max() < 0.1


@pytest.mark.parametrize("name", ALL_MODELS)
def test_every_model_returns_finite_scores(df, name):
    tr, te = split_indices(df["T"], df["conversion"])
    X, y, t = df[FEATURES], df["conversion"], df["T"]
    params = {"lr": 0.1, "leaves": 15, "iters": 50}
    scores, _ = fit_score(name, params, X.iloc[tr], y.iloc[tr], t.iloc[tr], X.iloc[te])
    assert scores.shape == (len(te),)
    assert np.isfinite(scores).all()


def test_t_learner_finds_the_effect(df):
    tr, te = split_indices(df["T"], df["conversion"])
    X, y, t = df[FEATURES], df["conversion"], df["T"]
    u, _ = fit_score("T-learner", GRID[0], X.iloc[tr], y.iloc[tr], t.iloc[tr], X.iloc[te])
    rnd, _ = fit_score("Random", None, X.iloc[tr], y.iloc[tr], t.iloc[tr], X.iloc[te])
    assert qini_auc_score(y.iloc[te], u, t.iloc[te]) > qini_auc_score(y.iloc[te], rnd, t.iloc[te]) + 0.05


def test_targeting_list_flags_budget():
    tl = targeting_list(np.arange(100), np.linspace(-1, 1, 100), budget=0.3)
    assert (tl["decision"] == "target").sum() == 30
    assert tl["predicted_uplift"].is_monotonic_decreasing
    assert tl.loc[0, "rank"] == 1


def test_pipeline_end_to_end(tmp_path, synthetic_csv):
    out = run(Paths(tmp_path), data_path=synthetic_csv, fast=True)
    assert out["final_model"] in {"S-learner", "T-learner", "Class transformation"}
    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert summary["success_criteria"]["4_coverage"]
    assert len(list((tmp_path / "figures").glob("fig*.png"))) == 7
    assert (tmp_path / "results" / "targeting_list_top30.csv.gz").exists()
