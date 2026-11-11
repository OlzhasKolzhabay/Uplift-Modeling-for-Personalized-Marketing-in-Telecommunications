"""Uplift models (S-learner, T-learner, class transformation) and baselines."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklift.models import ClassTransformation, SoloModel, TwoModels

from .config import GRID, SEED


def hgb(lr: float = 0.1, leaves: int = 31, iters: int = 200) -> HistGradientBoostingClassifier:
    """Gradient-boosted trees used as the base learner of every tree model."""
    return HistGradientBoostingClassifier(
        learning_rate=lr,
        max_leaf_nodes=leaves,
        max_iter=iters,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=SEED,
    )


def _logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def make_model(name: str, params: dict | None = None):
    """Build an (unfitted) scikit-uplift model by name."""
    params = params or GRID[0]
    if name == "S-learner":
        return SoloModel(estimator=hgb(**params), method="dummy")
    if name == "T-learner":
        return TwoModels(estimator_trmnt=hgb(**params), estimator_ctrl=hgb(**params), method="vanilla")
    if name == "Class transformation":
        return ClassTransformation(estimator=hgb(**params))
    if name == "T-learner (LogReg)":
        return TwoModels(estimator_trmnt=_logreg(), estimator_ctrl=_logreg(), method="vanilla")
    raise ValueError(f"Unknown uplift model: {name!r}")


def fit_score(name: str, params: dict | None, X_train, y_train, t_train, X_score):
    """Fit a model on the training data and return ``(scores, fitted_model)``.

    ``scores`` rank customers in ``X_score``: predicted uplift for the uplift models,
    ``P(Y=1 | x)`` for the conversion model and uniform noise for random targeting.
    """
    params = params or GRID[0]
    if name == "Random":
        return np.random.RandomState(SEED).rand(len(X_score)), None
    if name == "Conversion model":
        model = hgb(**params).fit(X_train, y_train)
        return model.predict_proba(X_score)[:, 1], model
    model = make_model(name, params).fit(X_train, y_train, t_train)
    return model.predict(X_score), model
