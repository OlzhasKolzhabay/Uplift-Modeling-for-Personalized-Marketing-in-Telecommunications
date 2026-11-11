"""Project-wide constants and output paths.

Every value here matches the settings used in the course notebook
(``notebooks/uplift_modeling_megafon.ipynb``), so the package and the notebook
produce identical numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEED: int = 42
"""Seed for the main train/test split, the models and the random baseline."""

SEEDS: tuple[int, ...] = (42, 7, 123, 2024, 99)
"""Seeds of the five repeated train/test splits used for the stability check."""

TEST_SIZE: float = 0.30
K: float = 0.30
"""Campaign budget: share of customers that receive the offer."""

CV_FOLDS: int = 3
FEATURES: list[str] = [f"X_{i}" for i in range(1, 51)]

# Hyper-parameter grid for the gradient-boosted base learner
# (learning rate, maximum leaves per tree, number of boosting iterations).
GRID: list[dict] = [
    {"lr": 0.10, "leaves": 31, "iters": 200},
    {"lr": 0.05, "leaves": 31, "iters": 300},
    {"lr": 0.10, "leaves": 15, "iters": 200},
    {"lr": 0.10, "leaves": 63, "iters": 150},
]

UPLIFT_MODELS: list[str] = ["S-learner", "T-learner", "Class transformation"]
"""Approaches whose base learner is tuned by cross-validated Qini."""

ALL_MODELS: list[str] = [
    "S-learner",
    "T-learner",
    "Class transformation",
    "T-learner (LogReg)",
    "Conversion model",
    "Random",
]
BASELINES: list[str] = ["Conversion model", "Random"]

DATA_URLS: list[str] = [
    # Mirror of the competition file used by the libuplift project
    "https://github.com/jszymon/uplift_sklearn_data/releases/download/Megafon/megafon_dataset.csv.gz",
]
DATA_FILENAME = "megafon_dataset.csv.gz"

COLORS: dict[str, str] = {
    "S-learner": "#2a78d6",
    "T-learner": "#eb6834",
    "Class transformation": "#1baf7a",
    "T-learner (LogReg)": "#eda100",
    "Conversion model": "#e87ba4",
    "Random": "#8a8984",
}


@dataclass(frozen=True)
class Paths:
    """Folder layout of the repository."""

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def data_file(self) -> Path:
        return self.data / DATA_FILENAME

    def make_dirs(self) -> None:
        for p in (self.data, self.figures, self.results):
            p.mkdir(parents=True, exist_ok=True)


def repo_root() -> Path:
    """Repository root (two levels above this file's package folder)."""
    return Path(__file__).resolve().parents[2]
