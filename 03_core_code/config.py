from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StudyConfig:
    root: Path
    grid_size: int = 56
    sample_count: int = 64
    candidate_count: int = 720
    instances_per_family: int = 5
    random_seed: int = 20260513
    local_quantile: float = 0.80
    min_distance: float = 0.055
    optimization_rounds: int = 14
    optimization_trials_per_round: int = 7
    phase2_train_instances_per_family: int = 5
    phase2_test_instances_per_family: int = 2

    @property
    def src(self) -> Path:
        return self.root / "src"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def configs(self) -> Path:
        return self.root / "configs"


SURFACE_FAMILIES = (
    "smooth_low_variation",
    "edge_rolloff",
    "local_bump",
    "mid_frequency",
)

POLY_DEGREES = (3, 5, 7, 9, 11)

PHASE1_METHODS = (
    "afp_phs",
    "hessian_weighted_phs",
    "afp_exchange_phs",
    "estimated_hessian_exchange_phs",
)
