from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_module(path: Path) -> None:
    assert (path / "module_summary.md").exists(), path
    assert (path / "machine_readable_summary.json").exists(), path
    assert list(path.glob("*.csv")), path
    assert list(path.glob("*.png")), path


def test_output_completeness() -> None:
    modules = ("00_mainline", "A_budget", "B_error_transfer", "C_point_scan", "D_trend_residual", "E_family_stability", "F_unified_stats")
    for mode in ("oracle", "practical"):
        for module in modules:
            check_module(ROOT / "outputs" / mode / module)
    strong = ROOT / "outputs" / "strong_baselines"
    assert (strong / "strong_baseline_summary.md").exists()
    assert (strong / "machine_readable_summary.json").exists()
    assert (strong / "strong_baseline_metrics.csv").exists()
    assert (strong / "strong_baseline_local_rmse_boxplot.png").exists()


if __name__ == "__main__":
    test_output_completeness()
    print("test_output_completeness passed")

