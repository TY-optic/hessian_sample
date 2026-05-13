from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_key() -> tuple[float, float]:
    data = json.loads((ROOT / "summary" / "machine_readable_final_summary.json").read_text(encoding="utf-8"))
    return (
        data["oracle"]["F_unified_stats"]["mean_local_improvement"],
        data["practical"]["F_unified_stats"]["mean_local_improvement"],
    )


def test_reproducibility() -> None:
    before = read_key()
    subprocess.run([sys.executable, str(ROOT / "src" / "run_all.py")], cwd=str(ROOT), check=True, capture_output=True, text=True)
    after = read_key()
    assert abs(before[0] - after[0]) < 1e-12
    assert abs(before[1] - after[1]) < 1e-12


if __name__ == "__main__":
    test_reproducibility()
    print("test_reproducibility passed")

