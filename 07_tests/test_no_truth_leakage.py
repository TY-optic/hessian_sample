from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_no_truth_leakage() -> None:
    data = json.loads((ROOT / "outputs" / "audits" / "leakage_audit.json").read_text(encoding="utf-8"))
    assert data["passed"], data
    final = json.loads((ROOT / "summary" / "machine_readable_final_summary.json").read_text(encoding="utf-8"))
    assert final["leakage_audit"]["passed"]


if __name__ == "__main__":
    test_no_truth_leakage()
    print("test_no_truth_leakage passed")

