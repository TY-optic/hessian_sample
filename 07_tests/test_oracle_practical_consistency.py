from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_practical_consistency() -> None:
    data = json.loads((ROOT / "summary" / "machine_readable_final_summary.json").read_text(encoding="utf-8"))
    oracle = data["oracle"]["F_unified_stats"]["mean_local_improvement"]
    practical = data["practical"]["F_unified_stats"]["mean_local_improvement"]
    assert practical <= oracle + 0.15, {"oracle": oracle, "practical": practical}


if __name__ == "__main__":
    test_oracle_practical_consistency()
    print("test_oracle_practical_consistency passed")

