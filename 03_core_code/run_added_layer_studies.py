from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from boundary import assert_inside_root, prepare_runtime


ROOT = Path(__file__).resolve().parents[1]


def run_script(script: Path) -> dict:
    assert_inside_root(script, ROOT)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def main() -> int:
    prepare_runtime(ROOT)
    traditional = run_script(ROOT / "traditional_sampling_layer" / "src" / "run_traditional_sampling_layer.py")
    polynomial = run_script(ROOT / "polynomial_only_layer" / "src" / "run_polynomial_only_layer.py")
    summary = (
        "# 新增层级验证总报告\n\n"
        "## 层级 1：传统采样方式对比\n\n"
        f"- 是否通过：{'是' if traditional['passed'] else '否'}。\n"
        f"- AFP exchange 相对 AFP 的 local RMSE 平均改善：{100 * traditional['mean_local_improvement_vs_afp']:.2f}%。\n"
        f"- AFP exchange 相对逐实例最佳传统采样的 local RMSE 平均改善：{100 * traditional['mean_local_improvement_vs_best_traditional']:.2f}%。\n"
        f"- 正改善比例：{100 * traditional['positive_ratio_vs_afp']:.1f}%。\n"
        f"- p 值：{traditional['p_vs_afp']:.4g}。\n\n"
        "详细报告：`traditional_sampling_layer/reports/traditional_sampling_layer_summary.md`。\n\n"
        "## 层级 2：仅多项式条件对比\n\n"
        f"- 解释性判定是否通过：{'是' if polynomial['passed_interpretation'] else '否'}。\n"
        f"- 最佳平均阶数：{polynomial['best_degree']}。\n"
        f"- 阶数效应/布局效应均值比：{polynomial['degree_to_layout_effect_ratio']:.2f}。\n"
        f"- AFP exchange 相对 AFP 的 local RMSE 平均改善：{100 * polynomial['mean_local_improvement_exchange_vs_afp']:.2f}%。\n"
        f"- p 值：{polynomial['p_value']:.4g}。\n\n"
        "详细报告：`polynomial_only_layer/reports/polynomial_only_layer_summary.md`。\n\n"
        "## 总体判断\n\n"
        + (
            "新增证据链较清晰：传统采样对比用于证明新方法不是只优于 AFP；多项式对比用于说明低阶/固定阶模型会限制采样优化收益表达，主效果应在 PHS 平台论证。\n"
            if traditional["passed"] and polynomial["passed_interpretation"]
            else "新增证据链尚不完全闭合。应优先查看两个层级报告中的原因检查，重点排查面型高频/局部性、采样点数、候选点池密度和多项式条件数。\n"
        )
    )
    report = ROOT / "reports" / "added_layer_validation_summary.md"
    assert_inside_root(report, ROOT)
    report.write_text(summary, encoding="utf-8")
    print(json.dumps({"traditional": traditional, "polynomial": polynomial}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

