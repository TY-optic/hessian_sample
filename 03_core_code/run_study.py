from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from boundary import assert_inside_root, checked_open, prepare_runtime, resolve_root
from config import StudyConfig


def write_config(config: StudyConfig) -> None:
    data = {
        "root": str(config.root),
        "grid_size": config.grid_size,
        "sample_count": config.sample_count,
        "candidate_count": config.candidate_count,
        "instances_per_family": config.instances_per_family,
        "random_seed": config.random_seed,
        "local_quantile": config.local_quantile,
        "min_distance": config.min_distance,
        "optimization_rounds": config.optimization_rounds,
        "optimization_trials_per_round": config.optimization_trials_per_round,
    }
    with checked_open(config.configs / "study_config.json", config.root, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def method_recommendation(config: StudyConfig, phase1_decision: dict, learning_result: dict) -> None:
    if phase1_decision["passed_phase1"]:
        keep = [
            "AFP + cubic PHS 作为主基线",
            "AFP initial + point exchange optimization + cubic PHS 作为核心方法",
            "Hessian weighted sampling + cubic PHS 作为无迭代对照",
            "regular + XY polynomial 与 AFP + XY polynomial 仅用于说明低阶模型限制",
        ]
        remove = ["Gaussian RBF 不建议作为主评价平台，仅保留稳定性说明"]
    else:
        keep = [
            "AFP + cubic PHS 作为基线",
            "AFP initial + point exchange optimization + cubic PHS 作为待改进方法",
            "Hessian-误差关系分析用于解释失败或局部有效性",
        ]
        remove = [
            "不建议把当前采样优化写成主结论",
            "第二阶段学习结果不应进入主文核心结论",
        ]
    text = (
        "# 方法保留建议\n\n"
        "## 建议保留\n\n"
        + "\n".join(f"- {item}" for item in keep)
        + "\n\n## 建议删除或降级\n\n"
        + "\n".join(f"- {item}" for item in remove)
        + "\n\n## 第二阶段状态\n\n"
        + (f"- 已运行：{learning_result}\n" if learning_result.get("ran") else "- 未运行或仅生成说明：第一阶段未满足进入条件。\n")
    )
    with checked_open(config.reports / "method_recommendation.md", config.root, "w", encoding="utf-8") as f:
        f.write(text)


def final_manifest(config: StudyConfig) -> None:
    files = []
    for path in sorted(config.root.rglob("*")):
        if path.is_file():
            assert_inside_root(path, config.root)
            files.append(path.relative_to(config.root).as_posix())
    text = (
        "# 最终目录清单\n\n"
        "## 关键文件\n\n"
        + "\n".join(f"- `{p}`" for p in files)
        + "\n\n## 边界说明\n\n"
        "- 本次代码、输出、日志、模型与报告均位于 `sampling_optimization_study/` 内。\n"
        "- 脚本包含路径边界检查；写入路径必须能解析到该目录内部。\n"
        "- 未修改或写入该新文件夹之外的任何项目文件。\n"
    )
    with checked_open(config.reports / "final_directory_manifest.md", config.root, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    root = resolve_root()
    os.chdir(root)
    prepare_runtime(root)
    assert_inside_root(Path.cwd(), root)
    config = StudyConfig(root=root)
    write_config(config)

    from phase1 import run_phase1
    from learning import run_learning_phase

    _, phase1_decision = run_phase1(config)
    learning_result = run_learning_phase(config, phase1_decision["passed_phase1"])
    method_recommendation(config, phase1_decision, learning_result)
    final_manifest(config)
    print(json.dumps({"phase1": phase1_decision, "learning": learning_result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

