from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

STUDY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from boundary import assert_inside_root, prepare_runtime
from config import POLY_DEGREES, StudyConfig, SURFACE_FAMILIES
from phase1 import instance_eval_context
from plotting import save_boxplot, save_line, save_scatter
from reconstruction import metrics, reconstruct_poly, sample_values
from sampling import afp_sampling, candidate_pool, exchange_optimize, regular_sampling, weights_from_hessian
from surfaces import generate_instance


LAYER_ROOT = STUDY_ROOT / "polynomial_only_layer"


def checked_path(path: Path) -> Path:
    return assert_inside_root(path, LAYER_ROOT)


def write_csv(path: Path, rows: list[dict]) -> None:
    path = checked_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path = checked_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def random_uniform(pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return pool[rng.choice(len(pool), size=n, replace=False)]


def jittered_grid(instance: dict, pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = regular_sampling(instance, n)
    desired = base + rng.normal(0.0, 0.035, size=base.shape)
    tree = cKDTree(pool)
    _, idx = tree.query(desired, k=1)
    idx = list(dict.fromkeys(idx.tolist()))
    if len(idx) < n:
        remain = [i for i in range(len(pool)) if i not in set(idx)]
        rng.shuffle(remain)
        idx.extend(remain[: n - len(idx)])
    return pool[idx[:n]]


def make_instances(config: StudyConfig) -> list[dict]:
    instances = []
    for f_id, family in enumerate(SURFACE_FAMILIES):
        for i in range(config.instances_per_family):
            seed = config.random_seed + 70000 + 1000 * f_id + i
            instances.append(generate_instance(config, family, i, seed))
    return instances


def run() -> dict:
    prepare_runtime(STUDY_ROOT)
    os.environ["MPLCONFIGDIR"] = str(checked_path(LAYER_ROOT / "outputs" / "matplotlib_cache"))
    config = StudyConfig(root=STUDY_ROOT)
    for sub in ("outputs", "reports", "src"):
        checked_path(LAYER_ROOT / sub).mkdir(parents=True, exist_ok=True)

    instances = make_instances(config)
    rows = []
    for inst in instances:
        ctx = instance_eval_context(config, inst)
        pool = candidate_pool(inst, config.candidate_count, inst["seed"] + 1)
        weights = weights_from_hessian(inst, ctx["hessian"], pool)
        afp = afp_sampling(pool, config.sample_count, inst["seed"] + 3, min_distance=config.min_distance)
        opt, _ = exchange_optimize(
            inst,
            afp,
            pool,
            weights,
            ctx["local_flat"],
            inst["seed"] + 4,
            max(8, config.optimization_rounds // 2),
            config.optimization_trials_per_round,
            config.min_distance,
        )
        layouts = {
            "regular_grid": regular_sampling(inst, config.sample_count),
            "random_uniform": random_uniform(pool, config.sample_count, inst["seed"] + 2),
            "jittered_grid": jittered_grid(inst, pool, config.sample_count, inst["seed"] + 5),
            "afp_baseline": afp,
            "hessian_weighted": afp_sampling(pool, config.sample_count, inst["seed"] + 6, weights, config.min_distance),
            "afp_exchange": opt,
        }
        for method, points in layouts.items():
            z = sample_values(inst, points)
            for degree in POLY_DEGREES:
                pred, cond = reconstruct_poly(points, z, ctx["eval_xy"], degree)
                met = metrics(pred, ctx["truth"], ctx["local_flat"])
                rows.append(
                    {
                        "family": inst["family"],
                        "index": inst["index"],
                        "seed": inst["seed"],
                        "method": method,
                        "degree": degree,
                        "global_rmse": met["global_rmse"],
                        "local_rmse": met["local_rmse"],
                        "p95_abs": met["p95_abs"],
                        "condition": cond,
                    }
                )
    write_csv(LAYER_ROOT / "outputs" / "polynomial_only_metrics.csv", rows)

    methods = sorted(set(r["method"] for r in rows))
    degree_series = {}
    for method in methods:
        degree_series[method] = [
            float(np.mean([r["local_rmse"] for r in rows if r["method"] == method and r["degree"] == d]))
            for d in POLY_DEGREES
        ]
    save_line(checked_path(LAYER_ROOT / "outputs" / "poly_degree_local_rmse_trends.png"), degree_series, "Polynomial-only local RMSE trends", "local RMSE")
    best_degree = min(
        POLY_DEGREES,
        key=lambda d: np.mean([r["local_rmse"] for r in rows if r["degree"] == d]),
    )
    best_rows = [r for r in rows if r["degree"] == best_degree]
    local_box = {m: [r["local_rmse"] for r in best_rows if r["method"] == m] for m in methods}
    save_boxplot(checked_path(LAYER_ROOT / "outputs" / f"poly_degree_{best_degree}_local_rmse_boxplot.png"), local_box, f"Polynomial degree {best_degree} local RMSE", "local RMSE")

    baseline = np.array([r["local_rmse"] for r in best_rows if r["method"] == "afp_baseline"])
    exchange = np.array([r["local_rmse"] for r in best_rows if r["method"] == "afp_exchange"])
    imp = (baseline - exchange) / baseline
    p_value = float(stats.wilcoxon(baseline, exchange, alternative="greater").pvalue)
    save_scatter(checked_path(LAYER_ROOT / "outputs" / "paired_poly_afp_vs_exchange_local.png"), baseline.tolist(), exchange.tolist(), "Polynomial AFP vs exchange local RMSE", "AFP", "exchange")

    degree_effect = []
    layout_effect = []
    for inst in instances:
        case = [r for r in rows if r["family"] == inst["family"] and r["index"] == inst["index"]]
        by_degree = [
            np.mean([r["local_rmse"] for r in case if r["degree"] == d])
            for d in POLY_DEGREES
        ]
        by_layout = [
            np.mean([r["local_rmse"] for r in case if r["method"] == m])
            for m in methods
        ]
        degree_effect.append(max(by_degree) - min(by_degree))
        layout_effect.append(max(by_layout) - min(by_layout))
    ratio = float(np.mean(degree_effect) / (np.mean(layout_effect) + 1e-12))
    model_limited = ratio > 1.5
    exchange_significant = float(np.mean(imp)) >= 0.10 and float(np.mean(imp > 0)) >= 0.65 and p_value < 0.05
    interpretation_passed = model_limited and not exchange_significant

    summary_rows = []
    for method in methods:
        for degree in POLY_DEGREES:
            subset = [r for r in rows if r["method"] == method and r["degree"] == degree]
            summary_rows.append(
                {
                    "method": method,
                    "degree": degree,
                    "mean_global_rmse": float(np.mean([r["global_rmse"] for r in subset])),
                    "mean_local_rmse": float(np.mean([r["local_rmse"] for r in subset])),
                    "mean_condition": float(np.mean([r["condition"] for r in subset])),
                }
            )
    write_csv(LAYER_ROOT / "outputs" / "polynomial_only_method_degree_summary.csv", summary_rows)

    reasons = []
    if not model_limited:
        reasons.append("阶数效应未明显大于布局效应，说明当前面型或点数下多项式模型限制不够突出；可增加中频/局部 bump 强度，或降低采样点数。")
    if exchange_significant:
        reasons.append("点交换在多项式平台下仍显著改善，说明采样布局效应未被多项式欠拟合完全掩盖；论文中需避免声称“多项式下采样不重要”。")
    if np.nanmax([r["condition"] for r in rows]) > 1e12:
        reasons.append("部分高阶多项式条件数过大，结果可能混入数值病态；需采用坐标归一化、正则化或限制最高阶数。")
    if not reasons:
        reasons.append("结果符合预期：多项式平台主要受模型表达能力和数值条件限制，新采样方法的优势不应主要在该平台上论证。")

    table = "\n".join(
        f"| {r['method']} | {r['degree']} | {r['mean_global_rmse']:.6g} | {r['mean_local_rmse']:.6g} | {r['mean_condition']:.3g} |"
        for r in summary_rows
    )
    report = (
        "# 仅多项式条件层级验证报告\n\n"
        "## 目的\n\n"
        "在 XY 多项式重建平台下比较传统采样与新方法，判断采样优化收益是否受模型表达能力限制，从而支撑后续使用 cubic PHS 作为主评价平台。\n\n"
        "## 自动评价\n\n"
        f"- 全局最佳平均阶数：{best_degree}。\n"
        f"- 阶数效应/布局效应均值比：{ratio:.2f}。\n"
        f"- 最佳阶数下 AFP exchange 相对 AFP 的平均 local RMSE 改善：{100 * float(np.mean(imp)):.2f}%。\n"
        f"- 最佳阶数下正改善比例：{100 * float(np.mean(imp > 0)):.1f}%。\n"
        f"- Wilcoxon p(exchange < AFP)：{p_value:.4g}。\n\n"
        f"**结论：{'通过，说明多项式平台更适合用于说明模型限制，而非证明采样优化主效果。' if interpretation_passed else '未完全通过，需要重新审视面型、点数或多项式平台的逻辑定位。'}**\n\n"
        "## 方法-阶数统计\n\n"
        "| 方法 | 阶数 | 平均 global RMSE | 平均 local RMSE | 平均条件数 |\n"
        "|---|---:|---:|---:|---:|\n"
        + table
        + "\n\n## 若结果不及预期的原因检查\n\n"
        + "\n".join(f"- {item}" for item in reasons)
        + "\n"
    )
    write_text(LAYER_ROOT / "reports" / "polynomial_only_layer_summary.md", report)
    decision = {
        "passed_interpretation": interpretation_passed,
        "best_degree": int(best_degree),
        "degree_to_layout_effect_ratio": ratio,
        "mean_local_improvement_exchange_vs_afp": float(np.mean(imp)),
        "positive_ratio": float(np.mean(imp > 0)),
        "p_value": p_value,
        "model_limited": model_limited,
        "exchange_significant_under_poly": exchange_significant,
    }
    write_text(LAYER_ROOT / "outputs" / "polynomial_only_decision.json", json.dumps(decision, ensure_ascii=False, indent=2))
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

