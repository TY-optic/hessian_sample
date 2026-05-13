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
from config import StudyConfig, SURFACE_FAMILIES
from phase1 import evaluate_layout, instance_eval_context
from plotting import save_boxplot, save_line, save_scatter
from sampling import afp_sampling, candidate_pool, exchange_optimize, regular_sampling, weights_from_hessian
from surfaces import generate_instance


LAYER_ROOT = STUDY_ROOT / "traditional_sampling_layer"


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


def pool_values_from_grid(instance: dict, pool: np.ndarray, grid_values: np.ndarray) -> np.ndarray:
    xy = np.column_stack([instance["x"][instance["mask"]], instance["y"][instance["mask"]]])
    tree = cKDTree(xy)
    _, idx = tree.query(pool, k=1)
    return grid_values[instance["mask"]][idx]


def random_uniform(pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n, replace=False)
    return pool[idx]


def latin_hypercube(instance: dict, pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy_valid = np.column_stack([instance["x"][instance["mask"]], instance["y"][instance["mask"]]])
    xmin, ymin = xy_valid.min(axis=0)
    xmax, ymax = xy_valid.max(axis=0)
    u = (np.arange(n) + rng.random(n)) / n
    v = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(v)
    desired = np.column_stack([xmin + u * (xmax - xmin), ymin + v * (ymax - ymin)])
    tree = cKDTree(pool)
    _, idx = tree.query(desired, k=1)
    idx = list(dict.fromkeys(idx.tolist()))
    if len(idx) < n:
        remain = [i for i in range(len(pool)) if i not in set(idx)]
        rng.shuffle(remain)
        idx.extend(remain[: n - len(idx)])
    return pool[idx[:n]]


def jittered_grid(instance: dict, pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = regular_sampling(instance, n)
    scale = 0.035
    desired = base + rng.normal(0.0, scale, size=base.shape)
    tree = cKDTree(pool)
    _, idx = tree.query(desired, k=1)
    idx = list(dict.fromkeys(idx.tolist()))
    if len(idx) < n:
        remain = [i for i in range(len(pool)) if i not in set(idx)]
        rng.shuffle(remain)
        idx.extend(remain[: n - len(idx)])
    return pool[idx[:n]]


def poisson_disk(pool: np.ndarray, n: int, seed: int, min_distance: float) -> np.ndarray:
    # AFP without Hessian weighting is a deterministic Poisson-like blue-noise proxy.
    return afp_sampling(pool, n, seed, min_distance=min_distance)


def make_instances(config: StudyConfig) -> list[dict]:
    instances = []
    for f_id, family in enumerate(SURFACE_FAMILIES):
        for i in range(config.instances_per_family):
            seed = config.random_seed + 60000 + 1000 * f_id + i
            instances.append(generate_instance(config, family, i, seed))
    return instances


def run() -> dict:
    prepare_runtime(STUDY_ROOT)
    os.environ["MPLCONFIGDIR"] = str(checked_path(LAYER_ROOT / "outputs" / "matplotlib_cache"))
    config = StudyConfig(root=STUDY_ROOT)
    for sub in ("outputs", "reports", "src"):
        checked_path(LAYER_ROOT / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    instances = make_instances(config)
    for inst in instances:
        ctx = instance_eval_context(config, inst)
        pool = candidate_pool(inst, config.candidate_count, inst["seed"] + 1)
        weights = weights_from_hessian(inst, ctx["hessian"], pool)
        afp = afp_sampling(pool, config.sample_count, inst["seed"] + 6, min_distance=config.min_distance)
        layouts = {
            "regular_grid": regular_sampling(inst, config.sample_count),
            "random_uniform": random_uniform(pool, config.sample_count, inst["seed"] + 2),
            "jittered_grid": jittered_grid(inst, pool, config.sample_count, inst["seed"] + 3),
            "latin_hypercube": latin_hypercube(inst, pool, config.sample_count, inst["seed"] + 4),
            "poisson_disk": poisson_disk(pool, config.sample_count, inst["seed"] + 5, config.min_distance),
            "afp_baseline": afp,
            "hessian_weighted": afp_sampling(pool, config.sample_count, inst["seed"] + 7, weights, config.min_distance),
        }
        opt, hist = exchange_optimize(
            inst,
            afp,
            pool,
            weights,
            ctx["local_flat"],
            inst["seed"] + 8,
            config.optimization_rounds,
            config.optimization_trials_per_round,
            config.min_distance,
        )
        layouts["afp_exchange"] = opt
        for name, points in layouts.items():
            met = evaluate_layout(inst, points, ctx, "phs")
            rows.append({"family": inst["family"], "index": inst["index"], "seed": inst["seed"], "method": name, **met})
    write_csv(LAYER_ROOT / "outputs" / "traditional_sampling_phs_metrics.csv", rows)

    methods = sorted(set(r["method"] for r in rows))
    local_box = {m: [r["local_rmse"] for r in rows if r["method"] == m] for m in methods}
    global_box = {m: [r["global_rmse"] for r in rows if r["method"] == m] for m in methods}
    save_boxplot(checked_path(LAYER_ROOT / "outputs" / "traditional_local_rmse_boxplot.png"), local_box, "Traditional sampling comparison under cubic PHS", "local RMSE")
    save_boxplot(checked_path(LAYER_ROOT / "outputs" / "traditional_global_rmse_boxplot.png"), global_box, "Traditional sampling comparison under cubic PHS", "global RMSE")

    baseline = np.array([r["local_rmse"] for r in rows if r["method"] == "afp_baseline"])
    exchange = np.array([r["local_rmse"] for r in rows if r["method"] == "afp_exchange"])
    best_traditional_per_case = []
    for inst in instances:
        case_rows = [r for r in rows if r["family"] == inst["family"] and r["index"] == inst["index"] and r["method"] not in ("hessian_weighted", "afp_exchange")]
        best_traditional_per_case.append(min(r["local_rmse"] for r in case_rows))
    best_trad = np.array(best_traditional_per_case)
    imp_afp = (baseline - exchange) / baseline
    imp_best = (best_trad - exchange) / best_trad
    p_afp = float(stats.wilcoxon(baseline, exchange, alternative="greater").pvalue)
    p_best = float(stats.wilcoxon(best_trad, exchange, alternative="greater").pvalue)
    save_scatter(checked_path(LAYER_ROOT / "outputs" / "paired_afp_vs_exchange_local.png"), baseline.tolist(), exchange.tolist(), "AFP vs exchange local RMSE", "AFP", "exchange")
    save_line(checked_path(LAYER_ROOT / "outputs" / "exchange_improvement_percent.png"), {"vs_afp": (100 * imp_afp).tolist(), "vs_best_traditional": (100 * imp_best).tolist()}, "Exchange improvement", "percent")

    by_method = []
    for m in methods:
        subset = [r for r in rows if r["method"] == m]
        by_method.append(
            {
                "method": m,
                "mean_global_rmse": float(np.mean([r["global_rmse"] for r in subset])),
                "mean_local_rmse": float(np.mean([r["local_rmse"] for r in subset])),
                "median_local_rmse": float(np.median([r["local_rmse"] for r in subset])),
                "std_local_rmse": float(np.std([r["local_rmse"] for r in subset], ddof=1)),
            }
        )
    write_csv(LAYER_ROOT / "outputs" / "traditional_sampling_method_summary.csv", by_method)
    passed = (
        float(np.mean(imp_afp)) >= 0.15
        and float(np.mean(imp_best)) >= 0.05
        and float(np.mean(imp_afp > 0)) >= 0.70
        and p_afp < 0.05
    )
    reasons = []
    if float(np.mean(imp_afp)) < 0.15:
        reasons.append("相对 AFP 的 local RMSE 平均改善不足 15%，可能说明 Hessian 引导或交换目标权重仍偏弱。")
    if float(np.mean(imp_best)) < 0.05:
        reasons.append("相对每个实例的最佳传统采样改善不足 5%，可能说明传统蓝噪声/空间均匀性已接近该点数下的上限。")
    if float(np.mean(imp_afp > 0)) < 0.70:
        reasons.append("正改善实例比例不足，需检查面型高变化区域是否足够局部化，或采样点数是否过密导致优化空间变小。")
    if p_afp >= 0.05:
        reasons.append("配对统计检验不显著，需增加随机实例数量或降低单实例噪声。")
    if not reasons:
        reasons.append("结果达到预期；仍建议正式论文复算时提高候选点池和随机实例数。")

    table = "\n".join(
        f"| {r['method']} | {r['mean_global_rmse']:.6g} | {r['mean_local_rmse']:.6g} | {r['median_local_rmse']:.6g} | {r['std_local_rmse']:.6g} |"
        for r in by_method
    )
    report = (
        "# 传统采样方式层级验证报告\n\n"
        "## 目的\n\n"
        "在统一 cubic PHS 重建平台下，比较传统采样策略与 Hessian 引导/点交换方法，判断新方法是否只是优于单一 AFP 基线。\n\n"
        "## 自动评价\n\n"
        f"- AFP exchange 相对 AFP baseline 的平均 local RMSE 改善：{100 * float(np.mean(imp_afp)):.2f}%。\n"
        f"- AFP exchange 相对逐实例最佳传统采样的平均 local RMSE 改善：{100 * float(np.mean(imp_best)):.2f}%。\n"
        f"- 相对 AFP baseline 的正改善比例：{100 * float(np.mean(imp_afp > 0)):.1f}%。\n"
        f"- Wilcoxon p(exchange < AFP)：{p_afp:.4g}。\n"
        f"- Wilcoxon p(exchange < best traditional)：{p_best:.4g}。\n\n"
        f"**结论：{'通过，说明新方法不只是优于单一 AFP 基线。' if passed else '未完全通过，需谨慎解释相对传统采样的优势。'}**\n\n"
        "## 方法统计\n\n"
        "| 方法 | 平均 global RMSE | 平均 local RMSE | local 中位数 | local 标准差 |\n"
        "|---|---:|---:|---:|---:|\n"
        + table
        + "\n\n## 若结果不及预期的原因检查\n\n"
        + "\n".join(f"- {item}" for item in reasons)
        + "\n"
    )
    write_text(LAYER_ROOT / "reports" / "traditional_sampling_layer_summary.md", report)
    decision = {
        "passed": passed,
        "mean_local_improvement_vs_afp": float(np.mean(imp_afp)),
        "mean_local_improvement_vs_best_traditional": float(np.mean(imp_best)),
        "positive_ratio_vs_afp": float(np.mean(imp_afp > 0)),
        "p_vs_afp": p_afp,
        "p_vs_best_traditional": p_best,
    }
    write_text(LAYER_ROOT / "outputs" / "traditional_sampling_decision.json", json.dumps(decision, ensure_ascii=False, indent=2))
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

