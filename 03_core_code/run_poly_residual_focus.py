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
from plotting import save_boxplot, save_line, save_map, save_scatter
from reconstruction import (
    hessian_strength_grid,
    local_mask_from_hessian,
    metrics,
    reconstruct_poly,
    reconstruct_rbf,
    sample_values,
    valid_points,
)
from sampling import afp_sampling, candidate_pool, regular_sampling, weights_from_hessian
from surfaces import generate_instance


LAYER_ROOT = STUDY_ROOT / "poly_residual_focus"
POLY_BASE_DEGREE = 5


def checked_path(path: Path) -> Path:
    return assert_inside_root(path, LAYER_ROOT)


def write_csv(path: Path, rows: list[dict]) -> None:
    path = checked_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path = checked_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def instance_context(config: StudyConfig, instance: dict) -> dict:
    eval_xy, truth = valid_points(instance)
    h_raw = hessian_strength_grid(instance["residual_sag"], instance["mask"], instance["x"], instance["y"])
    local_grid = local_mask_from_hessian(h_raw, instance["mask"], config.local_quantile)
    return {
        "eval_xy": eval_xy,
        "truth": truth,
        "raw_hessian": h_raw,
        "local_grid": local_grid,
        "local_flat": local_grid[instance["mask"]],
    }


def poly_plus_residual_reconstruct(instance: dict, points: np.ndarray, ctx: dict, degree: int) -> dict:
    z_sample = sample_values(instance, points)
    poly_eval, cond = reconstruct_poly(points, z_sample, ctx["eval_xy"], degree)
    poly_sample, _ = reconstruct_poly(points, z_sample, points, degree)
    residual_sample = z_sample - poly_sample
    residual_eval_truth = ctx["truth"] - poly_eval
    residual_pred, _, ok = reconstruct_rbf(points, residual_sample, ctx["eval_xy"], "phs")
    if not ok:
        final_pred = poly_eval
        residual_pred = np.zeros_like(poly_eval)
    else:
        final_pred = poly_eval + residual_pred
    final_metrics = metrics(final_pred, ctx["truth"], ctx["local_flat"])
    residual_metrics = metrics(residual_pred, residual_eval_truth, ctx["local_flat"])
    poly_only_metrics = metrics(poly_eval, ctx["truth"], ctx["local_flat"])
    return {
        "final_pred": final_pred,
        "poly_eval": poly_eval,
        "residual_pred": residual_pred,
        "residual_truth": residual_eval_truth,
        "global_rmse": final_metrics["global_rmse"],
        "local_rmse": final_metrics["local_rmse"],
        "p95_abs": final_metrics["p95_abs"],
        "residual_global_rmse": residual_metrics["global_rmse"],
        "residual_local_rmse": residual_metrics["local_rmse"],
        "poly_only_global_rmse": poly_only_metrics["global_rmse"],
        "poly_only_local_rmse": poly_only_metrics["local_rmse"],
        "condition": cond,
        "success": bool(ok),
    }


def residual_hessian_from_poly_baseline(instance: dict, points: np.ndarray, ctx: dict, degree: int) -> np.ndarray:
    rec = poly_plus_residual_reconstruct(instance, points, ctx, degree)
    residual_grid = np.full_like(instance["residual_sag"], np.nan)
    residual_grid[instance["mask"]] = rec["residual_truth"]
    return hessian_strength_grid(residual_grid, instance["mask"], instance["x"], instance["y"])


def objective(instance: dict, points: np.ndarray, ctx: dict, degree: int, pool: np.ndarray, weights: np.ndarray) -> tuple[float, dict]:
    rec = poly_plus_residual_reconstruct(instance, points, ctx, degree)
    tree = cKDTree(points)
    d, _ = tree.query(pool, k=1)
    coverage = float(np.sum(weights / (d + 0.02)) / (np.sum(weights) + 1e-12))
    score = rec["global_rmse"] + 1.35 * rec["local_rmse"] + 0.75 * rec["residual_local_rmse"] - 2e-4 * coverage
    return float(score), {
        "global_rmse": rec["global_rmse"],
        "local_rmse": rec["local_rmse"],
        "residual_local_rmse": rec["residual_local_rmse"],
        "weighted_coverage": coverage,
    }


def nearest_indices(pool: np.ndarray, points: np.ndarray) -> list[int]:
    tree = cKDTree(pool)
    _, idx = tree.query(points, k=1)
    return list(dict.fromkeys(idx.tolist()))


def exchange_optimize_poly_residual(
    instance: dict,
    initial_points: np.ndarray,
    pool: np.ndarray,
    weights: np.ndarray,
    ctx: dict,
    degree: int,
    seed: int,
    rounds: int,
    trials_per_round: int,
    min_distance: float,
) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    selected_idx = nearest_indices(pool, initial_points)
    selected = set(selected_idx)
    points = pool[selected_idx]
    score, met = objective(instance, points, ctx, degree, pool, weights)
    hist = [{"round": 0, "score": score, **met}]
    candidate_rank = np.argsort(weights)[::-1]
    for r in range(1, rounds + 1):
        replace_rank = np.argsort(weights[selected_idx])
        trial_candidates = [int(i) for i in candidate_rank[: min(220, len(candidate_rank))] if int(i) not in selected]
        rng.shuffle(trial_candidates)
        improved = False
        for cand in trial_candidates[:trials_per_round]:
            dmin = float(np.min(np.linalg.norm(points - pool[cand], axis=1)))
            if dmin < min_distance:
                continue
            rep = int(replace_rank[min(rng.integers(0, max(2, len(replace_rank) // 8)), len(replace_rank) - 1)])
            old = selected_idx[rep]
            trial_idx = selected_idx.copy()
            trial_idx[rep] = cand
            trial_points = pool[trial_idx]
            trial_score, trial_met = objective(instance, trial_points, ctx, degree, pool, weights)
            if trial_score < score:
                selected.remove(old)
                selected.add(cand)
                selected_idx = trial_idx
                points = trial_points
                score = trial_score
                met = trial_met
                improved = True
                break
        hist.append({"round": r, "score": score, **met, "improved": improved})
    return points, hist


def make_instances(config: StudyConfig) -> list[dict]:
    instances = []
    for f_id, family in enumerate(SURFACE_FAMILIES):
        for i in range(config.instances_per_family):
            seed = config.random_seed + 80000 + 1000 * f_id + i
            instances.append(generate_instance(config, family, i, seed))
    return instances


def run() -> dict:
    prepare_runtime(STUDY_ROOT)
    os.environ["MPLCONFIGDIR"] = str(checked_path(LAYER_ROOT / "outputs" / "matplotlib_cache"))
    config = StudyConfig(root=STUDY_ROOT)
    for sub in ("outputs", "reports", "src"):
        checked_path(LAYER_ROOT / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    histories = []
    instances = make_instances(config)
    for inst in instances:
        ctx = instance_context(config, inst)
        pool = candidate_pool(inst, config.candidate_count, inst["seed"] + 1)
        afp = afp_sampling(pool, config.sample_count, inst["seed"] + 2, min_distance=config.min_distance)
        residual_h = residual_hessian_from_poly_baseline(inst, afp, ctx, POLY_BASE_DEGREE)
        weights = weights_from_hessian(inst, residual_h, pool)
        layouts = {
            "regular_poly_only": regular_sampling(inst, config.sample_count),
            "afp_poly_only": afp,
            "regular_poly_residual_phs": regular_sampling(inst, config.sample_count),
            "afp_poly_residual_phs": afp,
            "hessian_weighted_poly_residual_phs": afp_sampling(pool, config.sample_count, inst["seed"] + 3, weights, config.min_distance),
        }
        opt, hist = exchange_optimize_poly_residual(
            inst,
            afp,
            pool,
            weights,
            ctx,
            POLY_BASE_DEGREE,
            inst["seed"] + 4,
            config.optimization_rounds,
            config.optimization_trials_per_round,
            config.min_distance,
        )
        layouts["afp_exchange_poly_residual_phs"] = opt
        for h in hist:
            histories.append({"family": inst["family"], "index": inst["index"], **h})
        for method, points in layouts.items():
            rec = poly_plus_residual_reconstruct(inst, points, ctx, POLY_BASE_DEGREE)
            if method.endswith("poly_only"):
                rows.append(
                    {
                        "family": inst["family"],
                        "index": inst["index"],
                        "seed": inst["seed"],
                        "method": method,
                        "degree": POLY_BASE_DEGREE,
                        "global_rmse": rec["poly_only_global_rmse"],
                        "local_rmse": rec["poly_only_local_rmse"],
                        "residual_global_rmse": np.nan,
                        "residual_local_rmse": np.nan,
                        "p95_abs": np.nan,
                        "condition": rec["condition"],
                        "success": True,
                    }
                )
            else:
                rows.append(
                    {
                        "family": inst["family"],
                        "index": inst["index"],
                        "seed": inst["seed"],
                        "method": method,
                        "degree": POLY_BASE_DEGREE,
                        "global_rmse": rec["global_rmse"],
                        "local_rmse": rec["local_rmse"],
                        "residual_global_rmse": rec["residual_global_rmse"],
                        "residual_local_rmse": rec["residual_local_rmse"],
                        "p95_abs": rec["p95_abs"],
                        "condition": rec["condition"],
                        "success": rec["success"],
                    }
                )
        if inst["index"] == 0:
            rec_afp = poly_plus_residual_reconstruct(inst, afp, ctx, POLY_BASE_DEGREE)
            rec_opt = poly_plus_residual_reconstruct(inst, opt, ctx, POLY_BASE_DEGREE)
            err_afp = np.full_like(inst["residual_sag"], np.nan)
            err_opt = np.full_like(inst["residual_sag"], np.nan)
            res_truth = np.full_like(inst["residual_sag"], np.nan)
            err_afp[inst["mask"]] = rec_afp["final_pred"] - ctx["truth"]
            err_opt[inst["mask"]] = rec_opt["final_pred"] - ctx["truth"]
            res_truth[inst["mask"]] = rec_afp["residual_truth"]
            save_map(checked_path(LAYER_ROOT / "outputs" / f"{inst['family']}_poly_residual_truth.png"), inst["x"], inst["y"], res_truth, f"{inst['family']} residual after polynomial")
            save_map(checked_path(LAYER_ROOT / "outputs" / f"{inst['family']}_afp_poly_residual_error.png"), inst["x"], inst["y"], err_afp, f"{inst['family']} AFP poly+residual error")
            save_map(checked_path(LAYER_ROOT / "outputs" / f"{inst['family']}_exchange_poly_residual_error.png"), inst["x"], inst["y"], err_opt, f"{inst['family']} exchange poly+residual error")
    write_csv(LAYER_ROOT / "outputs" / "poly_residual_focus_metrics.csv", rows)
    write_csv(LAYER_ROOT / "outputs" / "poly_residual_exchange_history.csv", histories)

    methods = sorted(set(r["method"] for r in rows))
    global_box = {m: [r["global_rmse"] for r in rows if r["method"] == m] for m in methods}
    local_box = {m: [r["local_rmse"] for r in rows if r["method"] == m] for m in methods}
    save_boxplot(checked_path(LAYER_ROOT / "outputs" / "poly_residual_global_rmse_boxplot.png"), global_box, "Polynomial baseline plus residual PHS", "global RMSE")
    save_boxplot(checked_path(LAYER_ROOT / "outputs" / "poly_residual_local_rmse_boxplot.png"), local_box, "Polynomial baseline plus residual PHS", "local RMSE")

    afp_poly = np.array([r["local_rmse"] for r in rows if r["method"] == "afp_poly_only"])
    afp_res = np.array([r["local_rmse"] for r in rows if r["method"] == "afp_poly_residual_phs"])
    exchange_res = np.array([r["local_rmse"] for r in rows if r["method"] == "afp_exchange_poly_residual_phs"])
    imp_residual_over_poly = (afp_poly - afp_res) / afp_poly
    imp_exchange_over_afp_res = (afp_res - exchange_res) / afp_res
    imp_exchange_over_poly = (afp_poly - exchange_res) / afp_poly
    p_residual = float(stats.wilcoxon(afp_poly, afp_res, alternative="greater").pvalue)
    p_exchange = float(stats.wilcoxon(afp_res, exchange_res, alternative="greater").pvalue)
    save_scatter(checked_path(LAYER_ROOT / "outputs" / "paired_poly_only_vs_poly_residual.png"), afp_poly.tolist(), afp_res.tolist(), "Polynomial only vs polynomial+residual", "poly only", "poly+residual")
    save_scatter(checked_path(LAYER_ROOT / "outputs" / "paired_afp_residual_vs_exchange_residual.png"), afp_res.tolist(), exchange_res.tolist(), "AFP residual vs exchange residual", "AFP residual", "exchange residual")
    save_line(
        checked_path(LAYER_ROOT / "outputs" / "poly_residual_improvement_percent.png"),
        {
            "afp_residual_over_poly": (100 * imp_residual_over_poly).tolist(),
            "exchange_over_afp_residual": (100 * imp_exchange_over_afp_res).tolist(),
            "exchange_over_poly": (100 * imp_exchange_over_poly).tolist(),
        },
        "Improvement under polynomial-baseline residual focus",
        "percent",
    )

    summary_rows = []
    for m in methods:
        subset = [r for r in rows if r["method"] == m]
        summary_rows.append(
            {
                "method": m,
                "mean_global_rmse": float(np.mean([r["global_rmse"] for r in subset])),
                "mean_local_rmse": float(np.mean([r["local_rmse"] for r in subset])),
                "median_local_rmse": float(np.median([r["local_rmse"] for r in subset])),
                "std_local_rmse": float(np.std([r["local_rmse"] for r in subset], ddof=1)),
                "mean_condition": float(np.mean([r["condition"] for r in subset])),
            }
        )
    write_csv(LAYER_ROOT / "outputs" / "poly_residual_focus_method_summary.csv", summary_rows)

    passed = (
        float(np.mean(imp_residual_over_poly)) >= 0.20
        and float(np.mean(imp_exchange_over_afp_res)) >= 0.10
        and float(np.mean(imp_exchange_over_poly)) >= 0.25
        and float(np.mean(imp_exchange_over_afp_res > 0)) >= 0.70
        and p_residual < 0.05
        and p_exchange < 0.05
    )
    reasons = []
    if float(np.mean(imp_residual_over_poly)) < 0.20:
        reasons.append("多项式后的残差 PHS 精修收益不足，可能是多项式阶数过高已吸收了多数可重建成分，或残差主要为全局低频而非局部细节。")
    if float(np.mean(imp_exchange_over_afp_res)) < 0.10:
        reasons.append("Hessian 引导交换相对 AFP 残差精修的增益不足，需检查残差 Hessian 是否稳定、候选点池是否过稀或采样点数是否过多。")
    if float(np.mean(imp_exchange_over_afp_res > 0)) < 0.70:
        reasons.append("正改善比例不足，说明该两阶段流程对部分面型不稳，需要按面型分组检查局部 bump、边界滚降和中频成分。")
    if np.mean([r["condition"] for r in rows]) > 1e5:
        reasons.append("多项式基底条件数偏高，建议固定 3-5 阶或加入正则化，避免低阶基底本身引入不稳定。")
    if not reasons:
        reasons.append("结果达到预期：多项式作为低阶基底，PHS 残差精修与 Hessian 引导采样构成了更清晰的主线。")

    table = "\n".join(
        f"| {r['method']} | {r['mean_global_rmse']:.6g} | {r['mean_local_rmse']:.6g} | {r['median_local_rmse']:.6g} | {r['std_local_rmse']:.6g} | {r['mean_condition']:.3g} |"
        for r in summary_rows
    )
    report = (
        "# 多项式基底与残差精修评价报告\n\n"
        "## 新逻辑定位\n\n"
        "本实验不再把 XY 多项式与 PHS 作为同层级竞争模型。XY 多项式被定位为低阶基底，用于描述 BFS residual 中的全局低阶形貌；PHS 与 Hessian 引导采样优化只用于描述多项式拟合后的剩余局部残差。\n\n"
        "这种定位避免了无限提高多项式阶数来追逐局部细节的问题，也更符合“低阶趋势 + 局部残差精修”的误差分解逻辑。\n\n"
        "## 自动评价\n\n"
        f"- 多项式基底阶数：{POLY_BASE_DEGREE}。\n"
        f"- `AFP poly+residual PHS` 相对 `AFP poly only` 的 local RMSE 平均改善：{100 * float(np.mean(imp_residual_over_poly)):.2f}%。\n"
        f"- `AFP exchange poly+residual PHS` 相对 `AFP poly+residual PHS` 的 local RMSE 平均改善：{100 * float(np.mean(imp_exchange_over_afp_res)):.2f}%。\n"
        f"- `AFP exchange poly+residual PHS` 相对 `AFP poly only` 的 local RMSE 平均改善：{100 * float(np.mean(imp_exchange_over_poly)):.2f}%。\n"
        f"- 交换相对 AFP 残差精修的正改善比例：{100 * float(np.mean(imp_exchange_over_afp_res > 0)):.1f}%。\n"
        f"- Wilcoxon p(poly only > poly+residual)：{p_residual:.4g}。\n"
        f"- Wilcoxon p(AFP residual > exchange residual)：{p_exchange:.4g}。\n\n"
        f"**结论：{'通过，建议将该两阶段框架作为论文主线表述。' if passed else '未完全通过，建议作为修正方向继续复算或调参。'}**\n\n"
        "## 方法统计\n\n"
        "| 方法 | 平均 global RMSE | 平均 local RMSE | local 中位数 | local 标准差 | 平均条件数 |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + table
        + "\n\n## 若结果不及预期的原因检查\n\n"
        + "\n".join(f"- {item}" for item in reasons)
        + "\n\n## 论文表述建议\n\n"
        "建议把多项式写成基础低阶校正项，而不是高精度重建模型。可表述为：低阶 XY 多项式用于去除 BFS residual 中的全局缓变分量；剩余误差包含边界滚降、局部 bump 和中频起伏等局部细节，因此采用 cubic PHS 进行残差插值，并用 Hessian 强度引导固定点数下的测点重分配。\n"
    )
    write_text(LAYER_ROOT / "reports" / "poly_residual_focus_assessment.md", report)
    decision = {
        "passed": passed,
        "poly_degree": POLY_BASE_DEGREE,
        "mean_local_improvement_afp_residual_over_poly": float(np.mean(imp_residual_over_poly)),
        "mean_local_improvement_exchange_over_afp_residual": float(np.mean(imp_exchange_over_afp_res)),
        "mean_local_improvement_exchange_over_poly": float(np.mean(imp_exchange_over_poly)),
        "positive_ratio_exchange_over_afp_residual": float(np.mean(imp_exchange_over_afp_res > 0)),
        "p_poly_vs_residual": p_residual,
        "p_afp_residual_vs_exchange": p_exchange,
    }
    write_text(LAYER_ROOT / "outputs" / "poly_residual_focus_decision.json", json.dumps(decision, ensure_ascii=False, indent=2))
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

