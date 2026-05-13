from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from boundary import checked_open
from config import POLY_DEGREES, PHASE1_METHODS, StudyConfig
from plotting import save_boxplot, save_line, save_map, save_points, save_scatter
from reconstruction import (
    hessian_strength_grid,
    local_mask_from_hessian,
    metrics,
    reconstruct_poly,
    reconstruct_rbf,
    sample_values,
    valid_points,
)
from sampling import afp_sampling, candidate_pool, exchange_optimize, objective, regular_sampling, weights_from_hessian
from surfaces import generate_dataset


def write_csv(path: Path, rows: list[dict], root: Path) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with checked_open(path, root, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, text: str, root: Path) -> None:
    with checked_open(path, root, "w", encoding="utf-8") as f:
        f.write(text)


def instance_eval_context(config: StudyConfig, instance: dict) -> dict:
    eval_xy, truth = valid_points(instance)
    h = hessian_strength_grid(instance["residual_sag"], instance["mask"], instance["x"], instance["y"])
    local_grid = local_mask_from_hessian(h, instance["mask"], config.local_quantile)
    local_flat = local_grid[instance["mask"]]
    return {"eval_xy": eval_xy, "truth": truth, "hessian": h, "local_grid": local_grid, "local_flat": local_flat}


def evaluate_layout(instance: dict, points: np.ndarray, context: dict, kind: str = "phs") -> dict:
    z = sample_values(instance, points)
    pred, cond, ok = reconstruct_rbf(points, z, context["eval_xy"], "phs" if kind == "phs" else "gaussian")
    if not ok:
        return {"global_rmse": np.inf, "local_rmse": np.inf, "p95_abs": np.inf, "condition": cond, "success": False}
    out = metrics(pred, context["truth"], context["local_flat"])
    out.update({"condition": cond, "success": True})
    return out


def module_a_dataset(config: StudyConfig) -> list[dict]:
    out_dir = config.outputs / "datasets"
    instances = generate_dataset(config, out_dir)
    rows = [
        {
            "family": inst["family"],
            "index": inst["index"],
            "seed": inst["seed"],
            "grid_size": config.grid_size,
            "valid_grid_points": int(inst["mask"].sum()),
        }
        for inst in instances
    ]
    write_csv(config.outputs / "datasets" / "dataset_manifest.csv", rows, config.root)
    return instances


def module_b_poly(config: StudyConfig, instances: list[dict]) -> list[dict]:
    out = config.outputs / "baseline_poly"
    rows = []
    for inst in instances:
        ctx = instance_eval_context(config, inst)
        regular = regular_sampling(inst, config.sample_count)
        pool = candidate_pool(inst, config.candidate_count, inst["seed"] + 11)
        afp = afp_sampling(pool, config.sample_count, inst["seed"] + 12, min_distance=config.min_distance)
        for layout_name, points in (("regular", regular), ("afp", afp)):
            sample_z = sample_values(inst, points)
            for degree in POLY_DEGREES:
                pred, cond = reconstruct_poly(points, sample_z, ctx["eval_xy"], degree)
                m = metrics(pred, ctx["truth"], ctx["local_flat"])
                rows.append(
                    {
                        "family": inst["family"],
                        "index": inst["index"],
                        "layout": layout_name,
                        "degree": degree,
                        "global_rmse": m["global_rmse"],
                        "local_rmse": m["local_rmse"],
                        "p95_abs": m["p95_abs"],
                        "condition": cond,
                    }
                )
    write_csv(out / "baseline_poly_metrics.csv", rows, config.root)
    series = {}
    for layout in ("regular", "afp"):
        series[layout] = [
            float(np.mean([r["global_rmse"] for r in rows if r["layout"] == layout and r["degree"] == d]))
            for d in POLY_DEGREES
        ]
    save_line(out / "poly_degree_global_rmse.png", series, "XY polynomial degree trend", "global RMSE")
    regular_afp_gap = np.mean(
        [
            abs(
                np.mean([r["global_rmse"] for r in rows if r["layout"] == "regular" and r["degree"] == d])
                - np.mean([r["global_rmse"] for r in rows if r["layout"] == "afp" and r["degree"] == d])
            )
            for d in POLY_DEGREES
        ]
    )
    degree_span = max(series["regular"] + series["afp"]) - min(series["regular"] + series["afp"])
    conclusion = regular_afp_gap < degree_span
    write_md(
        config.reports / "baseline_poly_summary.md",
        "# 基线多项式验证\n\n"
        f"- 实例数：{len(instances)}。\n"
        f"- regular 与 AFP 的平均布局差异：{regular_afp_gap:.6g}。\n"
        f"- 阶数变化导致的 RMSE 跨度：{degree_span:.6g}。\n"
        f"- 判定：{'基本支持低阶 XY 多项式主要受表达能力限制' if conclusion else '不支持该判定'}。\n",
        config.root,
    )
    return rows


def module_c_rbf_compare(config: StudyConfig, instances: list[dict]) -> list[dict]:
    out = config.outputs / "rbf_compare"
    rows = []
    for inst in instances:
        ctx = instance_eval_context(config, inst)
        pool = candidate_pool(inst, config.candidate_count, inst["seed"] + 21)
        afp = afp_sampling(pool, config.sample_count, inst["seed"] + 22, min_distance=config.min_distance)
        h_weight = weights_from_hessian(inst, ctx["hessian"], pool)
        weighted = afp_sampling(pool, config.sample_count, inst["seed"] + 23, h_weight, config.min_distance)
        for layout_name, points in (("afp", afp), ("hessian_weighted", weighted)):
            for kind in ("phs", "gaussian"):
                m = evaluate_layout(inst, points, ctx, kind)
                rows.append({"family": inst["family"], "index": inst["index"], "layout": layout_name, "rbf": kind, **m})
        if inst["index"] == 0:
            pred, _, _ = reconstruct_rbf(afp, sample_values(inst, afp), ctx["eval_xy"], "phs")
            err = np.full_like(inst["residual_sag"], np.nan)
            err[inst["mask"]] = pred - ctx["truth"]
            save_map(out / f"{inst['family']}_afp_phs_error.png", inst["x"], inst["y"], err, f"{inst['family']} AFP PHS error")
    write_csv(out / "rbf_compare_metrics.csv", rows, config.root)
    phs_fail = sum(1 for r in rows if r["rbf"] == "phs" and not r["success"])
    gau_fail = sum(1 for r in rows if r["rbf"] == "gaussian" and not r["success"])
    phs_mean = np.mean([r["global_rmse"] for r in rows if r["rbf"] == "phs" and np.isfinite(r["global_rmse"])])
    gau_mean = np.mean([r["global_rmse"] for r in rows if r["rbf"] == "gaussian" and np.isfinite(r["global_rmse"])])
    write_md(
        config.reports / "rbf_compare_summary.md",
        "# PHS 与 Gaussian RBF 对比\n\n"
        f"- cubic PHS 失败数：{phs_fail}。\n"
        f"- Gaussian RBF 失败数：{gau_fail}。\n"
        f"- cubic PHS 平均 global RMSE：{phs_mean:.6g}。\n"
        f"- Gaussian RBF 平均 global RMSE：{gau_mean:.6g}。\n"
        f"- 判定：{'cubic PHS 更适合作为后续采样评价平台' if phs_fail <= gau_fail and phs_mean <= 1.2 * gau_mean else 'PHS 优势不充分，需谨慎解释'}。\n",
        config.root,
    )
    return rows


def module_d_hessian_error(config: StudyConfig, instances: list[dict]) -> list[dict]:
    out = config.outputs / "hessian_error_relation"
    rows = []
    for inst in instances:
        ctx = instance_eval_context(config, inst)
        pool = candidate_pool(inst, config.candidate_count, inst["seed"] + 31)
        afp = afp_sampling(pool, config.sample_count, inst["seed"] + 32, min_distance=config.min_distance)
        z = sample_values(inst, afp)
        pred, _, _ = reconstruct_rbf(afp, z, ctx["eval_xy"], "phs")
        m = metrics(pred, ctx["truth"], ctx["local_flat"])
        abs_err = np.full_like(inst["residual_sag"], np.nan)
        abs_err[inst["mask"]] = np.abs(pred - ctx["truth"])
        corr = float(stats.spearmanr(ctx["hessian"][inst["mask"]], abs_err[inst["mask"]], nan_policy="omit").statistic)
        rows.append({"family": inst["family"], "index": inst["index"], "global_rmse": m["global_rmse"], "local_rmse": m["local_rmse"], "local_global_ratio": m["local_rmse"] / m["global_rmse"], "spearman_hessian_abs_error": corr})
        if inst["index"] == 0:
            save_map(out / f"{inst['family']}_hessian.png", inst["x"], inst["y"], ctx["hessian"], f"{inst['family']} Hessian strength")
            save_map(out / f"{inst['family']}_abs_error.png", inst["x"], inst["y"], abs_err, f"{inst['family']} baseline abs error")
            save_scatter(out / f"{inst['family']}_hessian_error_scatter.png", ctx["hessian"][inst["mask"]].tolist(), abs_err[inst["mask"]].tolist(), "Hessian vs abs error", "Hessian", "abs error")
    write_csv(out / "hessian_error_relation_metrics.csv", rows, config.root)
    mean_ratio = float(np.mean([r["local_global_ratio"] for r in rows]))
    mean_corr = float(np.nanmean([r["spearman_hessian_abs_error"] for r in rows]))
    write_md(
        config.reports / "hessian_error_relation_summary.md",
        "# 二阶变化与重建误差关系\n\n"
        f"- 平均 local/global RMSE 比值：{mean_ratio:.3f}。\n"
        f"- Hessian 与绝对误差 Spearman 相关均值：{mean_corr:.3f}。\n"
        f"- 判定：{'高二阶变化区域整体对应更高误差' if mean_ratio > 1.10 else '高二阶变化区域与误差集中关系不充分'}。\n",
        config.root,
    )
    return rows


def module_e_f_optimization_and_significance(config: StudyConfig, instances: list[dict]) -> tuple[list[dict], dict]:
    out_opt = config.outputs / "optimization"
    out_sig = config.outputs / "significance_tests"
    log_dir = config.logs / "optimization_logs"
    rows = []
    layouts: dict[str, dict[tuple[str, int], np.ndarray]] = {m: {} for m in PHASE1_METHODS}
    for inst in instances:
        ctx = instance_eval_context(config, inst)
        pool = candidate_pool(inst, config.candidate_count, inst["seed"] + 41)
        weights_true = weights_from_hessian(inst, ctx["hessian"], pool)
        afp = afp_sampling(pool, config.sample_count, inst["seed"] + 42, min_distance=config.min_distance)
        h_weighted = afp_sampling(pool, config.sample_count, inst["seed"] + 43, weights_true, config.min_distance)
        points_opt, hist = exchange_optimize(inst, afp, pool, weights_true, ctx["local_flat"], inst["seed"] + 44, config.optimization_rounds, config.optimization_trials_per_round, config.min_distance)
        afp_pred, _, _ = reconstruct_rbf(afp, sample_values(inst, afp), ctx["eval_xy"], "phs")
        coarse = np.full_like(inst["residual_sag"], np.nan)
        coarse[inst["mask"]] = afp_pred
        est_h = hessian_strength_grid(coarse, inst["mask"], inst["x"], inst["y"])
        weights_est = weights_from_hessian(inst, est_h, pool)
        points_est, hist_est = exchange_optimize(inst, afp, pool, weights_est, ctx["local_flat"], inst["seed"] + 45, max(12, config.optimization_rounds // 2), config.optimization_trials_per_round, config.min_distance)
        method_points = {
            "afp_phs": afp,
            "hessian_weighted_phs": h_weighted,
            "afp_exchange_phs": points_opt,
            "estimated_hessian_exchange_phs": points_est,
        }
        for method, points in method_points.items():
            m = evaluate_layout(inst, points, ctx, "phs")
            rows.append({"family": inst["family"], "index": inst["index"], "seed": inst["seed"], "method": method, **m})
            layouts[method][(inst["family"], inst["index"])] = points
        write_csv(log_dir / f"{inst['family']}_{inst['index']:02d}_true_hessian_exchange.csv", hist, config.root)
        write_csv(log_dir / f"{inst['family']}_{inst['index']:02d}_estimated_hessian_exchange.csv", hist_est, config.root)
        if inst["index"] == 0:
            save_points(out_opt / f"{inst['family']}_afp_exchange_points.png", inst["x"], inst["y"], inst["residual_sag"], points_opt, f"{inst['family']} AFP exchange")
    write_csv(out_opt / "optimization_metrics.csv", rows, config.root)
    write_csv(out_sig / "significance_metrics.csv", rows, config.root)

    baseline = [r for r in rows if r["method"] == "afp_phs"]
    exchange = [r for r in rows if r["method"] == "afp_exchange_phs"]
    base_g = np.array([r["global_rmse"] for r in baseline])
    base_l = np.array([r["local_rmse"] for r in baseline])
    ex_g = np.array([r["global_rmse"] for r in exchange])
    ex_l = np.array([r["local_rmse"] for r in exchange])
    global_imp = (base_g - ex_g) / base_g
    local_imp = (base_l - ex_l) / base_l
    try:
        p_global = float(stats.wilcoxon(base_g, ex_g, alternative="greater").pvalue)
        p_local = float(stats.wilcoxon(base_l, ex_l, alternative="greater").pvalue)
    except ValueError:
        p_global = 1.0
        p_local = 1.0
    criteria = {
        "mean_local_improvement_ge_15pct": float(np.mean(local_imp)) >= 0.15,
        "mean_global_improvement_ge_5pct": float(np.mean(global_imp)) >= 0.05,
        "local_positive_ratio_ge_70pct": float(np.mean(local_imp > 0)) >= 0.70,
        "global_positive_ratio_ge_60pct": float(np.mean(global_imp > 0)) >= 0.60,
        "wilcoxon_significant": p_local < 0.05 and p_global < 0.05,
    }
    passed = all(criteria.values())
    data_global = {m: [r["global_rmse"] for r in rows if r["method"] == m] for m in PHASE1_METHODS}
    data_local = {m: [r["local_rmse"] for r in rows if r["method"] == m] for m in PHASE1_METHODS}
    save_boxplot(out_sig / "global_rmse_boxplot.png", data_global, "Global RMSE by method", "global RMSE")
    save_boxplot(out_sig / "local_rmse_boxplot.png", data_local, "Local RMSE by method", "local RMSE")
    save_scatter(out_sig / "paired_local_afp_vs_exchange.png", base_l.tolist(), ex_l.tolist(), "Paired local RMSE", "AFP + PHS", "AFP exchange + PHS")
    save_line(out_sig / "improvement_percent.png", {"global": (100 * global_imp).tolist(), "local": (100 * local_imp).tolist()}, "Improvement over AFP", "improvement percent")

    best_i = int(np.argmax(local_imp))
    worst_i = int(np.argmin(local_imp))
    for tag, i in (("best", best_i), ("worst", worst_i)):
        inst = instances[i]
        ctx = instance_eval_context(config, inst)
        key = (inst["family"], inst["index"])
        for method in ("afp_phs", "afp_exchange_phs"):
            pts = layouts[method][key]
            pred, _, _ = reconstruct_rbf(pts, sample_values(inst, pts), ctx["eval_xy"], "phs")
            err = np.full_like(inst["residual_sag"], np.nan)
            err[inst["mask"]] = pred - ctx["truth"]
            save_map(out_sig / f"{tag}_{inst['family']}_{method}_error.png", inst["x"], inst["y"], err, f"{tag} {method} error")

    summary = {
        "passed_phase1": passed,
        "mean_global_improvement": float(np.mean(global_imp)),
        "mean_local_improvement": float(np.mean(local_imp)),
        "global_positive_ratio": float(np.mean(global_imp > 0)),
        "local_positive_ratio": float(np.mean(local_imp > 0)),
        "wilcoxon_p_global": p_global,
        "wilcoxon_p_local": p_local,
        "criteria": criteria,
    }
    with checked_open(out_sig / "phase1_decision.json", config.root, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    assessment = phase1_assessment_text(summary, rows)
    write_md(config.reports / "paper_level_assessment.md", assessment, config.root)
    return rows, summary


def phase1_assessment_text(summary: dict, rows: list[dict]) -> str:
    method_stats = []
    for method in PHASE1_METHODS:
        subset = [r for r in rows if r["method"] == method]
        method_stats.append(
            f"| {method} | {np.mean([r['global_rmse'] for r in subset]):.6g} | "
            f"{np.mean([r['local_rmse'] for r in subset]):.6g} | "
            f"{np.median([r['local_rmse'] for r in subset]):.6g} | "
            f"{np.std([r['local_rmse'] for r in subset], ddof=1):.6g} |"
        )
    criteria_lines = "\n".join([f"- {k}: {'是' if v else '否'}" for k, v in summary["criteria"].items()])
    verdict = "达到论文级显著区分，可进入第二阶段学习验证。" if summary["passed_phase1"] else "未达到论文级显著区分，不应进入第二阶段作为主线结果。"
    short = (
        "当前方法相对 AFP 基线形成了足够稳定的 local 与 global 改善。"
        if summary["passed_phase1"]
        else "当前方法的优势不足以支撑论文主结论，主要短板在于改善幅度、改善比例或统计显著性至少一项不满足阈值。"
    )
    return (
        "# 第一阶段论文级显著区分判断\n\n"
        f"**结论：{verdict}**\n\n"
        "## 关键统计\n\n"
        f"- AFP exchange 相对 AFP + PHS 的平均 local RMSE 改善：{100 * summary['mean_local_improvement']:.2f}%。\n"
        f"- AFP exchange 相对 AFP + PHS 的平均 global RMSE 改善：{100 * summary['mean_global_improvement']:.2f}%。\n"
        f"- local RMSE 正改善实例比例：{100 * summary['local_positive_ratio']:.1f}%。\n"
        f"- global RMSE 正改善实例比例：{100 * summary['global_positive_ratio']:.1f}%。\n"
        f"- Wilcoxon p(local)：{summary['wilcoxon_p_local']:.4g}。\n"
        f"- Wilcoxon p(global)：{summary['wilcoxon_p_global']:.4g}。\n\n"
        "## 方法统计\n\n"
        "| 方法 | 平均 global RMSE | 平均 local RMSE | local 中位数 | local 标准差 |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(method_stats)
        + "\n\n## 阈值自检\n\n"
        + criteria_lines
        + "\n\n## 论文可用性判断\n\n"
        + short
        + "\n"
    )


def run_phase1(config: StudyConfig) -> tuple[list[dict], dict]:
    instances = module_a_dataset(config)
    module_b_poly(config, instances)
    module_c_rbf_compare(config, instances)
    module_d_hessian_error(config, instances)
    rows, decision = module_e_f_optimization_and_significance(config, instances)
    checklist = (
        "# 第一阶段自检\n\n"
        f"- 随机面型实例数：{len(instances)}，{'满足' if len(instances) >= 20 else '不足'}。\n"
        "- 统一对比表格：已生成 `outputs/significance_tests/significance_metrics.csv`。\n"
        "- 误差热图、统计图、趋势图：已生成于 `outputs/` 各模块目录。\n"
        f"- 显著性判断：{'通过' if decision['passed_phase1'] else '未通过'}。\n"
        f"- 是否进入第二阶段：{'是' if decision['passed_phase1'] else '否'}。\n"
    )
    write_md(config.reports / "phase1_self_check.md", checklist, config.root)
    return rows, decision
