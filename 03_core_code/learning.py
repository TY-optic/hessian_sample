from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from boundary import checked_open
from config import StudyConfig, SURFACE_FAMILIES
from plotting import save_boxplot, save_points
from reconstruction import hessian_strength_grid, local_mask_from_hessian, reconstruct_rbf, sample_values, valid_points
from sampling import afp_sampling, candidate_pool, exchange_optimize, weights_from_hessian
from surfaces import generate_instance, save_instance
from phase1 import evaluate_layout, instance_eval_context, write_csv, write_md


def density_label(instance: dict, optimized_points: np.ndarray, sigma: float = 0.09) -> np.ndarray:
    xy, _ = valid_points(instance)
    d2 = np.sum((xy[:, None, :] - optimized_points[None, :, :]) ** 2, axis=2)
    label = np.exp(-np.min(d2, axis=1) / (2 * sigma**2))
    label = label / (np.max(label) + 1e-12)
    full = np.zeros_like(instance["residual_sag"], dtype=float)
    full[instance["mask"]] = label
    full[~instance["mask"]] = 0.0
    return full


def downsample_feature(arr: np.ndarray, mask: np.ndarray, size: int = 24) -> np.ndarray:
    ys = np.linspace(0, arr.shape[0] - 1, size).astype(int)
    xs = np.linspace(0, arr.shape[1] - 1, size).astype(int)
    out = np.nan_to_num(arr[np.ix_(ys, xs)], nan=0.0)
    out_mask = mask[np.ix_(ys, xs)].astype(float)
    return np.concatenate([out.ravel(), out_mask.ravel()])


def build_learning_dataset(config: StudyConfig, phase1_passed: bool) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    out = config.outputs / "learning_dataset"
    rows = []
    x_rows = []
    y_rows = []
    if not phase1_passed:
        write_md(
            config.reports / "learning_generalization_summary.md",
            "# 第二阶段学习型方法判断\n\n第一阶段未达到进入条件，因此未执行神经网络学习与泛化验证。该结果不应作为论文主线结论。\n",
            config.root,
        )
        return np.empty((0, 0)), np.empty((0, 0)), rows
    for f_id, family in enumerate(SURFACE_FAMILIES):
        for i in range(config.phase2_train_instances_per_family):
            seed = config.random_seed + 20000 + 1000 * f_id + i
            inst = generate_instance(config, family, i, seed)
            save_instance(inst, out / "raw_instances", config.root)
            ctx = instance_eval_context(config, inst)
            pool = candidate_pool(inst, config.candidate_count, seed + 1)
            weights = weights_from_hessian(inst, ctx["hessian"], pool)
            afp = afp_sampling(pool, config.sample_count, seed + 2, min_distance=config.min_distance)
            opt, hist = exchange_optimize(inst, afp, pool, weights, ctx["local_flat"], seed + 3, max(16, config.optimization_rounds // 2), config.optimization_trials_per_round, config.min_distance)
            label = density_label(inst, opt)
            feature = np.concatenate(
                [
                    downsample_feature(np.nan_to_num(inst["residual_sag"], nan=0.0), inst["mask"], 24),
                    downsample_feature(ctx["hessian"], inst["mask"], 24),
                ]
            )
            target = label[inst["mask"]]
            target_ds = downsample_feature(label, inst["mask"], 24)[: 24 * 24]
            x_rows.append(feature)
            y_rows.append(target_ds)
            rows.append({"family": family, "index": i, "seed": seed, "final_score": hist[-1]["score"], "history_rounds": len(hist)})
    x_arr = np.vstack(x_rows)
    y_arr = np.vstack(y_rows)
    np.savez_compressed(out / "learning_dataset.npz", x=x_arr, y=y_arr)
    write_csv(out / "learning_dataset_manifest.csv", rows, config.root)
    return x_arr, y_arr, rows


def train_density_mlp(config: StudyConfig, x: np.ndarray, y: np.ndarray) -> dict:
    if x.size == 0:
        return {"trained": False}
    rng = np.random.default_rng(config.random_seed + 30000)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0) + 1e-8
    y_mean = y.mean(axis=0)
    x_n = (x - x_mean) / x_std
    hidden = min(64, max(16, x.shape[0] * 2))
    w1 = rng.normal(0, 0.05, size=(x_n.shape[1], hidden))
    b1 = np.zeros(hidden)
    h = np.tanh(x_n @ w1 + b1)
    lam = 1e-2
    a = h.T @ h + lam * np.eye(hidden)
    b = h.T @ y
    w2 = np.linalg.solve(a, b)
    pred = h @ w2
    mse = float(np.mean((pred - y) ** 2))
    model = {"trained": True, "x_mean": x_mean, "x_std": x_std, "w1": w1, "b1": b1, "w2": w2, "train_mse": mse, "grid_size": 24}
    np.savez_compressed(config.models / "density_mlp.npz", **model)
    with checked_open(config.logs / "training" / "density_mlp_log.json", config.root, "w", encoding="utf-8") as f:
        json.dump({"train_mse": mse, "samples": int(x.shape[0]), "hidden": int(hidden)}, f, ensure_ascii=False, indent=2)
    return model


def predict_density(model: dict, feature: np.ndarray, instance: dict) -> np.ndarray:
    x = ((feature[None, :] - model["x_mean"]) / model["x_std"])
    h = np.tanh(x @ model["w1"] + model["b1"])
    pred_ds = np.maximum(h @ model["w2"], 0.0).reshape(model["grid_size"], model["grid_size"])
    pred_full = np.zeros_like(instance["residual_sag"], dtype=float)
    ys = np.linspace(0, pred_full.shape[0] - 1, model["grid_size"]).astype(int)
    xs = np.linspace(0, pred_full.shape[1] - 1, model["grid_size"]).astype(int)
    for row_i, yy in enumerate(ys):
        for col_i, xx in enumerate(xs):
            pred_full[yy, xx] = pred_ds[row_i, col_i]
    pred_full = np.nan_to_num(pred_full, nan=0.0)
    pred_full[~instance["mask"]] = 0.0
    return pred_full


def run_learning_phase(config: StudyConfig, phase1_passed: bool) -> dict:
    x, y, dataset_rows = build_learning_dataset(config, phase1_passed)
    if not phase1_passed:
        return {"ran": False, "reason": "phase1_not_passed"}
    model = train_density_mlp(config, x, y)
    rows = []
    out = config.outputs / "generalization_tests"
    for f_id, family in enumerate(SURFACE_FAMILIES):
        for i in range(config.phase2_test_instances_per_family):
            seed = config.random_seed + 40000 + 1000 * f_id + i
            inst = generate_instance(config, family, i, seed)
            ctx = instance_eval_context(config, inst)
            pool = candidate_pool(inst, config.candidate_count, seed + 1)
            weights = weights_from_hessian(inst, ctx["hessian"], pool)
            methods = {}
            t0 = time.perf_counter()
            afp = afp_sampling(pool, config.sample_count, seed + 2, min_distance=config.min_distance)
            methods["afp_baseline"] = (afp, time.perf_counter() - t0)
            t0 = time.perf_counter()
            methods["hessian_weighted"] = (afp_sampling(pool, config.sample_count, seed + 3, weights, config.min_distance), time.perf_counter() - t0)
            t0 = time.perf_counter()
            opt, _ = exchange_optimize(inst, afp, pool, weights, ctx["local_flat"], seed + 4, config.optimization_rounds, config.optimization_trials_per_round, config.min_distance)
            methods["full_exchange"] = (opt, time.perf_counter() - t0)
            feature = np.concatenate(
                [
                    downsample_feature(np.nan_to_num(inst["residual_sag"], nan=0.0), inst["mask"], 24),
                    downsample_feature(ctx["hessian"], inst["mask"], 24),
                ]
            )
            dens = predict_density(model, feature, inst)
            pred_weights = weights_from_hessian(inst, dens, pool)
            t0 = time.perf_counter()
            nn_init = afp_sampling(pool, config.sample_count, seed + 5, pred_weights, config.min_distance)
            methods["nn_predicted_init"] = (nn_init, time.perf_counter() - t0)
            t0 = time.perf_counter()
            nn_refined, _ = exchange_optimize(inst, nn_init, pool, weights, ctx["local_flat"], seed + 6, max(8, config.optimization_rounds // 4), config.optimization_trials_per_round, config.min_distance)
            methods["nn_predicted_plus_short_exchange"] = (nn_refined, time.perf_counter() - t0)
            base_metric = None
            for method, (pts, runtime) in methods.items():
                m = evaluate_layout(inst, pts, ctx, "phs")
                if method == "afp_baseline":
                    base_metric = m
                rows.append(
                    {
                        "family": family,
                        "index": i,
                        "method": method,
                        "global_rmse": m["global_rmse"],
                        "local_rmse": m["local_rmse"],
                        "runtime_sec": runtime,
                        "global_improvement_over_afp": 0.0 if base_metric is None else (base_metric["global_rmse"] - m["global_rmse"]) / base_metric["global_rmse"],
                        "local_improvement_over_afp": 0.0 if base_metric is None else (base_metric["local_rmse"] - m["local_rmse"]) / base_metric["local_rmse"],
                    }
                )
            if i == 0:
                save_points(out / f"{family}_nn_predicted_points.png", inst["x"], inst["y"], inst["residual_sag"], nn_init, f"{family} NN predicted init")
    write_csv(out / "generalization_metrics.csv", rows, config.root)
    box = {}
    for method in sorted(set(r["method"] for r in rows)):
        box[method] = [r["local_rmse"] for r in rows if r["method"] == method]
    save_boxplot(config.outputs / "learning_results" / "generalization_local_rmse_boxplot.png", box, "Learning generalization local RMSE", "local RMSE")
    nn = [r for r in rows if r["method"] == "nn_predicted_init"]
    nn_short = [r for r in rows if r["method"] == "nn_predicted_plus_short_exchange"]
    full = [r for r in rows if r["method"] == "full_exchange"]
    mean_nn = float(np.mean([r["local_improvement_over_afp"] for r in nn]))
    mean_short = float(np.mean([r["local_improvement_over_afp"] for r in nn_short]))
    mean_full = float(np.mean([r["local_improvement_over_afp"] for r in full]))
    worth = mean_nn > 0 and mean_short >= 0.6 * mean_full
    text = (
        "# 第二阶段学习型方法判断\n\n"
        f"- 训练样本数：{len(dataset_rows)}。\n"
        f"- 轻量 MLP 训练 MSE：{model['train_mse']:.6g}。\n"
        f"- NN 初始化相对 AFP 的平均 local RMSE 改善：{100 * mean_nn:.2f}%。\n"
        f"- NN 初始化加少量交换相对 AFP 的平均 local RMSE 改善：{100 * mean_short:.2f}%。\n"
        f"- 完整交换相对 AFP 的平均 local RMSE 改善：{100 * mean_full:.2f}%。\n\n"
        f"**结论：{'学习型初始化具有补充探索价值，可作为论文扩展候选' if worth else '学习型流程尚不足以作为主文核心结论'}。**\n"
    )
    write_md(config.reports / "learning_generalization_summary.md", text, config.root)
    return {"ran": True, "worth_paper_extension": worth, "mean_nn_local_improvement": mean_nn, "mean_short_local_improvement": mean_short}

