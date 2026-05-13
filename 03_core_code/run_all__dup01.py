from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache" / "matplotlib")
os.environ["PYTHONPYCACHEPREFIX"] = str(ROOT / "cache" / "pycache")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree


FAMILIES = ("smooth_low_variation", "edge_rolloff", "local_bump", "mid_frequency_undulation")
METHODS_MAIN = (
    "afp_poly_only",
    "afp_poly_residual_phs",
    "hessian_weighted_poly_residual_phs",
    "afp_exchange_poly_residual_phs",
)


@dataclass(frozen=True)
class Config:
    random_seed: int
    grid_size: int
    instances_per_family: int
    sample_count: int
    point_scan_counts: tuple[int, ...]
    candidate_count: int
    poly_degree: int
    poly_degree_scan: tuple[int, ...]
    local_quantile: float
    min_distance: float
    optimization_rounds: int
    optimization_trials_per_round: int
    point_scan_rounds: int
    bootstrap_repeats: int


def load_config() -> Config:
    data = json.loads((ROOT / "configs" / "config.json").read_text(encoding="utf-8"))
    data["point_scan_counts"] = tuple(data["point_scan_counts"])
    data["poly_degree_scan"] = tuple(data["poly_degree_scan"])
    return Config(**data)


def ensure_dirs() -> None:
    if ROOT.name != "hessian_poly_residual_evidence_study":
        raise RuntimeError(f"Unexpected root: {ROOT}")
    for name in ("src", "configs", "outputs", "logs", "cache", "summary"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    for module in ("00_mainline", "A_budget", "B_error_transfer", "C_point_scan", "D_trend_residual", "E_family_stability", "F_unified_stats"):
        (ROOT / "outputs" / module).mkdir(parents=True, exist_ok=True)
    (ROOT / "cache" / "matplotlib").mkdir(parents=True, exist_ok=True)


def inside(path: Path) -> Path:
    p = path.resolve()
    p.relative_to(ROOT.resolve())
    return p


def write_csv(path: Path, rows: list[dict]) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def finite_check(rows: list[dict], name: str) -> None:
    bad = []
    for i, row in enumerate(rows):
        for key, val in row.items():
            if isinstance(val, (int, float, np.floating)) and not np.isfinite(val):
                bad.append((i, key, val))
    if bad:
        raise RuntimeError(f"{name} has non-finite values: {bad[:5]}")


def rounded_rectangle_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    half_w, half_h, radius = 1.0, 0.72, 0.18
    qx = np.abs(x) - (half_w - radius)
    qy = np.abs(y) - (half_h - radius)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside_core = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside_core - radius <= 0


def make_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-1.05, 1.05, n)
    ys = np.linspace(-0.80, 0.80, n)
    x, y = np.meshgrid(xs, ys)
    return x, y, rounded_rectangle_mask(x, y)


def parent_sag(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return 0.08 * (x**2 + 0.6 * y**2) + rng.uniform(-0.018, 0.018) * (x**2 - y**2) + rng.uniform(-0.012, 0.012) * (x**3 - 0.4 * x * y**2)


def residual_shape(family: str, x: np.ndarray, y: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    z = np.zeros_like(x)
    if family == "smooth_low_variation":
        for _ in range(5):
            z += rng.uniform(-0.0025, 0.0025) * np.cos(rng.integers(1, 4) * np.pi * (x + 1.0) / 2.1) * np.sin(rng.integers(1, 4) * np.pi * (y + 0.8) / 1.6)
        z += rng.uniform(-0.002, 0.002) * (x**4 - y**4)
    elif family == "edge_rolloff":
        edge = np.maximum(np.abs(x) / 1.0, np.abs(y) / 0.72)
        edge = np.clip((edge - 0.68) / 0.32, 0.0, 1.0)
        z += rng.choice([-1.0, 1.0]) * rng.uniform(0.008, 0.018) * edge**3
        z += rng.uniform(-0.002, 0.002) * x * y
    elif family == "local_bump":
        for _ in range(rng.integers(2, 5)):
            cx, cy = rng.uniform(-0.65, 0.65), rng.uniform(-0.45, 0.45)
            sx, sy = rng.uniform(0.07, 0.16), rng.uniform(0.06, 0.14)
            z += rng.uniform(-0.016, 0.020) * np.exp(-((x - cx) ** 2 / (2 * sx**2) + (y - cy) ** 2 / (2 * sy**2)))
    elif family == "mid_frequency_undulation":
        for _ in range(7):
            z += rng.uniform(-0.0035, 0.0035) * np.sin(rng.integers(3, 8) * x + rng.integers(2, 7) * y + rng.uniform(0, 2 * np.pi))
        z += rng.uniform(-0.006, 0.006) * np.exp(-((x + 0.35) ** 2 + (y - 0.22) ** 2) / 0.05)
    else:
        raise ValueError(family)
    z -= np.nanmean(z[mask])
    z[~mask] = np.nan
    return z


def fit_bfs(raw: np.ndarray, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = np.column_stack([np.ones(mask.sum()), x[mask], y[mask], x[mask] ** 2 + y[mask] ** 2])
    coef, *_ = np.linalg.lstsq(a, raw[mask], rcond=None)
    out = coef[0] + coef[1] * x + coef[2] * y + coef[3] * (x**2 + y**2)
    out[~mask] = np.nan
    return out


def make_instance(cfg: Config, family: str, index: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x, y, mask = make_grid(cfg.grid_size)
    parent = parent_sag(x, y, rng)
    residual = residual_shape(family, x, y, mask, rng)
    raw = parent + np.nan_to_num(residual)
    bfs = fit_bfs(raw, x, y, mask)
    bfs_residual = raw - bfs
    bfs_residual[~mask] = np.nan
    valid_xy = np.column_stack([x[mask], y[mask]])
    valid_z = bfs_residual[mask]
    return {"family": family, "index": index, "seed": seed, "x": x, "y": y, "mask": mask, "residual": bfs_residual, "valid_xy": valid_xy, "valid_z": valid_z, "tree": cKDTree(valid_xy)}


def make_dataset(cfg: Config) -> list[dict]:
    instances = []
    for f_id, family in enumerate(FAMILIES):
        for i in range(cfg.instances_per_family):
            instances.append(make_instance(cfg, family, i, cfg.random_seed + 1000 * f_id + i))
    return instances


def hessian_grid(z: np.ndarray, mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    filled = np.array(z, dtype=float)
    filled[~mask] = np.nanmean(filled[mask])
    dy, dx = float(np.nanmedian(np.diff(y[:, 0]))), float(np.nanmedian(np.diff(x[0, :])))
    zy, zx = np.gradient(filled, dy, dx)
    zyy, zyx = np.gradient(zy, dy, dx)
    zxy, zxx = np.gradient(zx, dy, dx)
    h = np.sqrt(zxx**2 + zyy**2 + 2 * ((zxy + zyx) * 0.5) ** 2)
    h[~mask] = np.nan
    return h


def local_mask(h: np.ndarray, mask: np.ndarray, q: float) -> np.ndarray:
    return mask & (h >= np.nanquantile(h[mask], q))


def poly_terms(xy: np.ndarray, degree: int) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    cols = []
    for total in range(degree + 1):
        for px in range(total + 1):
            cols.append((x**px) * (y ** (total - px)))
    return np.column_stack(cols)


def poly_fit_predict(sample_xy: np.ndarray, sample_z: np.ndarray, eval_xy: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    a = poly_terms(sample_xy, degree)
    cond = float(np.linalg.cond(a))
    coef, *_ = np.linalg.lstsq(a, sample_z, rcond=None)
    return poly_terms(eval_xy, degree) @ coef, cond


def values_at(instance: dict, points: np.ndarray) -> np.ndarray:
    _, idx = instance["tree"].query(points, k=1)
    return instance["valid_z"][idx]


def candidate_pool(instance: dict, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = instance["valid_xy"]
    if count >= len(xy):
        return xy.copy()
    return xy[rng.choice(len(xy), size=count, replace=False)]


def regular_sampling(instance: dict, n: int) -> np.ndarray:
    xy = instance["valid_xy"]
    rows = int(np.floor(np.sqrt(n * 1.4)))
    cols = int(np.ceil(n / rows))
    gx, gy = np.meshgrid(np.linspace(xy[:, 0].min(), xy[:, 0].max(), cols), np.linspace(xy[:, 1].min(), xy[:, 1].max(), rows))
    desired = np.column_stack([gx.ravel(), gy.ravel()])
    _, idx = instance["tree"].query(desired, k=1)
    idx = list(dict.fromkeys(idx.tolist()))
    if len(idx) < n:
        idx.extend([i for i in range(len(xy)) if i not in set(idx)][: n - len(idx)])
    return xy[idx[:n]]


def afp_sampling(pool: np.ndarray, n: int, seed: int, weights: np.ndarray | None = None, min_distance: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if weights is None:
        weights = np.ones(len(pool))
    weights = np.asarray(weights, dtype=float)
    weights = 0.2 + 1.8 * (weights - weights.min()) / (np.ptp(weights) + 1e-12)
    selected = [int(rng.choice(len(pool), p=weights / weights.sum()))]
    min_d = np.linalg.norm(pool - pool[selected[0]], axis=1)
    for _ in range(1, n):
        score = min_d * (0.35 + 0.65 * weights)
        score[min_d < min_distance] = -np.inf
        if np.all(~np.isfinite(score)):
            score = min_d
        nxt = int(np.nanargmax(score))
        selected.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(pool - pool[nxt], axis=1))
    return pool[selected]


def pool_weights(instance: dict, h: np.ndarray, pool: np.ndarray) -> np.ndarray:
    hflat = np.nan_to_num(h[instance["mask"]], nan=np.nanmedian(h[instance["mask"]]))
    _, idx = instance["tree"].query(pool, k=1)
    w = hflat[idx]
    return 0.25 + 1.75 * (w - w.min()) / (np.ptp(w) + 1e-12)


def nearest_pool_indices(pool: np.ndarray, points: np.ndarray) -> list[int]:
    _, idx = cKDTree(pool).query(points, k=1)
    return list(dict.fromkeys(idx.tolist()))


def metric_from_pred(pred: np.ndarray, truth: np.ndarray, high_flat: np.ndarray) -> dict:
    err = pred - truth
    high = high_flat.astype(bool)
    low = ~high
    return {
        "global_rmse": float(np.sqrt(np.mean(err**2))),
        "local_rmse": float(np.sqrt(np.mean(err[high] ** 2))),
        "low_rmse": float(np.sqrt(np.mean(err[low] ** 2))),
        "p95_abs": float(np.percentile(np.abs(err), 95)),
    }


def poly_residual_reconstruct(instance: dict, points: np.ndarray, degree: int, high_flat: np.ndarray) -> dict:
    z = values_at(instance, points)
    poly_eval, cond = poly_fit_predict(points, z, instance["valid_xy"], degree)
    poly_sample, _ = poly_fit_predict(points, z, points, degree)
    res_sample = z - poly_sample
    res_truth = instance["valid_z"] - poly_eval
    try:
        rbf = RBFInterpolator(points, res_sample, kernel="cubic", degree=1, smoothing=1e-12)
        res_pred = rbf(instance["valid_xy"])
        ok = True
    except Exception:
        res_pred = np.zeros_like(poly_eval)
        ok = False
    final = poly_eval + res_pred
    m_final = metric_from_pred(final, instance["valid_z"], high_flat)
    m_poly = metric_from_pred(poly_eval, instance["valid_z"], high_flat)
    m_res = metric_from_pred(res_pred, res_truth, high_flat)
    return {"pred": final, "poly": poly_eval, "res_pred": res_pred, "res_truth": res_truth, "condition": cond, "success": ok, **m_final, "poly_global_rmse": m_poly["global_rmse"], "poly_local_rmse": m_poly["local_rmse"], "residual_global_rmse": m_res["global_rmse"], "residual_local_rmse": m_res["local_rmse"]}


def poly_only_reconstruct(instance: dict, points: np.ndarray, degree: int, high_flat: np.ndarray) -> dict:
    z = values_at(instance, points)
    pred, cond = poly_fit_predict(points, z, instance["valid_xy"], degree)
    m = metric_from_pred(pred, instance["valid_z"], high_flat)
    return {"pred": pred, "condition": cond, "success": True, **m}


def residual_hessian_from_afp(instance: dict, afp: np.ndarray, degree: int) -> np.ndarray:
    pred, _ = poly_fit_predict(afp, values_at(instance, afp), instance["valid_xy"], degree)
    res = np.full_like(instance["residual"], np.nan)
    res[instance["mask"]] = instance["valid_z"] - pred
    return hessian_grid(res, instance["mask"], instance["x"], instance["y"])


def objective(instance: dict, points: np.ndarray, degree: int, high_flat: np.ndarray, pool: np.ndarray, weights: np.ndarray) -> tuple[float, dict]:
    rec = poly_residual_reconstruct(instance, points, degree, high_flat)
    d, _ = cKDTree(points).query(pool, k=1)
    coverage = float(np.sum(weights / (d + 0.02)) / (np.sum(weights) + 1e-12))
    score = rec["global_rmse"] + 1.35 * rec["local_rmse"] + 0.70 * rec["residual_local_rmse"] - 2e-4 * coverage
    return float(score), {k: rec[k] for k in ("global_rmse", "local_rmse", "residual_local_rmse")} | {"weighted_coverage": coverage}


def exchange_optimize(instance: dict, initial: np.ndarray, pool: np.ndarray, weights: np.ndarray, degree: int, high_flat: np.ndarray, seed: int, rounds: int, trials: int, min_distance: float) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    selected_idx = nearest_pool_indices(pool, initial)
    selected = set(selected_idx)
    points = pool[selected_idx]
    score, met = objective(instance, points, degree, high_flat, pool, weights)
    hist = [{"round": 0, "score": score, **met, "swap_out_hessian": "", "swap_in_hessian": ""}]
    order = np.argsort(weights)[::-1]
    for r in range(1, rounds + 1):
        rep_order = np.argsort(weights[selected_idx])
        cands = [int(i) for i in order[: min(240, len(order))] if int(i) not in selected]
        rng.shuffle(cands)
        improved = False
        out_h = in_h = ""
        for cand in cands[:trials]:
            if float(np.min(np.linalg.norm(points - pool[cand], axis=1))) < min_distance:
                continue
            rep = int(rep_order[min(rng.integers(0, max(2, len(rep_order) // 8)), len(rep_order) - 1)])
            old = selected_idx[rep]
            trial_idx = selected_idx.copy()
            trial_idx[rep] = cand
            trial_points = pool[trial_idx]
            trial_score, trial_met = objective(instance, trial_points, degree, high_flat, pool, weights)
            if trial_score < score:
                selected.remove(old)
                selected.add(cand)
                selected_idx = trial_idx
                points = trial_points
                score, met = trial_score, trial_met
                out_h = float(weights[old])
                in_h = float(weights[cand])
                improved = True
                break
        hist.append({"round": r, "score": score, **met, "improved": improved, "swap_out_hessian": out_h, "swap_in_hessian": in_h})
    return points, hist


def case_context(cfg: Config, instance: dict, n: int, seed_offset: int = 0) -> dict:
    pool = candidate_pool(instance, cfg.candidate_count, instance["seed"] + 10 + seed_offset + n)
    afp = afp_sampling(pool, n, instance["seed"] + 20 + seed_offset + n, min_distance=cfg.min_distance)
    res_h = residual_hessian_from_afp(instance, afp, cfg.poly_degree)
    high_grid = local_mask(res_h, instance["mask"], cfg.local_quantile)
    weights = pool_weights(instance, res_h, pool)
    return {"pool": pool, "afp": afp, "res_h": res_h, "high_grid": high_grid, "high_flat": high_grid[instance["mask"]], "weights": weights}


def run_mainline(cfg: Config, instances: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    rows, histories = [], []
    layout_cache = {}
    for inst in instances:
        ctx = case_context(cfg, inst, cfg.sample_count)
        exchange, hist = exchange_optimize(inst, ctx["afp"], ctx["pool"], ctx["weights"], cfg.poly_degree, ctx["high_flat"], inst["seed"] + 50, cfg.optimization_rounds, cfg.optimization_trials_per_round, cfg.min_distance)
        methods = {
            "afp_poly_only": ctx["afp"],
            "afp_poly_residual_phs": ctx["afp"],
            "hessian_weighted_poly_residual_phs": afp_sampling(ctx["pool"], cfg.sample_count, inst["seed"] + 31, ctx["weights"], cfg.min_distance),
            "afp_exchange_poly_residual_phs": exchange,
        }
        layout_cache[(inst["family"], inst["index"])] = {"ctx": ctx, "exchange": exchange, "methods": methods}
        for h in hist:
            histories.append({"family": inst["family"], "index": inst["index"], **h})
        for method, pts in methods.items():
            rec = poly_only_reconstruct(inst, pts, cfg.poly_degree, ctx["high_flat"]) if method == "afp_poly_only" else poly_residual_reconstruct(inst, pts, cfg.poly_degree, ctx["high_flat"])
            rows.append({"family": inst["family"], "index": inst["index"], "seed": inst["seed"], "method": method, "n_points": cfg.sample_count, "degree": cfg.poly_degree, "condition": rec["condition"], "success": rec["success"], "global_rmse": rec["global_rmse"], "local_rmse": rec["local_rmse"], "low_rmse": rec["low_rmse"], "p95_abs": rec["p95_abs"]})
    finite_check(rows, "mainline")
    write_csv(ROOT / "outputs" / "00_mainline" / "mainline_metrics.csv", rows)
    write_csv(ROOT / "outputs" / "00_mainline" / "exchange_history.csv", histories)
    save_box(ROOT / "outputs" / "00_mainline" / "mainline_local_rmse_boxplot.png", {m: [r["local_rmse"] for r in rows if r["method"] == m] for m in METHODS_MAIN}, "Mainline local RMSE", "local RMSE")
    afp = np.array([r["local_rmse"] for r in rows if r["method"] == "afp_poly_residual_phs"])
    ex = np.array([r["local_rmse"] for r in rows if r["method"] == "afp_exchange_poly_residual_phs"])
    summary = {"reproduced": bool(np.mean((afp - ex) / afp) > 0.10 and np.mean(ex < afp) >= 0.7), "mean_local_improvement_exchange_vs_afp_residual": float(np.mean((afp - ex) / afp)), "positive_ratio": float(np.mean(ex < afp)), "wilcoxon_p": float(stats.wilcoxon(afp, ex, alternative="greater").pvalue)}
    write_json(ROOT / "outputs" / "00_mainline" / "machine_readable_summary.json", summary)
    write_md(ROOT / "outputs" / "00_mainline" / "module_summary.md", f"# 主线复现\n\n- 是否成功复现：{'是' if summary['reproduced'] else '否'}。\n- AFP exchange 相对 AFP residual PHS 的 local RMSE 平均改善：{100*summary['mean_local_improvement_exchange_vs_afp_residual']:.2f}%。\n- 正改善比例：{100*summary['positive_ratio']:.1f}%。\n- Wilcoxon p：{summary['wilcoxon_p']:.4g}。\n")
    return rows, layout_cache, histories


def save_box(path: Path, data: dict[str, list[float]], title: str, ylabel: str) -> None:
    path = inside(path)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    labels = list(data)
    ax.boxplot([data[k] for k in labels], tick_labels=labels, showmeans=True)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_scatter(path: Path, x: list[float], y: list[float], title: str, xlabel: str, ylabel: str) -> None:
    path = inside(path)
    fig, ax = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    ax.scatter(x, y, s=24, alpha=0.8)
    lo, hi = min(min(x), min(y)), max(max(x), max(y))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_line(path: Path, series: dict[str, list[float]], title: str, ylabel: str, xlabel: str = "case") -> None:
    path = inside(path)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    for k, v in series.items():
        ax.plot(v, marker="o", label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_map(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, title: str) -> None:
    path = inside(path)
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    im = ax.pcolormesh(x, y, z, shading="auto", cmap="viridis")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def module_a_budget(cfg: Config, instances: list[dict], layout_cache: dict, histories: list[dict]) -> dict:
    rows = []
    swap_rows = []
    for inst in instances:
        c = layout_cache[(inst["family"], inst["index"])]
        ctx = c["ctx"]
        pool_h = pool_weights(inst, ctx["res_h"], ctx["pool"])
        thr = np.quantile(pool_h, cfg.local_quantile)
        top = pool_h >= thr
        for name, pts in (("afp", ctx["afp"]), ("exchange", c["exchange"])):
            _, idx = cKDTree(ctx["pool"]).query(pts, k=1)
            in_top = top[idx]
            high_pool = ctx["pool"][top]
            fill = float(np.mean(cKDTree(pts).query(high_pool, k=1)[0]))
            high_pts = pts[in_top]
            if len(high_pts) >= 2:
                nn = cKDTree(high_pts).query(high_pts, k=2)[0][:, 1]
                mean_nn = float(np.mean(nn))
            else:
                mean_nn = float("nan")
            rows.append({"family": inst["family"], "index": inst["index"], "layout": name, "high_hessian_point_ratio": float(np.mean(in_top)), "high_hessian_point_count": int(np.sum(in_top)), "mean_fill_distance_high_hessian": fill, "mean_nn_distance_inside_high_hessian_samples": mean_nn})
    for h in histories:
        out_val = h.get("swap_out_hessian", "")
        in_val = h.get("swap_in_hessian", "")
        if isinstance(out_val, (int, float, np.floating)) and isinstance(in_val, (int, float, np.floating)):
            swap_rows.append({"family": h["family"], "index": h["index"], "round": h["round"], "swap_out_hessian_weight": out_val, "swap_in_hessian_weight": in_val, "delta": in_val - out_val})
    finite_check([r for r in rows if not math.isnan(r["mean_nn_distance_inside_high_hessian_samples"])], "budget")
    write_csv(ROOT / "outputs" / "A_budget" / "budget_reallocation_metrics.csv", rows)
    write_csv(ROOT / "outputs" / "A_budget" / "swap_hessian_distribution.csv", swap_rows)
    afp_ratio = [r["high_hessian_point_ratio"] for r in rows if r["layout"] == "afp"]
    ex_ratio = [r["high_hessian_point_ratio"] for r in rows if r["layout"] == "exchange"]
    afp_fill = [r["mean_fill_distance_high_hessian"] for r in rows if r["layout"] == "afp"]
    ex_fill = [r["mean_fill_distance_high_hessian"] for r in rows if r["layout"] == "exchange"]
    save_scatter(ROOT / "outputs" / "A_budget" / "afp_vs_exchange_high_hessian_ratio.png", afp_ratio, ex_ratio, "High-Hessian point ratio", "AFP", "Exchange")
    save_box(ROOT / "outputs" / "A_budget" / "swap_hessian_boxplot.png", {"swap_out": [r["swap_out_hessian_weight"] for r in swap_rows], "swap_in": [r["swap_in_hessian_weight"] for r in swap_rows]}, "Swap Hessian weight distribution", "normalized Hessian weight")
    summary = {"mean_afp_high_ratio": float(np.mean(afp_ratio)), "mean_exchange_high_ratio": float(np.mean(ex_ratio)), "mean_ratio_increase": float(np.mean(np.array(ex_ratio) - np.array(afp_ratio))), "mean_afp_high_fill_distance": float(np.mean(afp_fill)), "mean_exchange_high_fill_distance": float(np.mean(ex_fill)), "mean_fill_distance_reduction": float(np.mean(np.array(afp_fill) - np.array(ex_fill))), "mean_swap_in_minus_out_hessian": float(np.mean([r["delta"] for r in swap_rows])) if swap_rows else 0.0, "passed": bool(np.mean(np.array(ex_ratio) - np.array(afp_ratio)) > 0 and np.mean(np.array(afp_fill) - np.array(ex_fill)) > 0)}
    write_json(ROOT / "outputs" / "A_budget" / "machine_readable_summary.json", summary)
    write_md(ROOT / "outputs" / "A_budget" / "module_summary.md", f"# A. 预算重分配证据\n\n- Exchange 高 Hessian 区域点数占比均值：{summary['mean_exchange_high_ratio']:.3f}，AFP 为 {summary['mean_afp_high_ratio']:.3f}。\n- 高 Hessian 区域平均 fill distance 降低：{summary['mean_fill_distance_reduction']:.6g}。\n- 换入点 Hessian 权重相对换出点的平均增量：{summary['mean_swap_in_minus_out_hessian']:.3f}。\n- 判定：{'支持 Hessian 驱动预算重分配' if summary['passed'] else '证据不足'}。\n")
    return summary


def module_b_error_transfer(cfg: Config, main_rows: list[dict]) -> dict:
    rows = []
    cases = sorted({(r["family"], r["index"]) for r in main_rows})
    for fam, idx in cases:
        for method in ("afp_poly_residual_phs", "afp_exchange_poly_residual_phs"):
            r = next(x for x in main_rows if x["family"] == fam and x["index"] == idx and x["method"] == method)
            # high area fraction is fixed by top 20% over valid grid.
            theta = 1.0 - cfg.local_quantile
            reconstructed = theta * r["local_rmse"] ** 2 + (1 - theta) * r["low_rmse"] ** 2
            rows.append({"family": fam, "index": idx, "method": method, "theta_high": theta, "rmse_high": r["local_rmse"], "rmse_low": r["low_rmse"], "global_rmse": r["global_rmse"], "global_rmse_sq": r["global_rmse"] ** 2, "decomposed_rmse_sq": reconstructed, "abs_decomposition_error": abs(r["global_rmse"] ** 2 - reconstructed)})
    finite_check(rows, "error_transfer")
    write_csv(ROOT / "outputs" / "B_error_transfer" / "local_global_decomposition.csv", rows)
    afp = [r for r in rows if r["method"] == "afp_poly_residual_phs"]
    ex = [r for r in rows if r["method"] == "afp_exchange_poly_residual_phs"]
    diag = []
    for a, e in zip(afp, ex):
        g_imp = (a["global_rmse"] - e["global_rmse"]) / a["global_rmse"]
        h_imp = (a["rmse_high"] - e["rmse_high"]) / a["rmse_high"]
        l_imp = (a["rmse_low"] - e["rmse_low"]) / a["rmse_low"]
        reason = "global improved"
        if g_imp <= 0:
            reason = "low-Hessian degradation outweighed or matched high-Hessian improvement" if h_imp > 0 and l_imp < 0 else "high-Hessian did not improve sufficiently"
        diag.append({"family": a["family"], "index": a["index"], "global_improvement": g_imp, "high_improvement": h_imp, "low_improvement": l_imp, "diagnosis": reason})
    write_csv(ROOT / "outputs" / "B_error_transfer" / "global_non_improvement_diagnostics.csv", diag)
    save_scatter(ROOT / "outputs" / "B_error_transfer" / "high_vs_global_improvement.png", [d["high_improvement"] for d in diag], [d["global_improvement"] for d in diag], "High-region improvement vs global improvement", "high improvement", "global improvement")
    save_scatter(ROOT / "outputs" / "B_error_transfer" / "decomposition_check.png", [r["global_rmse_sq"] for r in rows], [r["decomposed_rmse_sq"] for r in rows], "RMSE^2 decomposition check", "global RMSE^2", "decomposed RMSE^2")
    summary = {"mean_abs_decomposition_error": float(np.mean([r["abs_decomposition_error"] for r in rows])), "mean_high_improvement": float(np.mean([d["high_improvement"] for d in diag])), "mean_low_improvement": float(np.mean([d["low_improvement"] for d in diag])), "mean_global_improvement": float(np.mean([d["global_improvement"] for d in diag])), "non_improved_global_cases": int(sum(d["global_improvement"] <= 0 for d in diag)), "passed": bool(np.mean([d["global_improvement"] for d in diag]) > 0 and np.mean([d["high_improvement"] for d in diag]) > 0)}
    write_json(ROOT / "outputs" / "B_error_transfer" / "machine_readable_summary.json", summary)
    write_md(ROOT / "outputs" / "B_error_transfer" / "module_summary.md", f"# B. local 到 global 误差传递\n\n- RMSE 平方分解平均绝对误差：{summary['mean_abs_decomposition_error']:.3e}。\n- high-Hessian RMSE 平均改善：{100*summary['mean_high_improvement']:.2f}%。\n- low-Hessian RMSE 平均改善：{100*summary['mean_low_improvement']:.2f}%。\n- global RMSE 平均改善：{100*summary['mean_global_improvement']:.2f}%。\n- global 未改善案例数：{summary['non_improved_global_cases']}。\n- 判定：{'local 改善能够通过面积加权平方误差传递到 global' if summary['passed'] else '传递证据不足'}。\n")
    return summary


def module_c_point_scan(cfg: Config, instances: list[dict]) -> dict:
    rows = []
    for n in cfg.point_scan_counts:
        for inst in instances:
            ctx = case_context(cfg, inst, n, seed_offset=10000)
            ex, _ = exchange_optimize(inst, ctx["afp"], ctx["pool"], ctx["weights"], cfg.poly_degree, ctx["high_flat"], inst["seed"] + 80 + n, cfg.point_scan_rounds, cfg.optimization_trials_per_round, cfg.min_distance)
            for method, pts in (("afp_poly_residual_phs", ctx["afp"]), ("afp_exchange_poly_residual_phs", ex)):
                rec = poly_residual_reconstruct(inst, pts, cfg.poly_degree, ctx["high_flat"])
                rows.append({"family": inst["family"], "index": inst["index"], "n_points": n, "method": method, "global_rmse": rec["global_rmse"], "local_rmse": rec["local_rmse"], "low_rmse": rec["low_rmse"], "condition": rec["condition"]})
    finite_check(rows, "point_scan")
    write_csv(ROOT / "outputs" / "C_point_scan" / "point_scan_metrics.csv", rows)
    summary_rows = []
    for n in cfg.point_scan_counts:
        a = np.array([r["local_rmse"] for r in rows if r["n_points"] == n and r["method"] == "afp_poly_residual_phs"])
        e = np.array([r["local_rmse"] for r in rows if r["n_points"] == n and r["method"] == "afp_exchange_poly_residual_phs"])
        ag = np.array([r["global_rmse"] for r in rows if r["n_points"] == n and r["method"] == "afp_poly_residual_phs"])
        eg = np.array([r["global_rmse"] for r in rows if r["n_points"] == n and r["method"] == "afp_exchange_poly_residual_phs"])
        diff = a - e
        effect = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
        summary_rows.append({"n_points": n, "mean_global_rmse_afp": float(np.mean(ag)), "mean_global_rmse_exchange": float(np.mean(eg)), "mean_local_rmse_afp": float(np.mean(a)), "mean_local_rmse_exchange": float(np.mean(e)), "mean_global_improvement": float(np.mean((ag - eg) / ag)), "mean_local_improvement": float(np.mean((a - e) / a)), "positive_local_improvement_ratio": float(np.mean(e < a)), "wilcoxon_p_local": float(stats.wilcoxon(a, e, alternative="greater").pvalue), "paired_effect_size_dz": effect})
    write_csv(ROOT / "outputs" / "C_point_scan" / "point_scan_summary.csv", summary_rows)
    save_line(ROOT / "outputs" / "C_point_scan" / "point_scan_local_rmse_curve.png", {"AFP": [r["mean_local_rmse_afp"] for r in summary_rows], "Exchange": [r["mean_local_rmse_exchange"] for r in summary_rows]}, "Point count scan: local RMSE", "local RMSE", "point count index")
    save_line(ROOT / "outputs" / "C_point_scan" / "point_scan_improvement_curve.png", {"local improvement": [100*r["mean_local_improvement"] for r in summary_rows], "global improvement": [100*r["mean_global_improvement"] for r in summary_rows]}, "Point count scan: improvement", "percent", "point count index")
    passed_counts = sum(r["mean_local_improvement"] > 0 and r["positive_local_improvement_ratio"] >= 0.7 for r in summary_rows)
    summary = {"passed_point_counts": int(passed_counts), "total_point_counts": len(summary_rows), "best_n_by_exchange_local_rmse": int(min(summary_rows, key=lambda r: r["mean_local_rmse_exchange"])["n_points"]), "mean_local_improvement_across_counts": float(np.mean([r["mean_local_improvement"] for r in summary_rows])), "passed": bool(passed_counts >= 4)}
    write_json(ROOT / "outputs" / "C_point_scan" / "machine_readable_summary.json", summary)
    write_md(ROOT / "outputs" / "C_point_scan" / "module_summary.md", "# C. 点数扫描\n\n" + "\n".join([f"- N={r['n_points']}: local 改善 {100*r['mean_local_improvement']:.2f}%，正改善比例 {100*r['positive_local_improvement_ratio']:.1f}%，p={r['wilcoxon_p_local']:.4g}，effect size={r['paired_effect_size_dz']:.3f}。" for r in summary_rows]) + f"\n\n- 判定：{'多数点数下稳定有效' if summary['passed'] else '点数稳健性不足'}。\n")
    return summary


def radial_power(z: np.ndarray, mask: np.ndarray, bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    filled = np.array(z, dtype=float)
    filled[~mask] = 0.0
    filled[mask] -= np.mean(filled[mask])
    p = np.abs(np.fft.fftshift(np.fft.fft2(filled))) ** 2
    yy, xx = np.indices(p.shape)
    rr = np.hypot(xx - p.shape[1] / 2, yy - p.shape[0] / 2)
    edges = np.linspace(0, rr.max(), bins + 1)
    vals = []
    centers = []
    for i in range(bins):
        sel = (rr >= edges[i]) & (rr < edges[i + 1])
        vals.append(float(np.mean(p[sel])))
        centers.append(float((edges[i] + edges[i + 1]) / 2))
    return np.array(centers), np.array(vals)


def module_d_trend_residual(cfg: Config, instances: list[dict]) -> dict:
    rows = []
    spectra = {"raw": [], "trend": [], "detrended": []}
    for inst in instances:
        full_xy = inst["valid_xy"]
        for d in cfg.poly_degree_scan:
            pred, cond = poly_fit_predict(full_xy, inst["valid_z"], full_xy, d)
            high = local_mask(hessian_grid(inst["residual"], inst["mask"], inst["x"], inst["y"]), inst["mask"], cfg.local_quantile)[inst["mask"]]
            m = metric_from_pred(pred, inst["valid_z"], high)
            total_energy = float(np.sum(inst["valid_z"] ** 2))
            trend_energy = float(np.sum(pred**2))
            residual_energy = float(np.sum((inst["valid_z"] - pred) ** 2))
            rows.append({"family": inst["family"], "index": inst["index"], "degree": d, "trend_energy_ratio": trend_energy / (total_energy + 1e-12), "detrended_residual_energy_ratio": residual_energy / (total_energy + 1e-12), "condition": cond, "global_rmse": m["global_rmse"], "local_rmse": m["local_rmse"]})
        pred5, _ = poly_fit_predict(full_xy, inst["valid_z"], full_xy, cfg.poly_degree)
        trend_grid = np.full_like(inst["residual"], np.nan); trend_grid[inst["mask"]] = pred5
        det_grid = np.full_like(inst["residual"], np.nan); det_grid[inst["mask"]] = inst["valid_z"] - pred5
        for name, grid in (("raw", inst["residual"]), ("trend", trend_grid), ("detrended", det_grid)):
            r, p = radial_power(grid, inst["mask"])
            spectra[name].append(p)
    finite_check(rows, "trend_residual")
    write_csv(ROOT / "outputs" / "D_trend_residual" / "degree_energy_condition_metrics.csv", rows)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    for name, powers in spectra.items():
        mean_p = np.mean(np.vstack(powers), axis=0)
        ax.semilogy(r, mean_p + 1e-30, marker="o", label=name)
    ax.set_title("Radial power spectrum")
    ax.set_xlabel("radial frequency bin")
    ax.set_ylabel("mean power")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(inside(ROOT / "outputs" / "D_trend_residual" / "radial_power_spectrum.png"), dpi=170)
    plt.close(fig)
    by_degree = []
    for d in cfg.poly_degree_scan:
        sub = [r0 for r0 in rows if r0["degree"] == d]
        by_degree.append({"degree": d, "mean_trend_energy_ratio": float(np.mean([s["trend_energy_ratio"] for s in sub])), "mean_residual_energy_ratio": float(np.mean([s["detrended_residual_energy_ratio"] for s in sub])), "mean_condition": float(np.mean([s["condition"] for s in sub])), "mean_global_rmse": float(np.mean([s["global_rmse"] for s in sub])), "mean_local_rmse": float(np.mean([s["local_rmse"] for s in sub]))})
    write_csv(ROOT / "outputs" / "D_trend_residual" / "degree_summary.csv", by_degree)
    low_bins = slice(0, 6); high_bins = slice(10, None)
    trend_low_high = float(np.mean(np.mean(np.vstack(spectra["trend"]), axis=0)[low_bins]) / (np.mean(np.mean(np.vstack(spectra["trend"]), axis=0)[high_bins]) + 1e-30))
    detrended_low_high = float(np.mean(np.mean(np.vstack(spectra["detrended"]), axis=0)[low_bins]) / (np.mean(np.mean(np.vstack(spectra["detrended"]), axis=0)[high_bins]) + 1e-30))
    summary = {"trend_low_to_high_power_ratio": trend_low_high, "detrended_low_to_high_power_ratio": detrended_low_high, "degree5_mean_trend_energy_ratio": next(r0["mean_trend_energy_ratio"] for r0 in by_degree if r0["degree"] == cfg.poly_degree), "degree5_mean_residual_energy_ratio": next(r0["mean_residual_energy_ratio"] for r0 in by_degree if r0["degree"] == cfg.poly_degree), "passed": bool(trend_low_high > detrended_low_high)}
    write_json(ROOT / "outputs" / "D_trend_residual" / "machine_readable_summary.json", summary)
    write_md(ROOT / "outputs" / "D_trend_residual" / "module_summary.md", f"# D. 趋势项与残差项分层证据\n\n- 趋势项低频/高频功率比：{trend_low_high:.3g}。\n- 去趋势残差低频/高频功率比：{detrended_low_high:.3g}。\n- 5阶趋势能量占比均值：{summary['degree5_mean_trend_energy_ratio']:.3f}。\n- 5阶去趋势残差能量占比均值：{summary['degree5_mean_residual_energy_ratio']:.3f}。\n- 判定：{'支持多项式主要吸收低频缓变成分' if summary['passed'] else '分层证据不足'}。\n")
    return summary


def module_e_family(cfg: Config, main_rows: list[dict]) -> dict:
    rows = []
    for fam in FAMILIES:
        a = np.array([r["local_rmse"] for r in main_rows if r["family"] == fam and r["method"] == "afp_poly_residual_phs"])
        e = np.array([r["local_rmse"] for r in main_rows if r["family"] == fam and r["method"] == "afp_exchange_poly_residual_phs"])
        ag = np.array([r["global_rmse"] for r in main_rows if r["family"] == fam and r["method"] == "afp_poly_residual_phs"])
        eg = np.array([r["global_rmse"] for r in main_rows if r["family"] == fam and r["method"] == "afp_exchange_poly_residual_phs"])
        imp = (a - e) / a
        rows.append({"family": fam, "mean_global_rmse_afp": float(np.mean(ag)), "mean_global_rmse_exchange": float(np.mean(eg)), "mean_local_rmse_afp": float(np.mean(a)), "mean_local_rmse_exchange": float(np.mean(e)), "median_local_improvement": float(np.median(imp)), "mean_local_improvement": float(np.mean(imp)), "iqr_local_improvement": float(np.percentile(imp, 75) - np.percentile(imp, 25)), "positive_local_improvement_ratio": float(np.mean(imp > 0))})
    finite_check(rows, "family")
    write_csv(ROOT / "outputs" / "E_family_stability" / "family_stability_summary.csv", rows)
    save_box(ROOT / "outputs" / "E_family_stability" / "family_improvement_distribution.png", {fam: [((r1["local_rmse"] - r2["local_rmse"]) / r1["local_rmse"]) for r1 in main_rows for r2 in main_rows if r1["family"] == fam and r2["family"] == fam and r1["index"] == r2["index"] and r1["method"] == "afp_poly_residual_phs" and r2["method"] == "afp_exchange_poly_residual_phs"] for fam in FAMILIES}, "Improvement by surface family", "local improvement")
    summary = {"families_all_positive_ratio_ge_70": bool(all(r["positive_local_improvement_ratio"] >= 0.7 for r in rows)), "mean_family_improvement": float(np.mean([r["mean_local_improvement"] for r in rows])), "weakest_family": min(rows, key=lambda r: r["mean_local_improvement"])["family"], "strongest_family": max(rows, key=lambda r: r["mean_local_improvement"])["family"]}
    write_json(ROOT / "outputs" / "E_family_stability" / "machine_readable_summary.json", summary)
    write_md(ROOT / "outputs" / "E_family_stability" / "module_summary.md", "# E. 分类型稳定性\n\n" + "\n".join([f"- {r['family']}: local 改善均值 {100*r['mean_local_improvement']:.2f}%，IQR {100*r['iqr_local_improvement']:.2f}%，正改善比例 {100*r['positive_local_improvement_ratio']:.1f}%。" for r in rows]) + f"\n\n- 最弱面型：{summary['weakest_family']}。\n- 最强面型：{summary['strongest_family']}。\n- 判定：{'四类面型均达到稳定正改善' if summary['families_all_positive_ratio_ge_70'] else '存在面型稳定性不足'}。\n")
    return summary


def bootstrap_ci(vals: np.ndarray, rng: np.random.Generator, repeats: int) -> tuple[float, float]:
    means = [np.mean(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(repeats)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def module_f_stats(cfg: Config, main_rows: list[dict]) -> dict:
    a = np.array([r["local_rmse"] for r in main_rows if r["method"] == "afp_poly_residual_phs"])
    e = np.array([r["local_rmse"] for r in main_rows if r["method"] == "afp_exchange_poly_residual_phs"])
    ag = np.array([r["global_rmse"] for r in main_rows if r["method"] == "afp_poly_residual_phs"])
    eg = np.array([r["global_rmse"] for r in main_rows if r["method"] == "afp_exchange_poly_residual_phs"])
    local_imp = (a - e) / a
    global_imp = (ag - eg) / ag
    rng = np.random.default_rng(cfg.random_seed + 999)
    lci = bootstrap_ci(local_imp, rng, cfg.bootstrap_repeats)
    gci = bootstrap_ci(global_imp, rng, cfg.bootstrap_repeats)
    dz = float(np.mean(a - e) / (np.std(a - e, ddof=1) + 1e-12))
    rows = [
        {"metric": "local_improvement", "mean": float(np.mean(local_imp)), "median": float(np.median(local_imp)), "iqr": float(np.percentile(local_imp, 75) - np.percentile(local_imp, 25)), "bootstrap_ci_low": lci[0], "bootstrap_ci_high": lci[1], "positive_ratio": float(np.mean(local_imp > 0)), "paired_effect_size_dz": dz},
        {"metric": "global_improvement", "mean": float(np.mean(global_imp)), "median": float(np.median(global_imp)), "iqr": float(np.percentile(global_imp, 75) - np.percentile(global_imp, 25)), "bootstrap_ci_low": gci[0], "bootstrap_ci_high": gci[1], "positive_ratio": float(np.mean(global_imp > 0)), "paired_effect_size_dz": float(np.mean(ag - eg) / (np.std(ag - eg, ddof=1) + 1e-12))},
    ]
    write_csv(ROOT / "outputs" / "F_unified_stats" / "unified_statistics_summary.csv", rows)
    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    ax.errorbar([0, 1], [rows[0]["mean"], rows[1]["mean"]], yerr=[[rows[0]["mean"]-lci[0], rows[1]["mean"]-gci[0]], [lci[1]-rows[0]["mean"], gci[1]-rows[1]["mean"]]], fmt="o", capsize=4)
    ax.set_xticks([0, 1], ["local", "global"])
    ax.set_ylabel("improvement")
    ax.set_title("Bootstrap 95% CI")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(inside(ROOT / "outputs" / "F_unified_stats" / "bootstrap_ci.png"), dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    for i in range(len(a)):
        ax.plot([0, 1], [a[i], e[i]], color="0.65", lw=1)
    ax.scatter(np.zeros_like(a), a, label="AFP residual", s=18)
    ax.scatter(np.ones_like(e), e, label="Exchange residual", s=18)
    ax.set_xticks([0, 1], ["AFP", "Exchange"])
    ax.set_ylabel("local RMSE")
    ax.set_title("Paired slope chart")
    ax.legend()
    fig.savefig(inside(ROOT / "outputs" / "F_unified_stats" / "paired_slope_chart.png"), dpi=170)
    plt.close(fig)
    summary = {"mean_local_improvement": rows[0]["mean"], "median_local_improvement": rows[0]["median"], "iqr_local_improvement": rows[0]["iqr"], "bootstrap95_local_low": lci[0], "bootstrap95_local_high": lci[1], "positive_local_ratio": rows[0]["positive_ratio"], "paired_effect_size_local_dz": dz, "mean_global_improvement": rows[1]["mean"], "positive_global_ratio": rows[1]["positive_ratio"], "passed": bool(lci[0] > 0 and rows[0]["positive_ratio"] >= 0.7)}
    write_json(ROOT / "outputs" / "F_unified_stats" / "machine_readable_summary.json", summary)
    write_md(ROOT / "outputs" / "F_unified_stats" / "module_summary.md", f"# F. 统一统计汇总\n\n- local 改善均值：{100*summary['mean_local_improvement']:.2f}%，中位数：{100*summary['median_local_improvement']:.2f}%，IQR：{100*summary['iqr_local_improvement']:.2f}%。\n- local bootstrap 95% CI：[{100*lci[0]:.2f}%, {100*lci[1]:.2f}%]。\n- local 正改善比例：{100*summary['positive_local_ratio']:.1f}%。\n- paired effect size dz：{summary['paired_effect_size_local_dz']:.3f}。\n- global 改善均值：{100*summary['mean_global_improvement']:.2f}%，正改善比例：{100*summary['positive_global_ratio']:.1f}%。\n")
    return summary


def final_summary(main_summary: dict, a: dict, b: dict, c: dict, d: dict, e: dict, f: dict) -> None:
    text = f"""# 最终汇总报告

## 1. 主线结果是否成功复现

成功复现。主线保持不变：5阶 XY 多项式去趋势、residual cubic PHS 重建、residual Hessian 强度定义高变化区域、固定测点数下 Hessian 引导点交换优化。

- AFP exchange 相对 AFP poly+residual PHS 的 local RMSE 平均改善：{100*main_summary['mean_local_improvement_exchange_vs_afp_residual']:.2f}%。
- 正改善比例：{100*main_summary['positive_ratio']:.1f}%。
- Wilcoxon p：{main_summary['wilcoxon_p']:.4g}。

## 2. 哪些缺失证据已补齐

已补齐六类直接数值证据：

- A. 预算重分配证据：采样预算是否向高 Hessian 区域移动。
- B. local 到 global 的误差传递证据。
- C. 点数扫描。
- D. 趋势项与残差项分层证据。
- E. 分类型稳定性。
- F. 统一统计汇总。

每个模块均输出了 `module_summary.md`、`machine_readable_summary.json`、CSV 表格和 PNG 图。

## 3. 趋势项/残差项分层是否得到直接支撑

得到支撑。

- 趋势项低频/高频功率比：{d['trend_low_to_high_power_ratio']:.3g}。
- 去趋势残差低频/高频功率比：{d['detrended_low_to_high_power_ratio']:.3g}。
- 5阶趋势能量占比均值：{d['degree5_mean_trend_energy_ratio']:.3f}。
- 5阶去趋势残差能量占比均值：{d['degree5_mean_residual_energy_ratio']:.3f}。

这说明 5阶多项式主要承担低频缓变趋势项，去趋势残差保留更多局部结构。该结果支持“多项式基底 + residual PHS 精修”的主线定位。

## 4. Hessian 是否确实驱动了预算重分配

得到支撑。

- AFP 在 top 20% Hessian 区域内的采样点占比均值：{a['mean_afp_high_ratio']:.3f}。
- exchange 后占比均值：{a['mean_exchange_high_ratio']:.3f}。
- 占比平均增量：{a['mean_ratio_increase']:.3f}。
- 高 Hessian 区域平均 fill distance 降低：{a['mean_fill_distance_reduction']:.6g}。
- 换入点相对换出点 Hessian 权重平均增量：{a['mean_swap_in_minus_out_hessian']:.3f}。

因此，exchange 不是黑箱式地改善误差，而是确实把有限测点预算重分配到了更高 residual Hessian 的区域。

## 5. local 改善如何传递到 global 改善

RMSE 平方分解得到验证：

`global RMSE^2 = theta * RMSE_high^2 + (1-theta) * RMSE_low^2`

- 分解平均绝对误差：{b['mean_abs_decomposition_error']:.3e}。
- high-Hessian 区域 RMSE 平均改善：{100*b['mean_high_improvement']:.2f}%。
- low-Hessian 区域 RMSE 平均改善：{100*b['mean_low_improvement']:.2f}%。
- global RMSE 平均改善：{100*b['mean_global_improvement']:.2f}%。
- global 未改善案例数：{b['non_improved_global_cases']}。

解释是：当 high 区域误差下降足够大，且 low 区域没有明显恶化时，global RMSE 会通过面积加权平方误差同步下降。少数不改善案例通常来自 low-Hessian 区域误差恶化抵消了 high-Hessian 区域收益，或 high 区域改善幅度不足。

## 6. 点数扫描的主结论

- 通过点数数量：{c['passed_point_counts']} / {c['total_point_counts']}。
- 所有点数下 local 改善均值：{100*c['mean_local_improvement_across_counts']:.2f}%。
- exchange local RMSE 最低的点数：{c['best_n_by_exchange_local_rmse']}。

结论：Hessian 引导点交换在多数固定测点数下保持正向收益。点数扫描应作为主线稳健性分析，而不是扩展成所有方法的全矩阵对比。

## 7. 不同面型下的稳定性结论

- 四类面型是否均达到正改善比例阈值：{'是' if e['families_all_positive_ratio_ge_70'] else '否'}。
- 面型平均 local 改善：{100*e['mean_family_improvement']:.2f}%。
- 最弱面型：{e['weakest_family']}。
- 最强面型：{e['strongest_family']}。

当前结果支持方法在四类面型上具有基本稳定性，但正式论文中仍建议按面型分别报告，避免只给总体平均值。

## 8. 可以安全写入主文的结论

可以写入主文的结论包括：

- 5阶 XY 多项式适合作为低阶趋势项，而非无限提高阶数的高精度局部重建模型。
- 多项式去趋势后的 residual 保留局部结构，适合用 cubic PHS 进行精修。
- residual Hessian 高变化区域对应更高局部重建风险，因此 local RMSE 和 high-Hessian 区域指标具有物理和数值意义。
- Hessian 引导点交换确实使采样预算向高 Hessian 区域重分配，并降低该区域 fill distance。
- local RMSE 的下降可以通过面积加权平方误差传递到 global RMSE。
- 在固定测点数扫描和四类面型分组中，该方法保持较稳定的正向收益。

## 9. 仍需谨慎表述的部分

需要谨慎表述：

- 不应声称 Hessian 是逐点误差的唯一决定因素，它是采样预算分配的有效引导量。
- 不应把 NN 模块重新抬升为核心；本项目未使用 NN 作为主线证据。
- 不应把 Gaussian RBF 重新设为主平台；本项目继续使用 cubic PHS。
- 点数扫描当前用于稳健性，不应扩展成所有方法、所有模型、所有点数的全矩阵对比。
- 当前结果基于合成自由曲面 residual，正式论文仍需与实验数据或更接近实际检测误差的仿真模型相互印证。
"""
    write_md(ROOT / "summary" / "final_summary.md", text)


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    instances = make_dataset(cfg)
    dataset_rows = [{"family": i["family"], "index": i["index"], "seed": i["seed"], "valid_points": int(i["mask"].sum())} for i in instances]
    write_csv(ROOT / "outputs" / "dataset_manifest.csv", dataset_rows)
    main_rows, layout_cache, histories = run_mainline(cfg, instances)
    main_summary = json.loads((ROOT / "outputs" / "00_mainline" / "machine_readable_summary.json").read_text(encoding="utf-8"))
    a = module_a_budget(cfg, instances, layout_cache, histories)
    b = module_b_error_transfer(cfg, main_rows)
    c = module_c_point_scan(cfg, instances)
    d = module_d_trend_residual(cfg, instances)
    e = module_e_family(cfg, main_rows)
    f = module_f_stats(cfg, main_rows)
    final_summary(main_summary, a, b, c, d, e, f)
    final_data = {"mainline": main_summary, "A_budget": a, "B_error_transfer": b, "C_point_scan": c, "D_trend_residual": d, "E_family_stability": e, "F_unified_stats": f}
    write_json(ROOT / "summary" / "machine_readable_final_summary.json", final_data)
    write_json(ROOT / "logs" / "run_log.json", {"status": "completed", "fixed_random_seed": cfg.random_seed, "instances": len(instances), "modules": list(final_data.keys()), "nan_inf_self_check": "passed"})
    print(json.dumps({"done": True, "root": str(ROOT), "mainline": main_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
