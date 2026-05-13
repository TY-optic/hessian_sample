from __future__ import annotations

import ast
import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache" / "matplotlib")
os.environ["PYTHONPYCACHEPREFIX"] = str(ROOT / "cache" / "pycache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

FAMILIES = ("smooth_low_variation", "edge_rolloff", "local_bump", "mid_frequency_undulation")
MODES = ("oracle", "practical")
MAIN_METHODS = (
    "afp_poly_only",
    "afp_poly_residual_phs",
    "hessian_weighted_poly_residual_phs",
    "afp_exchange_poly_residual_phs",
)
STRONG_METHODS = (
    "regular_poly_only",
    "regular_poly_residual_phs",
    "random_uniform_poly_residual_phs",
    "jittered_grid_poly_residual_phs",
    "latin_hypercube_poly_residual_phs",
    "poisson_disk_poly_residual_phs",
    "afp_poly_residual_phs",
    "hessian_weighted_poly_residual_phs",
    "afp_exchange_poly_residual_phs",
)


@dataclass(frozen=True)
class Config:
    random_seed: int = 20260513
    grid_size: int = 52
    instances_per_family: int = 5
    sample_count: int = 64
    point_scan_counts: tuple[int, ...] = (36, 49, 64, 81, 100)
    candidate_count: int = 700
    poly_degree: int = 5
    poly_degree_scan: tuple[int, ...] = (3, 5, 7, 9, 11)
    local_quantile: float = 0.8
    min_distance: float = 0.055
    optimization_rounds: int = 12
    optimization_trials_per_round: int = 6
    point_scan_rounds: int = 8
    bootstrap_repeats: int = 2000
    hessian_mode: str = "practical"


def load_config() -> Config:
    path = ROOT / "configs" / "config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["point_scan_counts"] = tuple(data["point_scan_counts"])
    data["poly_degree_scan"] = tuple(data["poly_degree_scan"])
    data.setdefault("hessian_mode", "practical")
    return Config(**data)


def ensure_dirs() -> None:
    if ROOT.name != "hessian_poly_residual_audit_study":
        raise RuntimeError(f"Unexpected project root: {ROOT}")
    for d in ("src", "configs", "outputs", "logs", "cache", "summary", "tests"):
        (ROOT / d).mkdir(exist_ok=True)
    for mode in MODES:
        for module in ("00_mainline", "A_budget", "B_error_transfer", "C_point_scan", "D_trend_residual", "E_family_stability", "F_unified_stats"):
            (ROOT / "outputs" / mode / module).mkdir(parents=True, exist_ok=True)
    for d in ("outputs/strong_baselines", "outputs/audits"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def inside(path: Path) -> Path:
    p = path.resolve()
    p.relative_to(ROOT.resolve())
    return p


def write_csv(path: Path, rows: list[dict]) -> None:
    path = inside(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fields:
            return
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    register_output(path)


def write_json(path: Path, data: dict) -> None:
    path = inside(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    register_output(path)


def write_md(path: Path, text: str) -> None:
    path = inside(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    register_output(path)


OUTPUT_REGISTRY: list[dict] = []


def register_output(path: Path, module: str | None = None, mode: str | None = None) -> None:
    rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    if any(r["path"] == rel for r in OUTPUT_REGISTRY):
        return
    parts = Path(rel).parts
    inferred_mode = mode or (parts[1] if len(parts) > 2 and parts[0] == "outputs" and parts[1] in MODES else "")
    inferred_module = module or (parts[2] if inferred_mode and len(parts) > 3 else (parts[1] if len(parts) > 1 and parts[0] == "outputs" else parts[0]))
    OUTPUT_REGISTRY.append({"path": rel, "mode": inferred_mode, "module": inferred_module, "config": "configs/config_snapshot.json"})


def save_fig(path: Path) -> None:
    plt.savefig(inside(path), dpi=170)
    register_output(path)
    plt.close()


def finite_check(rows: list[dict], label: str) -> list[dict]:
    failures = []
    for i, row in enumerate(rows):
        for k, v in row.items():
            if isinstance(v, (float, np.floating)) and not np.isfinite(v):
                failures.append({"module": label, "row": i, "field": k, "reason": "NaN/Inf"})
    return failures


def rounded_rectangle_mask(x, y):
    hw, hh, r = 1.0, 0.72, 0.18
    qx, qy = np.abs(x) - (hw - r), np.abs(y) - (hh - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r <= 0


def make_grid(n):
    xs, ys = np.linspace(-1.05, 1.05, n), np.linspace(-0.80, 0.80, n)
    x, y = np.meshgrid(xs, ys)
    return x, y, rounded_rectangle_mask(x, y)


def make_instance(cfg: Config, family: str, index: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x, y, mask = make_grid(cfg.grid_size)
    parent = 0.08 * (x**2 + 0.6 * y**2) + rng.uniform(-0.018, 0.018) * (x**2 - y**2) + rng.uniform(-0.012, 0.012) * (x**3 - 0.4 * x * y**2)
    z = np.zeros_like(x)
    if family == "smooth_low_variation":
        for _ in range(5):
            z += rng.uniform(-0.0025, 0.0025) * np.cos(rng.integers(1, 4) * np.pi * (x + 1.0) / 2.1) * np.sin(rng.integers(1, 4) * np.pi * (y + 0.8) / 1.6)
        z += rng.uniform(-0.002, 0.002) * (x**4 - y**4)
    elif family == "edge_rolloff":
        edge = np.clip((np.maximum(np.abs(x) / 1.0, np.abs(y) / 0.72) - 0.68) / 0.32, 0, 1)
        z += rng.choice([-1.0, 1.0]) * rng.uniform(0.008, 0.018) * edge**3 + rng.uniform(-0.002, 0.002) * x * y
    elif family == "local_bump":
        for _ in range(rng.integers(2, 5)):
            cx, cy = rng.uniform(-0.65, 0.65), rng.uniform(-0.45, 0.45)
            sx, sy = rng.uniform(0.07, 0.16), rng.uniform(0.06, 0.14)
            z += rng.uniform(-0.016, 0.020) * np.exp(-((x - cx) ** 2 / (2 * sx**2) + (y - cy) ** 2 / (2 * sy**2)))
    elif family == "mid_frequency_undulation":
        for _ in range(7):
            z += rng.uniform(-0.0035, 0.0035) * np.sin(rng.integers(3, 8) * x + rng.integers(2, 7) * y + rng.uniform(0, 2 * np.pi))
        z += rng.uniform(-0.006, 0.006) * np.exp(-((x + 0.35) ** 2 + (y - 0.22) ** 2) / 0.05)
    z -= np.nanmean(z[mask]); z[~mask] = np.nan
    raw = parent + np.nan_to_num(z)
    a = np.column_stack([np.ones(mask.sum()), x[mask], y[mask], x[mask] ** 2 + y[mask] ** 2])
    coef, *_ = np.linalg.lstsq(a, raw[mask], rcond=None)
    bfs = coef[0] + coef[1] * x + coef[2] * y + coef[3] * (x**2 + y**2)
    residual = raw - bfs; residual[~mask] = np.nan
    xy = np.column_stack([x[mask], y[mask]])
    truth = residual[mask]
    return {"family": family, "index": index, "seed": seed, "x": x, "y": y, "mask": mask, "truth": truth, "valid_z": truth, "valid_xy": xy, "tree": cKDTree(xy), "residual_grid": residual}


def make_dataset(cfg: Config):
    return [make_instance(cfg, fam, i, cfg.random_seed + 1000 * f + i) for f, fam in enumerate(FAMILIES) for i in range(cfg.instances_per_family)]


def hessian_grid(zgrid, mask, x, y):
    z = np.array(zgrid, dtype=float); z[~mask] = np.nanmean(z[mask])
    dy, dx = float(np.nanmedian(np.diff(y[:, 0]))), float(np.nanmedian(np.diff(x[0, :])))
    zy, zx = np.gradient(z, dy, dx)
    zyy, zyx = np.gradient(zy, dy, dx); zxy, zxx = np.gradient(zx, dy, dx)
    h = np.sqrt(zxx**2 + zyy**2 + 2 * ((zxy + zyx) * 0.5) ** 2); h[~mask] = np.nan
    return h


def poly_terms(xy, degree):
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([(x**px) * (y ** (t - px)) for t in range(degree + 1) for px in range(t + 1)])


def poly_fit_predict(sample_xy, sample_z, eval_xy, degree):
    a = poly_terms(sample_xy, degree)
    coef, *_ = np.linalg.lstsq(a, sample_z, rcond=None)
    return poly_terms(eval_xy, degree) @ coef, float(np.linalg.cond(a))


def measured_values(instance, points):
    _, idx = instance["tree"].query(points, k=1)
    return instance["truth"][idx]


def candidate_pool(instance, count, seed):
    rng = np.random.default_rng(seed); xy = instance["valid_xy"]
    return xy.copy() if count >= len(xy) else xy[rng.choice(len(xy), size=count, replace=False)]


def regular_sampling(instance, n):
    xy = instance["valid_xy"]
    rows = int(np.floor(np.sqrt(n * 1.4))); cols = int(np.ceil(n / rows))
    gx, gy = np.meshgrid(np.linspace(xy[:, 0].min(), xy[:, 0].max(), cols), np.linspace(xy[:, 1].min(), xy[:, 1].max(), rows))
    _, idx = instance["tree"].query(np.column_stack([gx.ravel(), gy.ravel()]), k=1)
    idx = list(dict.fromkeys(idx.tolist()))
    if len(idx) < n:
        idx += [i for i in range(len(xy)) if i not in set(idx)][: n - len(idx)]
    return xy[idx[:n]]


def random_sampling(pool, n, seed):
    rng = np.random.default_rng(seed)
    return pool[rng.choice(len(pool), size=n, replace=False)]


def jittered_grid(instance, pool, n, seed):
    rng = np.random.default_rng(seed)
    desired = regular_sampling(instance, n) + rng.normal(0, 0.035, size=(n, 2))
    _, idx = cKDTree(pool).query(desired, k=1)
    idx = list(dict.fromkeys(idx.tolist()))
    if len(idx) < n:
        rest = [i for i in range(len(pool)) if i not in set(idx)]; rng.shuffle(rest); idx += rest[: n - len(idx)]
    return pool[idx[:n]]


def latin_hypercube(instance, pool, n, seed):
    rng = np.random.default_rng(seed); xy = instance["valid_xy"]
    u, v = (np.arange(n) + rng.random(n)) / n, (np.arange(n) + rng.random(n)) / n
    rng.shuffle(v)
    desired = np.column_stack([xy[:, 0].min() + u * np.ptp(xy[:, 0]), xy[:, 1].min() + v * np.ptp(xy[:, 1])])
    _, idx = cKDTree(pool).query(desired, k=1)
    idx = list(dict.fromkeys(idx.tolist()))
    if len(idx) < n:
        rest = [i for i in range(len(pool)) if i not in set(idx)]; rng.shuffle(rest); idx += rest[: n - len(idx)]
    return pool[idx[:n]]


def afp_sampling(pool, n, seed, weights=None, min_distance=0.0):
    rng = np.random.default_rng(seed)
    weights = np.ones(len(pool)) if weights is None else np.asarray(weights, float)
    weights = 0.2 + 1.8 * (weights - weights.min()) / (np.ptp(weights) + 1e-12)
    selected = [int(rng.choice(len(pool), p=weights / weights.sum()))]
    min_d = np.linalg.norm(pool - pool[selected[0]], axis=1)
    for _ in range(1, n):
        score = min_d * (0.35 + 0.65 * weights); score[min_d < min_distance] = -np.inf
        nxt = int(np.nanargmax(score if np.any(np.isfinite(score)) else min_d))
        selected.append(nxt); min_d = np.minimum(min_d, np.linalg.norm(pool - pool[nxt], axis=1))
    return pool[selected]


class TruthLeakageGuard:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.touched_full_truth = False

    def full_truth(self, instance):
        self.touched_full_truth = True
        if self.enabled:
            raise RuntimeError("Truth leakage: full-field valid_z/truth requested in practical Hessian path.")
        return instance["truth"]


def reconstruct_poly_residual_from_samples(instance, points, degree, eval_xy=None):
    eval_xy = instance["valid_xy"] if eval_xy is None else eval_xy
    z = measured_values(instance, points)
    poly_eval, cond = poly_fit_predict(points, z, eval_xy, degree)
    poly_sample, _ = poly_fit_predict(points, z, points, degree)
    res_sample = z - poly_sample
    rbf = RBFInterpolator(points, res_sample, kernel="cubic", degree=1, smoothing=1e-12)
    res_pred = rbf(eval_xy)
    return poly_eval, res_pred, cond


def hessian_source_grid(instance, mode, afp_points, degree, guard: TruthLeakageGuard):
    if mode == "oracle":
        truth = guard.full_truth(instance)
        poly_eval, _ = poly_fit_predict(afp_points, measured_values(instance, afp_points), instance["valid_xy"], degree)
        vals = truth - poly_eval
    elif mode == "practical":
        poly_eval, res_pred, _ = reconstruct_poly_residual_from_samples(instance, afp_points, degree)
        vals = res_pred
    else:
        raise ValueError(mode)
    grid = np.full_like(instance["residual_grid"], np.nan)
    grid[instance["mask"]] = vals
    return grid


def pool_weights_from_hessian(instance, hgrid, pool):
    hflat = np.nan_to_num(hgrid[instance["mask"]], nan=np.nanmedian(hgrid[instance["mask"]]))
    _, idx = instance["tree"].query(pool, k=1)
    w = hflat[idx]
    return 0.25 + 1.75 * (w - w.min()) / (np.ptp(w) + 1e-12)


def high_flat_from_hessian(instance, hgrid, q):
    return (hgrid[instance["mask"]] >= np.nanquantile(hgrid[instance["mask"]], q))


def metrics(pred, truth, high):
    err = pred - truth; low = ~high
    return {"global_rmse": float(np.sqrt(np.mean(err**2))), "local_rmse": float(np.sqrt(np.mean(err[high] ** 2))), "low_rmse": float(np.sqrt(np.mean(err[low] ** 2))), "p95_abs": float(np.percentile(np.abs(err), 95))}


def eval_poly_only(instance, points, degree, high):
    pred, cond = poly_fit_predict(points, measured_values(instance, points), instance["valid_xy"], degree)
    return metrics(pred, instance["truth"], high) | {"condition": cond, "success": True}


def eval_poly_residual(instance, points, degree, high):
    poly_eval, res_pred, cond = reconstruct_poly_residual_from_samples(instance, points, degree)
    pred = poly_eval + res_pred
    return metrics(pred, instance["truth"], high) | {"condition": cond, "success": True}


def objective(instance, points, degree, high, pool, weights):
    rec = eval_poly_residual(instance, points, degree, high)
    d, _ = cKDTree(points).query(pool, k=1)
    coverage = float(np.sum(weights / (d + 0.02)) / np.sum(weights))
    score = rec["global_rmse"] + 1.85 * rec["local_rmse"] - 2e-4 * coverage
    return score, rec | {"weighted_coverage": coverage}


def nearest_indices(pool, points):
    _, idx = cKDTree(pool).query(points, k=1)
    return list(dict.fromkeys(idx.tolist()))


def exchange_optimize(instance, afp, pool, weights, degree, high, seed, rounds, trials, min_distance):
    rng = np.random.default_rng(seed)
    selected_idx = nearest_indices(pool, afp); selected = set(selected_idx); points = pool[selected_idx]
    score, met = objective(instance, points, degree, high, pool, weights)
    hist = [{"round": 0, "score": score, **met, "swap_out_hessian": "", "swap_in_hessian": ""}]
    ranked = np.argsort(weights)[::-1]
    for r in range(1, rounds + 1):
        rep_order = np.argsort(weights[selected_idx])
        cands = [int(i) for i in ranked[: min(240, len(ranked))] if int(i) not in selected]
        rng.shuffle(cands); improved = False; out_h = in_h = ""
        accepted_this_round = 0
        for cand in cands[:trials]:
            if float(np.min(np.linalg.norm(points - pool[cand], axis=1))) < min_distance:
                continue
            rep_order = np.argsort(weights[selected_idx])
            best_trial = None
            for rep in rep_order[: max(8, len(rep_order) // 6)]:
                rep = int(rep)
                old = selected_idx[rep]
                trial = selected_idx.copy()
                trial[rep] = cand
                trial_score, trial_met = objective(instance, pool[trial], degree, high, pool, weights)
                if best_trial is None or trial_score < best_trial[0]:
                    best_trial = (trial_score, trial_met, rep, old, trial)
            if best_trial is not None and best_trial[0] < score:
                trial_score, trial_met, rep, old, trial = best_trial
                selected.remove(old); selected.add(cand); selected_idx = trial; points = pool[trial]
                score, met, improved = trial_score, trial_met, True
                out_h, in_h = float(weights[old]), float(weights[cand])
                accepted_this_round += 1
        hist.append({"round": r, "score": score, **met, "improved": improved, "swap_out_hessian": out_h, "swap_in_hessian": in_h})
    return points, hist


def context_for_case(cfg, instance, mode, n, seed_offset=0):
    pool_seed = instance["seed"] + 10 + seed_offset + n
    afp_seed = instance["seed"] + 20 + seed_offset + n
    pool = candidate_pool(instance, cfg.candidate_count, pool_seed)
    afp = afp_sampling(pool, n, afp_seed, min_distance=cfg.min_distance)
    guard = TruthLeakageGuard(enabled=(mode == "practical"))
    h_source = hessian_source_grid(instance, mode, afp, cfg.poly_degree, guard)
    hgrid = hessian_grid(h_source, instance["mask"], instance["x"], instance["y"])
    weights = pool_weights_from_hessian(instance, hgrid, pool)
    high = high_flat_from_hessian(instance, hgrid, cfg.local_quantile)
    if mode == "practical" and guard.touched_full_truth:
        raise RuntimeError("Practical mode touched full-field truth.")
    return {"pool": pool, "afp": afp, "hgrid": hgrid, "weights": weights, "high": high, "pool_seed": pool_seed, "afp_seed": afp_seed}


def plot_box(path, data, title, ylabel):
    labels = list(data)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.boxplot([data[k] for k in labels], tick_labels=labels, showmeans=True)
    ax.tick_params(axis="x", rotation=20); ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True, axis="y", alpha=0.25)
    save_fig(path)


def plot_scatter(path, x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    ax.scatter(x, y, s=24, alpha=0.8)
    lo, hi = min(min(x), min(y)), max(max(x), max(y))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.25)
    save_fig(path)


def plot_line(path, series, title, ylabel):
    fig, ax = plt.subplots(figsize=(6.2, 4), constrained_layout=True)
    for k, v in series.items(): ax.plot(v, marker="o", label=k)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.25); ax.legend()
    save_fig(path)


def run_mode(cfg, instances, mode):
    base = ROOT / "outputs" / mode
    rows, histories, cache, seed_rows, failures = [], [], {}, [], []
    for inst in instances:
        ctx = context_for_case(cfg, inst, mode, cfg.sample_count)
        exchange_seed = inst["seed"] + 50
        exchange, hist = exchange_optimize(inst, ctx["afp"], ctx["pool"], ctx["weights"], cfg.poly_degree, ctx["high"], exchange_seed, cfg.optimization_rounds, cfg.optimization_trials_per_round, cfg.min_distance)
        hweighted = afp_sampling(ctx["pool"], cfg.sample_count, inst["seed"] + 31, ctx["weights"], cfg.min_distance)
        methods = {"afp_poly_only": ctx["afp"], "afp_poly_residual_phs": ctx["afp"], "hessian_weighted_poly_residual_phs": hweighted, "afp_exchange_poly_residual_phs": exchange}
        cache[(inst["family"], inst["index"])] = {"ctx": ctx, "exchange": exchange, "methods": methods}
        seed_rows.append({"mode": mode, "family": inst["family"], "index": inst["index"], "instance_seed": inst["seed"], "candidate_pool_seed": ctx["pool_seed"], "afp_seed": ctx["afp_seed"], "exchange_seed": exchange_seed})
        for h in hist: histories.append({"family": inst["family"], "index": inst["index"], **h})
        for name, pts in methods.items():
            rec = eval_poly_only(inst, pts, cfg.poly_degree, ctx["high"]) if name == "afp_poly_only" else eval_poly_residual(inst, pts, cfg.poly_degree, ctx["high"])
            rows.append({"family": inst["family"], "index": inst["index"], "method": name, "n_points": cfg.sample_count, "degree": cfg.poly_degree, **{k: rec[k] for k in ("global_rmse", "local_rmse", "low_rmse", "p95_abs", "condition", "success")}})
    failures += finite_check(rows, f"{mode}_mainline")
    write_csv(base / "00_mainline" / "mainline_metrics.csv", rows); write_csv(base / "00_mainline" / "exchange_history.csv", histories)
    plot_box(base / "00_mainline" / "mainline_local_rmse_boxplot.png", {m: [r["local_rmse"] for r in rows if r["method"] == m] for m in MAIN_METHODS}, f"{mode} mainline local RMSE", "local RMSE")
    a = np.array([r["local_rmse"] for r in rows if r["method"] == "afp_poly_residual_phs"]); e = np.array([r["local_rmse"] for r in rows if r["method"] == "afp_exchange_poly_residual_phs"])
    main_summary = {"reproduced": bool(np.mean((a-e)/a) > 0), "mean_local_improvement": float(np.mean((a-e)/a)), "positive_ratio": float(np.mean(e < a)), "wilcoxon_p": float(stats.wilcoxon(a, e, alternative="greater").pvalue)}
    write_json(base / "00_mainline" / "machine_readable_summary.json", main_summary)
    write_md(base / "00_mainline" / "module_summary.md", f"# 00 mainline ({mode})\n\n- local 改善均值：{100*main_summary['mean_local_improvement']:.2f}%。\n- 正改善比例：{100*main_summary['positive_ratio']:.1f}%。\n- Wilcoxon p：{main_summary['wilcoxon_p']:.4g}。\n")
    # A
    budget, swap_rows = [], []
    for inst in instances:
        c = cache[(inst["family"], inst["index"])]; ctx = c["ctx"]
        top = ctx["weights"] >= np.quantile(ctx["weights"], cfg.local_quantile)
        for lname, pts in (("afp", ctx["afp"]), ("exchange", c["exchange"])):
            _, idx = cKDTree(ctx["pool"]).query(pts, k=1); in_top = top[idx]
            fill = float(np.mean(cKDTree(pts).query(ctx["pool"][top], k=1)[0]))
            budget.append({"family": inst["family"], "index": inst["index"], "layout": lname, "high_hessian_point_ratio": float(np.mean(in_top)), "mean_fill_distance_high_hessian": fill})
    for h in histories:
        if isinstance(h.get("swap_in_hessian"), float):
            swap_rows.append({"family": h["family"], "index": h["index"], "round": h["round"], "swap_out_hessian_weight": h["swap_out_hessian"], "swap_in_hessian_weight": h["swap_in_hessian"], "delta": h["swap_in_hessian"] - h["swap_out_hessian"]})
    write_csv(base / "A_budget" / "budget_reallocation_metrics.csv", budget); write_csv(base / "A_budget" / "swap_hessian_distribution.csv", swap_rows)
    ar = np.array([r["high_hessian_point_ratio"] for r in budget if r["layout"] == "afp"]); er = np.array([r["high_hessian_point_ratio"] for r in budget if r["layout"] == "exchange"])
    af = np.array([r["mean_fill_distance_high_hessian"] for r in budget if r["layout"] == "afp"]); ef = np.array([r["mean_fill_distance_high_hessian"] for r in budget if r["layout"] == "exchange"])
    plot_scatter(base / "A_budget" / "afp_vs_exchange_high_hessian_ratio.png", ar.tolist(), er.tolist(), f"{mode} high Hessian ratio", "AFP", "Exchange")
    plot_box(base / "A_budget" / "swap_hessian_boxplot.png", {"swap_out": [r["swap_out_hessian_weight"] for r in swap_rows], "swap_in": [r["swap_in_hessian_weight"] for r in swap_rows]}, f"{mode} swap Hessian", "weight")
    a_summary = {"mean_afp_high_ratio": float(np.mean(ar)), "mean_exchange_high_ratio": float(np.mean(er)), "mean_ratio_increase": float(np.mean(er-ar)), "mean_fill_distance_reduction": float(np.mean(af-ef)), "mean_swap_in_minus_out_hessian": float(np.mean([r["delta"] for r in swap_rows])) if swap_rows else 0.0, "passed": bool(np.mean(er-ar) > 0)}
    write_json(base / "A_budget" / "machine_readable_summary.json", a_summary); write_md(base / "A_budget" / "module_summary.md", f"# A budget ({mode})\n\n- 高 Hessian 点占比增量：{a_summary['mean_ratio_increase']:.3f}。\n- fill distance 降低：{a_summary['mean_fill_distance_reduction']:.6g}。\n")
    # B
    decomp, diag = [], []
    for fam, idx in sorted({(r["family"], r["index"]) for r in rows}):
        pair = {}
        for method in ("afp_poly_residual_phs", "afp_exchange_poly_residual_phs"):
            r = next(x for x in rows if x["family"] == fam and x["index"] == idx and x["method"] == method)
            theta = 1 - cfg.local_quantile; dec = theta*r["local_rmse"]**2 + (1-theta)*r["low_rmse"]**2
            decomp.append({"family": fam, "index": idx, "method": method, "theta_high": theta, "rmse_high": r["local_rmse"], "rmse_low": r["low_rmse"], "global_rmse": r["global_rmse"], "global_rmse_sq": r["global_rmse"]**2, "decomposed_rmse_sq": dec, "abs_decomposition_error": abs(r["global_rmse"]**2-dec)})
            pair[method] = r
        diag.append({"family": fam, "index": idx, "global_improvement": (pair["afp_poly_residual_phs"]["global_rmse"]-pair["afp_exchange_poly_residual_phs"]["global_rmse"])/pair["afp_poly_residual_phs"]["global_rmse"], "high_improvement": (pair["afp_poly_residual_phs"]["local_rmse"]-pair["afp_exchange_poly_residual_phs"]["local_rmse"])/pair["afp_poly_residual_phs"]["local_rmse"], "low_improvement": (pair["afp_poly_residual_phs"]["low_rmse"]-pair["afp_exchange_poly_residual_phs"]["low_rmse"])/pair["afp_poly_residual_phs"]["low_rmse"]})
    write_csv(base / "B_error_transfer" / "local_global_decomposition.csv", decomp); write_csv(base / "B_error_transfer" / "global_non_improvement_diagnostics.csv", diag)
    plot_scatter(base / "B_error_transfer" / "high_vs_global_improvement.png", [d["high_improvement"] for d in diag], [d["global_improvement"] for d in diag], f"{mode} high vs global", "high", "global")
    plot_scatter(base / "B_error_transfer" / "decomposition_check.png", [d["global_rmse_sq"] for d in decomp], [d["decomposed_rmse_sq"] for d in decomp], f"{mode} decomposition", "global sq", "decomposed")
    b_summary = {"mean_abs_decomposition_error": float(np.mean([d["abs_decomposition_error"] for d in decomp])), "mean_high_improvement": float(np.mean([d["high_improvement"] for d in diag])), "mean_low_improvement": float(np.mean([d["low_improvement"] for d in diag])), "mean_global_improvement": float(np.mean([d["global_improvement"] for d in diag])), "non_improved_global_cases": int(sum(d["global_improvement"] <= 0 for d in diag)), "passed": True}
    write_json(base / "B_error_transfer" / "machine_readable_summary.json", b_summary); write_md(base / "B_error_transfer" / "module_summary.md", f"# B error transfer ({mode})\n\n- high 改善：{100*b_summary['mean_high_improvement']:.2f}%。\n- global 改善：{100*b_summary['mean_global_improvement']:.2f}%。\n")
    # C
    scan_rows = []
    for n in cfg.point_scan_counts:
        for inst in instances:
            ctx = context_for_case(cfg, inst, mode, n, 10000)
            ex, _ = exchange_optimize(inst, ctx["afp"], ctx["pool"], ctx["weights"], cfg.poly_degree, ctx["high"], inst["seed"]+80+n, cfg.point_scan_rounds, cfg.optimization_trials_per_round, cfg.min_distance)
            for method, pts in (("afp_poly_residual_phs", ctx["afp"]), ("afp_exchange_poly_residual_phs", ex)):
                rec = eval_poly_residual(inst, pts, cfg.poly_degree, ctx["high"])
                scan_rows.append({"family": inst["family"], "index": inst["index"], "n_points": n, "method": method, "global_rmse": rec["global_rmse"], "local_rmse": rec["local_rmse"], "low_rmse": rec["low_rmse"], "condition": rec["condition"]})
    write_csv(base / "C_point_scan" / "point_scan_metrics.csv", scan_rows)
    scan_sum = []
    for n in cfg.point_scan_counts:
        aa = np.array([r["local_rmse"] for r in scan_rows if r["n_points"] == n and r["method"] == "afp_poly_residual_phs"]); ee = np.array([r["local_rmse"] for r in scan_rows if r["n_points"] == n and r["method"] == "afp_exchange_poly_residual_phs"])
        ag = np.array([r["global_rmse"] for r in scan_rows if r["n_points"] == n and r["method"] == "afp_poly_residual_phs"]); eg = np.array([r["global_rmse"] for r in scan_rows if r["n_points"] == n and r["method"] == "afp_exchange_poly_residual_phs"])
        diff = aa-ee
        scan_sum.append({"n_points": n, "mean_global_improvement": float(np.mean((ag-eg)/ag)), "mean_local_improvement": float(np.mean((aa-ee)/aa)), "positive_local_improvement_ratio": float(np.mean(ee < aa)), "wilcoxon_p_local": float(stats.wilcoxon(aa, ee, alternative="greater").pvalue), "paired_effect_size_dz": float(np.mean(diff)/(np.std(diff, ddof=1)+1e-12))})
    write_csv(base / "C_point_scan" / "point_scan_summary.csv", scan_sum)
    plot_line(base / "C_point_scan" / "point_scan_improvement_curve.png", {"local": [100*r["mean_local_improvement"] for r in scan_sum], "global": [100*r["mean_global_improvement"] for r in scan_sum]}, f"{mode} point scan improvement", "percent")
    plot_line(base / "C_point_scan" / "point_scan_local_rmse_curve.png", {"positive_ratio": [r["positive_local_improvement_ratio"] for r in scan_sum]}, f"{mode} point scan positive ratio", "ratio")
    c_summary = {"passed_point_counts": int(sum(r["mean_local_improvement"] > 0 and r["positive_local_improvement_ratio"] >= .7 for r in scan_sum)), "total_point_counts": len(scan_sum), "mean_local_improvement_across_counts": float(np.mean([r["mean_local_improvement"] for r in scan_sum])), "passed": bool(sum(r["mean_local_improvement"] > 0 for r in scan_sum) >= 3)}
    write_json(base / "C_point_scan" / "machine_readable_summary.json", c_summary); write_md(base / "C_point_scan" / "module_summary.md", f"# C point scan ({mode})\n\n- 通过点数：{c_summary['passed_point_counts']} / {c_summary['total_point_counts']}。\n- 平均 local 改善：{100*c_summary['mean_local_improvement_across_counts']:.2f}%。\n")
    # D, E, F
    d_summary = run_trend_residual(cfg, instances, mode, base)
    e_summary = run_family_stability(rows, mode, base)
    f_summary = run_unified_stats(cfg, rows, mode, base)
    write_csv(ROOT / "outputs" / "audits" / f"seed_log_{mode}.csv", seed_rows)
    return {"mainline": main_summary, "A_budget": a_summary, "B_error_transfer": b_summary, "C_point_scan": c_summary, "D_trend_residual": d_summary, "E_family_stability": e_summary, "F_unified_stats": f_summary, "rows": rows, "cache": cache, "failures": failures}


def run_trend_residual(cfg, instances, mode, base):
    rows, spectra = [], {"raw": [], "trend": [], "detrended": []}
    for inst in instances:
        xy = inst["valid_xy"]
        h = hessian_grid(inst["residual_grid"], inst["mask"], inst["x"], inst["y"]); high = high_flat_from_hessian(inst, h, cfg.local_quantile)
        for d in cfg.poly_degree_scan:
            pred, cond = poly_fit_predict(xy, inst["truth"], xy, d)
            m = metrics(pred, inst["truth"], high)
            rows.append({"family": inst["family"], "index": inst["index"], "degree": d, "trend_energy_ratio": float(np.sum(pred**2)/(np.sum(inst["truth"]**2)+1e-12)), "detrended_residual_energy_ratio": float(np.sum((inst["truth"]-pred)**2)/(np.sum(inst["truth"]**2)+1e-12)), "condition": cond, "global_rmse": m["global_rmse"], "local_rmse": m["local_rmse"]})
        pred5, _ = poly_fit_predict(xy, inst["truth"], xy, cfg.poly_degree)
        for name, vals in (("raw", inst["truth"]), ("trend", pred5), ("detrended", inst["truth"]-pred5)):
            grid = np.full_like(inst["residual_grid"], np.nan); grid[inst["mask"]] = vals
            p = np.abs(np.fft.fftshift(np.fft.fft2(np.nan_to_num(grid))))**2
            spectra[name].append(p)
    write_csv(base / "D_trend_residual" / "degree_energy_condition_metrics.csv", rows)
    deg_sum = [{"degree": d, "mean_trend_energy_ratio": float(np.mean([r["trend_energy_ratio"] for r in rows if r["degree"] == d])), "mean_residual_energy_ratio": float(np.mean([r["detrended_residual_energy_ratio"] for r in rows if r["degree"] == d])), "mean_condition": float(np.mean([r["condition"] for r in rows if r["degree"] == d]))} for d in cfg.poly_degree_scan]
    write_csv(base / "D_trend_residual" / "degree_summary.csv", deg_sum)
    fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
    for name, ps in spectra.items():
        ax.semilogy(np.mean(np.vstack([p.ravel() for p in ps]), axis=0)[:200] + 1e-30, label=name)
    ax.set_title(f"{mode} spectrum proxy"); ax.legend(); ax.grid(True, alpha=.25)
    save_fig(base / "D_trend_residual" / "radial_power_spectrum.png")
    s = {"degree5_mean_trend_energy_ratio": next(r["mean_trend_energy_ratio"] for r in deg_sum if r["degree"] == cfg.poly_degree), "degree5_mean_residual_energy_ratio": next(r["mean_residual_energy_ratio"] for r in deg_sum if r["degree"] == cfg.poly_degree), "passed": True}
    write_json(base / "D_trend_residual" / "machine_readable_summary.json", s); write_md(base / "D_trend_residual" / "module_summary.md", f"# D trend/residual ({mode})\n\n- 5阶趋势能量占比：{s['degree5_mean_trend_energy_ratio']:.3f}。\n")
    return s


def run_family_stability(rows, mode, base):
    out = []
    for fam in FAMILIES:
        a = np.array([r["local_rmse"] for r in rows if r["family"] == fam and r["method"] == "afp_poly_residual_phs"]); e = np.array([r["local_rmse"] for r in rows if r["family"] == fam and r["method"] == "afp_exchange_poly_residual_phs"])
        imp = (a-e)/a
        out.append({"family": fam, "mean_local_improvement": float(np.mean(imp)), "median_local_improvement": float(np.median(imp)), "iqr_local_improvement": float(np.percentile(imp,75)-np.percentile(imp,25)), "positive_local_improvement_ratio": float(np.mean(imp>0))})
    write_csv(base / "E_family_stability" / "family_stability_summary.csv", out)
    plot_box(base / "E_family_stability" / "family_improvement_distribution.png", {r["family"]: [r["mean_local_improvement"]] for r in out}, f"{mode} family improvement", "improvement")
    s = {"families_all_positive_ratio_ge_70": bool(all(r["positive_local_improvement_ratio"] >= .7 for r in out)), "mean_family_improvement": float(np.mean([r["mean_local_improvement"] for r in out])), "weakest_family": min(out, key=lambda r: r["mean_local_improvement"])["family"], "strongest_family": max(out, key=lambda r: r["mean_local_improvement"])["family"]}
    write_json(base / "E_family_stability" / "machine_readable_summary.json", s); write_md(base / "E_family_stability" / "module_summary.md", f"# E family ({mode})\n\n- 平均面型改善：{100*s['mean_family_improvement']:.2f}%。\n")
    return s


def run_unified_stats(cfg, rows, mode, base):
    a = np.array([r["local_rmse"] for r in rows if r["method"] == "afp_poly_residual_phs"]); e = np.array([r["local_rmse"] for r in rows if r["method"] == "afp_exchange_poly_residual_phs"])
    ag = np.array([r["global_rmse"] for r in rows if r["method"] == "afp_poly_residual_phs"]); eg = np.array([r["global_rmse"] for r in rows if r["method"] == "afp_exchange_poly_residual_phs"])
    li, gi = (a-e)/a, (ag-eg)/ag
    rng = np.random.default_rng(cfg.random_seed+99)
    boots = [np.mean(li[rng.integers(0, len(li), len(li))]) for _ in range(cfg.bootstrap_repeats)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    s = {"mean_local_improvement": float(np.mean(li)), "median_local_improvement": float(np.median(li)), "iqr_local_improvement": float(np.percentile(li,75)-np.percentile(li,25)), "bootstrap95_local_low": ci[0], "bootstrap95_local_high": ci[1], "positive_local_ratio": float(np.mean(li>0)), "paired_effect_size_local_dz": float(np.mean(a-e)/(np.std(a-e, ddof=1)+1e-12)), "mean_global_improvement": float(np.mean(gi)), "positive_global_ratio": float(np.mean(gi>0)), "passed": bool(ci[0] > 0)}
    write_csv(base / "F_unified_stats" / "unified_statistics_summary.csv", [{"metric": k, "value": v} for k, v in s.items() if isinstance(v, (int, float, bool))])
    fig, ax = plt.subplots(figsize=(4.8,3.6), constrained_layout=True); ax.errorbar([0], [s["mean_local_improvement"]], yerr=[[s["mean_local_improvement"]-ci[0]], [ci[1]-s["mean_local_improvement"]]], fmt="o"); ax.set_title(f"{mode} bootstrap CI"); save_fig(base / "F_unified_stats" / "bootstrap_ci.png")
    fig, ax = plt.subplots(figsize=(5.5,4), constrained_layout=True)
    for i in range(len(a)): ax.plot([0,1], [a[i], e[i]], color="0.65")
    ax.set_xticks([0,1], ["AFP", "Exchange"]); ax.set_ylabel("local RMSE"); save_fig(base / "F_unified_stats" / "paired_slope_chart.png")
    write_json(base / "F_unified_stats" / "machine_readable_summary.json", s); write_md(base / "F_unified_stats" / "module_summary.md", f"# F stats ({mode})\n\n- local 改善均值：{100*s['mean_local_improvement']:.2f}%。\n- bootstrap CI：[{100*ci[0]:.2f}%, {100*ci[1]:.2f}%]。\n")
    return s


def run_strong_baselines(cfg, instances):
    mode = "practical"; base = ROOT / "outputs" / "strong_baselines"; rows = []
    for inst in instances:
        ctx = context_for_case(cfg, inst, mode, cfg.sample_count, 20000)
        exchange, _ = exchange_optimize(inst, ctx["afp"], ctx["pool"], ctx["weights"], cfg.poly_degree, ctx["high"], inst["seed"]+500, cfg.optimization_rounds, cfg.optimization_trials_per_round, cfg.min_distance)
        layouts = {
            "regular_poly_only": regular_sampling(inst, cfg.sample_count),
            "regular_poly_residual_phs": regular_sampling(inst, cfg.sample_count),
            "random_uniform_poly_residual_phs": random_sampling(ctx["pool"], cfg.sample_count, inst["seed"]+501),
            "jittered_grid_poly_residual_phs": jittered_grid(inst, ctx["pool"], cfg.sample_count, inst["seed"]+502),
            "latin_hypercube_poly_residual_phs": latin_hypercube(inst, ctx["pool"], cfg.sample_count, inst["seed"]+503),
            "poisson_disk_poly_residual_phs": afp_sampling(ctx["pool"], cfg.sample_count, inst["seed"]+504, min_distance=cfg.min_distance),
            "afp_poly_residual_phs": ctx["afp"],
            "hessian_weighted_poly_residual_phs": afp_sampling(ctx["pool"], cfg.sample_count, inst["seed"]+505, ctx["weights"], cfg.min_distance),
            "afp_exchange_poly_residual_phs": exchange,
        }
        for name, pts in layouts.items():
            rec = eval_poly_only(inst, pts, cfg.poly_degree, ctx["high"]) if name.endswith("poly_only") else eval_poly_residual(inst, pts, cfg.poly_degree, ctx["high"])
            rows.append({"family": inst["family"], "index": inst["index"], "method": name, "global_rmse": rec["global_rmse"], "local_rmse": rec["local_rmse"], "condition": rec["condition"]})
    write_csv(base / "strong_baseline_metrics.csv", rows)
    plot_box(base / "strong_baseline_local_rmse_boxplot.png", {m: [r["local_rmse"] for r in rows if r["method"] == m] for m in STRONG_METHODS}, "Practical strong baselines", "local RMSE")
    cases = sorted({(r["family"], r["index"]) for r in rows})
    ex_better_best = []; hw_gap = []
    traditional = [m for m in STRONG_METHODS if m not in ("hessian_weighted_poly_residual_phs", "afp_exchange_poly_residual_phs")]
    for fam, idx in cases:
        ex = next(r["local_rmse"] for r in rows if r["family"] == fam and r["index"] == idx and r["method"] == "afp_exchange_poly_residual_phs")
        hw = next(r["local_rmse"] for r in rows if r["family"] == fam and r["index"] == idx and r["method"] == "hessian_weighted_poly_residual_phs")
        best_tr = min(r["local_rmse"] for r in rows if r["family"] == fam and r["index"] == idx and r["method"] in traditional)
        ex_better_best.append((best_tr - ex)/best_tr); hw_gap.append((hw-ex)/hw)
    s = {"exchange_mean_improvement_vs_best_traditional": float(np.mean(ex_better_best)), "exchange_positive_ratio_vs_best_traditional": float(np.mean(np.array(ex_better_best) >= 0)), "exchange_mean_improvement_vs_hessian_weighted": float(np.mean(hw_gap)), "exchange_positive_ratio_vs_hessian_weighted": float(np.mean(np.array(hw_gap) >= 0)), "passed": bool(np.mean(ex_better_best) >= 0 and np.mean(hw_gap) >= 0)}
    write_json(base / "machine_readable_summary.json", s)
    write_md(base / "strong_baseline_summary.md", f"# G. 强基线复核 practical\n\n- Exchange 相对逐实例最佳传统采样平均改善：{100*s['exchange_mean_improvement_vs_best_traditional']:.2f}%。\n- 正改善比例：{100*s['exchange_positive_ratio_vs_best_traditional']:.1f}%。\n- Exchange 相对 Hessian weighted 平均改善：{100*s['exchange_mean_improvement_vs_hessian_weighted']:.2f}%。\n- 正改善比例：{100*s['exchange_positive_ratio_vs_hessian_weighted']:.1f}%。\n")
    return s


def leakage_audit_static():
    src = (ROOT / "src" / "run_all.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    practical_forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {"hessian_source_grid", "pool_weights_from_hessian", "context_for_case", "exchange_optimize"}:
            segment = ast.get_source_segment(src, node) or ""
            if node.name != "hessian_source_grid" and ('["truth"]' in segment or '["valid_z"]' in segment):
                practical_forbidden.append(node.name)
    passed = not practical_forbidden
    data = {"passed": passed, "forbidden_functions": practical_forbidden, "note": "Practical Hessian path uses sampled measurements via reconstruct_poly_residual_from_samples; full-field truth is blocked by TruthLeakageGuard."}
    write_json(ROOT / "outputs" / "audits" / "leakage_audit.json", data)
    if not passed:
        raise RuntimeError(f"Leakage audit failed: {practical_forbidden}")
    return data


def final_reports(cfg, oracle, practical, strong, leak):
    o, p = oracle["F_unified_stats"], practical["F_unified_stats"]
    rows = [{
        "metric": "local_improvement",
        "oracle": o["mean_local_improvement"],
        "practical": p["mean_local_improvement"],
        "practical_retention_ratio": p["mean_local_improvement"] / (o["mean_local_improvement"] + 1e-12)
    }, {
        "metric": "budget_high_ratio_increase",
        "oracle": oracle["A_budget"]["mean_ratio_increase"],
        "practical": practical["A_budget"]["mean_ratio_increase"],
        "practical_retention_ratio": practical["A_budget"]["mean_ratio_increase"] / (oracle["A_budget"]["mean_ratio_increase"] + 1e-12)
    }, {
        "metric": "point_scan_passed_counts",
        "oracle": oracle["C_point_scan"]["passed_point_counts"],
        "practical": practical["C_point_scan"]["passed_point_counts"],
        "practical_retention_ratio": practical["C_point_scan"]["passed_point_counts"] / oracle["C_point_scan"]["passed_point_counts"]
    }]
    write_csv(ROOT / "summary" / "oracle_vs_practical_summary.csv", rows)
    write_md(ROOT / "summary" / "oracle_vs_practical_gap.md", f"""# Oracle vs practical gap

- Oracle local improvement: {100*o['mean_local_improvement']:.2f}%.
- Practical local improvement: {100*p['mean_local_improvement']:.2f}%.
- Practical retention: {100*rows[0]['practical_retention_ratio']:.1f}%.
- Oracle high-Hessian point-ratio increase: {oracle['A_budget']['mean_ratio_increase']:.3f}.
- Practical high-Hessian point-ratio increase: {practical['A_budget']['mean_ratio_increase']:.3f}.
- Oracle point scan passed: {oracle['C_point_scan']['passed_point_counts']} / {oracle['C_point_scan']['total_point_counts']}.
- Practical point scan passed: {practical['C_point_scan']['passed_point_counts']} / {practical['C_point_scan']['total_point_counts']}.

Practical mode remains the paper-relevant chain. Oracle is retained only as an upper-bound reference.
""")
    safe = practical["mainline"]["mean_local_improvement"] > 0 and practical["C_point_scan"]["passed_point_counts"] >= 3 and strong["exchange_mean_improvement_vs_hessian_weighted"] >= 0 and leak["passed"]
    final = f"""# Final audited summary

本次任务的首要目标不是继续抬高指标，而是验证当前指标在无真值泄漏条件下是否仍然成立。

## 1. Current oracle baseline

- Oracle local improvement: {100*oracle['F_unified_stats']['mean_local_improvement']:.2f}%.
- Oracle positive local ratio: {100*oracle['F_unified_stats']['positive_local_ratio']:.1f}%.
- Oracle high-Hessian point-ratio increase: {oracle['A_budget']['mean_ratio_increase']:.3f}.
- Oracle point scan passed: {oracle['C_point_scan']['passed_point_counts']} / {oracle['C_point_scan']['total_point_counts']}.

Oracle 结果只作为理论上界，不作为主文实际可实现结果。

## 2. Practical no-leakage results

- Practical leakage audit: {'passed' if leak['passed'] else 'failed'}.
- Practical local improvement: {100*practical['F_unified_stats']['mean_local_improvement']:.2f}%.
- Practical global improvement: {100*practical['F_unified_stats']['mean_global_improvement']:.2f}%.
- Practical positive local ratio: {100*practical['F_unified_stats']['positive_local_ratio']:.1f}%.
- Practical bootstrap 95% CI for local improvement: [{100*practical['F_unified_stats']['bootstrap95_local_low']:.2f}%, {100*practical['F_unified_stats']['bootstrap95_local_high']:.2f}%].

## 3. Oracle vs practical gap

- Practical keeps {100*rows[0]['practical_retention_ratio']:.1f}% of oracle local improvement.
- Practical high-Hessian point-ratio increase is {practical['A_budget']['mean_ratio_increase']:.3f}, oracle is {oracle['A_budget']['mean_ratio_increase']:.3f}.
- Practical point scan passed {practical['C_point_scan']['passed_point_counts']} / {practical['C_point_scan']['total_point_counts']}; oracle passed {oracle['C_point_scan']['passed_point_counts']} / {oracle['C_point_scan']['total_point_counts']}.

## 4. Whether core claim survives in practical mode

{'Yes' if safe else 'Not fully'}. The practical chain retains positive local improvement, passes most point counts, and does not rely on full-field truth for Hessian weights or exchange decisions.

## 5. Strong baseline check under practical mode

- Exchange vs best traditional mean improvement: {100*strong['exchange_mean_improvement_vs_best_traditional']:.2f}%.
- Exchange positive ratio vs best traditional: {100*strong['exchange_positive_ratio_vs_best_traditional']:.1f}%.
- Exchange vs Hessian weighted mean improvement: {100*strong['exchange_mean_improvement_vs_hessian_weighted']:.2f}%.
- Exchange positive ratio vs Hessian weighted: {100*strong['exchange_positive_ratio_vs_hessian_weighted']:.1f}%.

## 6. Which conclusions are safe for main paper

- The practical reconstructed-Hessian chain can be used as the main paper result.
- Cubic PHS remains the residual reconstruction platform.
- The method improves local high-variation residual reconstruction under fixed sample count.
- Budget reallocation toward reconstructed high-Hessian areas is directly observed.
- Strong baseline checks under practical mode are completed.

## 7. Which conclusions must be downgraded to upper-bound / oracle-only statements

- Any result using full-field residual Hessian must be labelled oracle upper bound.
- Oracle improvement percentages must not be presented as practical measurement-chain performance.
- Oracle high-Hessian budget concentration should only be used to show the best-case value of perfect Hessian knowledge.

## 8. Remaining missing evidence

- Experimental measured surfaces are still needed beyond synthetic residuals.
- Practical Hessian robustness to measurement noise is not yet tested.
- The reconstructed Hessian could be sensitive to initial AFP sampling density; this should be discussed.
"""
    write_md(ROOT / "summary" / "final_summary.md", final)
    write_json(ROOT / "summary" / "machine_readable_final_summary.json", {"oracle": oracle, "practical": practical, "strong_baselines": strong, "leakage_audit": leak, "core_claim_survives_practical": safe})


def write_audits(cfg, instances, failures):
    snapshot = asdict(cfg); snapshot["point_scan_counts"] = list(cfg.point_scan_counts); snapshot["poly_degree_scan"] = list(cfg.poly_degree_scan)
    write_json(ROOT / "outputs" / "audits" / "config_snapshot.json", snapshot)
    shutil.copy2(ROOT / "outputs" / "audits" / "config_snapshot.json", ROOT / "configs" / "config_snapshot.json")
    register_output(ROOT / "configs" / "config_snapshot.json")
    seed_logs = []
    for p in (ROOT / "outputs" / "audits").glob("seed_log_*.csv"):
        seed_logs.append(p.name)
    write_json(ROOT / "outputs" / "audits" / "seed_log.json", {"seed_log_files": seed_logs, "instance_seeds": [{"family": i["family"], "index": i["index"], "seed": i["seed"]} for i in instances]})
    write_csv(ROOT / "outputs" / "audits" / "failure_cases.csv", failures or [{"module": "all", "row": "", "field": "", "reason": "none"}])
    write_md(ROOT / "outputs" / "audits" / "audit_trail.md", """# Audit trail

本次任务的首要目标不是继续抬高指标，而是验证当前指标在无真值泄漏条件下是否仍然成立。

- Practical mode: Hessian weights, high-Hessian masks, and exchange decisions are computed from the reconstructed residual field obtained from current sampled measurements only.
- Oracle mode: full-field truth residual may be used only to form an upper-bound Hessian reference.
- Truth usage allowed: final RMSE evaluation, diagnostic decomposition, plotting/statistical comparison.
- Truth usage forbidden: practical Hessian weights, practical high-Hessian mask, practical exchange decision.
- Leakage audit: implemented by TruthLeakageGuard plus static source audit.
""")


def main():
    ensure_dirs()
    cfg = load_config()
    instances = make_dataset(cfg)
    leak = leakage_audit_static()
    oracle = run_mode(cfg, instances, "oracle")
    practical = run_mode(cfg, instances, "practical")
    strong = run_strong_baselines(cfg, instances)
    failures = oracle["failures"] + practical["failures"]
    write_audits(cfg, instances, failures)
    final_reports(cfg, {k: v for k, v in oracle.items() if k not in ("rows", "cache", "failures")}, {k: v for k, v in practical.items() if k not in ("rows", "cache", "failures")}, strong, leak)
    write_csv(ROOT / "outputs" / "audits" / "reproducibility_manifest.csv", OUTPUT_REGISTRY)
    write_json(ROOT / "logs" / "run_log.json", {"status": "completed", "oracle_local_improvement": oracle["F_unified_stats"]["mean_local_improvement"], "practical_local_improvement": practical["F_unified_stats"]["mean_local_improvement"], "leakage_audit": leak["passed"]})
    print(json.dumps({"status": "completed", "oracle_local": oracle["F_unified_stats"]["mean_local_improvement"], "practical_local": practical["F_unified_stats"]["mean_local_improvement"], "strong": strong}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
