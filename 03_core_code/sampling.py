from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from reconstruction import reconstruct_rbf, sample_values, valid_points, metrics


def candidate_pool(instance: dict, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy, _ = valid_points(instance)
    if count >= len(xy):
        return xy.copy()
    idx = rng.choice(len(xy), size=count, replace=False)
    return xy[idx]


def regular_sampling(instance: dict, n: int) -> np.ndarray:
    mask = instance["mask"]
    xs = instance["x"][mask]
    ys = instance["y"][mask]
    rows = int(np.floor(np.sqrt(n * 1.4)))
    cols = int(np.ceil(n / rows))
    gx, gy = np.meshgrid(np.linspace(xs.min(), xs.max(), cols), np.linspace(ys.min(), ys.max(), rows))
    desired = np.column_stack([gx.ravel(), gy.ravel()])
    all_xy, _ = valid_points(instance)
    tree = cKDTree(all_xy)
    _, idx = tree.query(desired, k=1)
    idx = list(dict.fromkeys(idx.tolist()))
    if len(idx) < n:
        remain = [i for i in range(len(all_xy)) if i not in set(idx)]
        idx.extend(remain[: n - len(idx)])
    return all_xy[idx[:n]]


def afp_sampling(pool: np.ndarray, n: int, seed: int, weights: np.ndarray | None = None, min_distance: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if weights is None:
        weights = np.ones(len(pool))
    weights = np.asarray(weights, dtype=float)
    weights = weights / (np.nanmax(weights) + 1e-12)
    start = int(rng.choice(len(pool), p=weights / weights.sum()))
    selected = [start]
    min_d = np.linalg.norm(pool - pool[start], axis=1)
    for _ in range(1, n):
        score = min_d * (0.35 + 0.65 * weights)
        if min_distance > 0:
            score[min_d < min_distance] = -np.inf
        nxt = int(np.nanargmax(score))
        if not np.isfinite(score[nxt]):
            nxt = int(np.argmax(min_d))
        selected.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(pool - pool[nxt], axis=1))
    return pool[selected]


def weights_from_hessian(instance: dict, hessian: np.ndarray, pool: np.ndarray) -> np.ndarray:
    grid_xy, _ = valid_points(instance)
    h = hessian[instance["mask"]]
    h = np.nan_to_num(h, nan=np.nanmedian(h))
    tree = cKDTree(grid_xy)
    _, idx = tree.query(pool, k=1)
    w = h[idx]
    w = (w - np.min(w)) / (np.ptp(w) + 1e-12)
    return 0.25 + 1.75 * w


def nearest_indices(pool: np.ndarray, points: np.ndarray) -> list[int]:
    tree = cKDTree(pool)
    _, idx = tree.query(points, k=1)
    return list(dict.fromkeys(idx.tolist()))


def objective(instance: dict, points: np.ndarray, local_mask_flat: np.ndarray, weighted_pool: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[float, dict]:
    eval_xy, truth = valid_points(instance)
    z = sample_values(instance, points)
    pred, _, ok = reconstruct_rbf(points, z, eval_xy, "phs")
    if not ok:
        return np.inf, {"global_rmse": np.inf, "local_rmse": np.inf, "weighted_coverage": 0.0}
    m = metrics(pred, truth, local_mask_flat)
    coverage = 0.0
    if weighted_pool is not None:
        pool, weights = weighted_pool
        tree = cKDTree(points)
        d, _ = tree.query(pool, k=1)
        coverage = float(np.sum(weights / (d + 0.02)) / (np.sum(weights) + 1e-12))
    score = m["global_rmse"] + 1.2 * m["local_rmse"] - 2e-4 * coverage
    m["weighted_coverage"] = coverage
    return float(score), m


def exchange_optimize(instance: dict, initial_points: np.ndarray, pool: np.ndarray, weights: np.ndarray, local_mask_flat: np.ndarray, seed: int, rounds: int, trials_per_round: int, min_distance: float) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    selected_idx = nearest_indices(pool, initial_points)
    selected = set(selected_idx)
    points = pool[selected_idx].copy()
    score, met = objective(instance, points, local_mask_flat, (pool, weights))
    history = [{"round": 0, "score": score, **met}]
    replace_rank = np.argsort(weights[selected_idx])
    candidate_rank = np.argsort(weights)[::-1]
    for r in range(1, rounds + 1):
        improved = False
        trial_candidates = [int(i) for i in candidate_rank[: min(len(candidate_rank), 220)] if int(i) not in selected]
        rng.shuffle(trial_candidates)
        for cand in trial_candidates[:trials_per_round]:
            current_tree = cKDTree(points)
            dmin, _ = current_tree.query(pool[cand], k=1)
            if dmin < min_distance:
                continue
            rep_local = int(replace_rank[min(rng.integers(0, max(2, len(replace_rank) // 8)), len(replace_rank) - 1)])
            trial_idx = selected_idx.copy()
            old = trial_idx[rep_local]
            trial_idx[rep_local] = cand
            trial_points = pool[trial_idx]
            trial_score, trial_met = objective(instance, trial_points, local_mask_flat, (pool, weights))
            if trial_score < score:
                selected.remove(old)
                selected.add(cand)
                selected_idx = trial_idx
                points = trial_points
                score, met = trial_score, trial_met
                improved = True
                break
            if improved:
                break
        history.append({"round": r, "score": score, **met, "improved": improved})
        if not improved:
            replace_rank = np.argsort(weights[selected_idx])
    return points, history
