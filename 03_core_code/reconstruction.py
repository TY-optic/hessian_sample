from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator


def valid_points(instance: dict) -> tuple[np.ndarray, np.ndarray]:
    mask = instance["mask"]
    xy = np.column_stack([instance["x"][mask], instance["y"][mask]])
    z = instance["residual_sag"][mask]
    return xy, z


def hessian_strength_grid(z: np.ndarray, mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    filled = np.array(z, dtype=float)
    mean = np.nanmean(filled[mask])
    filled[~mask] = mean
    dy = float(np.nanmedian(np.diff(y[:, 0])))
    dx = float(np.nanmedian(np.diff(x[0, :])))
    zy, zx = np.gradient(filled, dy, dx)
    zyy, zyx = np.gradient(zy, dy, dx)
    zxy, zxx = np.gradient(zx, dy, dx)
    h = np.sqrt(zxx**2 + zyy**2 + 2.0 * ((zxy + zyx) * 0.5) ** 2)
    h[~mask] = np.nan
    return h


def local_mask_from_hessian(h: np.ndarray, mask: np.ndarray, quantile: float) -> np.ndarray:
    threshold = np.nanquantile(h[mask], quantile)
    return mask & (h >= threshold)


def poly_terms(xy: np.ndarray, degree: int) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    cols = []
    for total in range(degree + 1):
        for px in range(total + 1):
            py = total - px
            cols.append((x**px) * (y**py))
    return np.column_stack(cols)


def reconstruct_poly(sample_xy: np.ndarray, sample_z: np.ndarray, eval_xy: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    a = poly_terms(sample_xy, degree)
    cond = float(np.linalg.cond(a))
    coef, *_ = np.linalg.lstsq(a, sample_z, rcond=None)
    return poly_terms(eval_xy, degree) @ coef, cond


def reconstruct_rbf(sample_xy: np.ndarray, sample_z: np.ndarray, eval_xy: np.ndarray, kind: str = "phs") -> tuple[np.ndarray, float, bool]:
    try:
        if kind == "phs":
            model = RBFInterpolator(sample_xy, sample_z, kernel="cubic", degree=1, smoothing=1e-12)
        elif kind == "gaussian":
            scale = np.median(np.linalg.norm(sample_xy[:, None, :] - sample_xy[None, :, :], axis=2))
            scale = max(float(scale), 1e-3)
            model = RBFInterpolator(sample_xy, sample_z, kernel="gaussian", epsilon=scale, smoothing=1e-12)
        else:
            raise ValueError(kind)
        pred = model(eval_xy)
        return pred, np.nan, True
    except Exception:
        return np.full(eval_xy.shape[0], np.nan), np.nan, False


def metrics(pred: np.ndarray, truth: np.ndarray, local: np.ndarray) -> dict:
    err = pred - truth
    good = np.isfinite(err)
    local_good = good & local
    return {
        "global_rmse": float(np.sqrt(np.mean(err[good] ** 2))),
        "local_rmse": float(np.sqrt(np.mean(err[local_good] ** 2))),
        "p95_abs": float(np.percentile(np.abs(err[good]), 95)),
    }


def sample_values(instance: dict, xy: np.ndarray) -> np.ndarray:
    all_xy, all_z = valid_points(instance)
    model = RBFInterpolator(all_xy, all_z, kernel="cubic", degree=1, smoothing=1e-14)
    return model(xy)

