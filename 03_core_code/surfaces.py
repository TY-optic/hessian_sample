from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from boundary import assert_inside_root
from config import StudyConfig, SURFACE_FAMILIES


def rounded_rectangle_mask(x: np.ndarray, y: np.ndarray, half_w: float = 1.0, half_h: float = 0.72, radius: float = 0.18) -> np.ndarray:
    qx = np.abs(x) - (half_w - radius)
    qy = np.abs(y) - (half_h - radius)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside_core = np.minimum(np.maximum(qx, qy), 0.0)
    signed = outside + inside_core - radius
    return signed <= 0


def make_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-1.05, 1.05, n)
    ys = np.linspace(-0.80, 0.80, n)
    x, y = np.meshgrid(xs, ys)
    mask = rounded_rectangle_mask(x, y)
    return x, y, mask


def parent_sag(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = 0.08 * (x**2 + 0.6 * y**2)
    astig = rng.uniform(-0.018, 0.018) * (x**2 - y**2)
    coma = rng.uniform(-0.012, 0.012) * (x**3 - 0.4 * x * y**2)
    return base + astig + coma


def residual_shape(family: str, x: np.ndarray, y: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    z = np.zeros_like(x)
    if family == "smooth_low_variation":
        for _ in range(5):
            ax = rng.integers(1, 4)
            ay = rng.integers(1, 4)
            amp = rng.uniform(-0.0025, 0.0025)
            z += amp * np.cos(ax * np.pi * (x + 1.0) / 2.1) * np.sin(ay * np.pi * (y + 0.8) / 1.6)
        z += rng.uniform(-0.002, 0.002) * (x**4 - y**4)
    elif family == "edge_rolloff":
        edge = np.maximum(np.abs(x) / 1.0, np.abs(y) / 0.72)
        edge = np.clip((edge - 0.68) / 0.32, 0.0, 1.0)
        z += rng.uniform(0.008, 0.018) * edge**3
        z *= rng.choice([-1.0, 1.0])
        z += rng.uniform(-0.002, 0.002) * x * y
    elif family == "local_bump":
        for _ in range(rng.integers(2, 5)):
            cx = rng.uniform(-0.65, 0.65)
            cy = rng.uniform(-0.45, 0.45)
            sx = rng.uniform(0.07, 0.16)
            sy = rng.uniform(0.06, 0.14)
            amp = rng.uniform(-0.016, 0.020)
            z += amp * np.exp(-((x - cx) ** 2 / (2 * sx**2) + (y - cy) ** 2 / (2 * sy**2)))
    elif family == "mid_frequency":
        for _ in range(7):
            kx = rng.integers(3, 8)
            ky = rng.integers(2, 7)
            phase = rng.uniform(0.0, 2 * np.pi)
            amp = rng.uniform(-0.0035, 0.0035)
            z += amp * np.sin(kx * x + ky * y + phase)
        z += rng.uniform(-0.006, 0.006) * np.exp(-((x + 0.35) ** 2 + (y - 0.22) ** 2) / 0.05)
    else:
        raise ValueError(f"Unknown surface family: {family}")
    z = z - np.nanmean(z[mask])
    z[~mask] = np.nan
    return z


def fit_bfs_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = np.column_stack([np.ones(mask.sum()), x[mask], y[mask], x[mask] ** 2 + y[mask] ** 2])
    coef, *_ = np.linalg.lstsq(a, z[mask], rcond=None)
    bfs = coef[0] + coef[1] * x + coef[2] * y + coef[3] * (x**2 + y**2)
    bfs[~mask] = np.nan
    return bfs


def generate_instance(config: StudyConfig, family: str, index: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x, y, mask = make_grid(config.grid_size)
    parent = parent_sag(x, y, rng)
    residual = residual_shape(family, x, y, mask, rng)
    raw = parent + np.nan_to_num(residual)
    bfs = fit_bfs_plane(x, y, raw, mask)
    bfs_residual = raw - bfs
    bfs_residual[~mask] = np.nan
    return {
        "family": family,
        "index": index,
        "seed": seed,
        "x": x,
        "y": y,
        "mask": mask,
        "parent_sag": parent,
        "bfs": bfs,
        "residual_sag": bfs_residual,
        "aperture": {"type": "rounded_rectangle", "half_w": 1.0, "half_h": 0.72, "corner_radius": 0.18},
    }


def save_instance(instance: dict, out_dir: Path, root: Path) -> Path:
    out_dir = assert_inside_root(out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{instance['family']}_{instance['index']:02d}_seed_{instance['seed']}"
    path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        path,
        x=instance["x"],
        y=instance["y"],
        mask=instance["mask"],
        parent_sag=instance["parent_sag"],
        bfs=instance["bfs"],
        residual_sag=instance["residual_sag"],
        family=instance["family"],
        index=instance["index"],
        seed=instance["seed"],
        aperture=json.dumps(instance["aperture"]),
    )
    return path


def generate_dataset(config: StudyConfig, out_dir: Path) -> list[dict]:
    instances = []
    seed0 = config.random_seed
    for f_id, family in enumerate(SURFACE_FAMILIES):
        for i in range(config.instances_per_family):
            seed = seed0 + 1000 * f_id + i
            inst = generate_instance(config, family, i, seed)
            save_instance(inst, out_dir, config.root)
            instances.append(inst)
    return instances

