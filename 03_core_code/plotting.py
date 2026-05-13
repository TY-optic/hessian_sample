from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_map(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    im = ax.pcolormesh(x, y, z, shading="auto", cmap="viridis")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_points(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, points: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    im = ax.pcolormesh(x, y, z, shading="auto", cmap="coolwarm")
    ax.scatter(points[:, 0], points[:, 1], s=8, c="k", alpha=0.75)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_line(path: Path, series: dict[str, list[float]], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    for label, values in series.items():
        ax.plot(values, marker="o", label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_boxplot(path: Path, data: dict[str, list[float]], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    labels = list(data)
    ax.boxplot([data[k] for k in labels], labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_scatter(path: Path, x: list[float], y: list[float], title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.4), constrained_layout=True)
    ax.scatter(x, y, s=24, alpha=0.8)
    lo = min(min(x), min(y))
    hi = max(max(x), max(y))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)

