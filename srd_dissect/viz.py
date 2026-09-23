"""Figures: each target next to its arrangement, pieces coloured consistently across targets."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon as MplPolygon

from .geom import loop_order, sample_boundary
from .render import RenderConfig
from .state import Dissection


def piece_world_polygons(D: Dissection, t: int, n_per_prim: int = 16) -> dict[int, np.ndarray]:
    out = {}
    for p in D.pieces:
        P = sample_boundary(loop_order(p.shape), n_per_prim)
        out[p.pid] = p.poses[t].apply(P).numpy()
    return out


def plot_dissection(D: Dissection, targets: torch.Tensor, cfg: RenderConfig, path, title: str = "",
                    target_names: list[str] | None = None):
    T = targets.shape[0]
    cmap = plt.get_cmap("tab20")
    colors = {p.pid: cmap(i % 20) for i, p in enumerate(sorted(D.pieces, key=lambda q: q.pid))}
    lo, hi = cfg.lim
    fig, axes = plt.subplots(2, T, figsize=(4 * T, 8), squeeze=False)
    for t in range(T):
        ax = axes[0, t]
        ax.imshow(targets[t].numpy(), cmap="Greys", vmin=0, vmax=1, extent=(lo, hi, hi, lo))
        ax.set_title(target_names[t] if target_names else f"target {t}")
        ax = axes[1, t]
        ax.imshow(targets[t].numpy(), cmap="Greys", vmin=0, vmax=3, extent=(lo, hi, hi, lo))
        for pid, poly in piece_world_polygons(D, t).items():
            ax.add_patch(MplPolygon(poly, closed=True, facecolor=colors[pid], edgecolor="k", lw=0.6, alpha=0.85))
            c = poly.mean(0)
            ax.text(c[0], c[1], str(pid), fontsize=6, ha="center", va="center")
        ax.set_title(f"arrangement {t}")
        for a in axes[:, t]:
            a.set_xlim(lo, hi)
            a.set_ylim(hi, lo)
            a.set_aspect("equal")
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
