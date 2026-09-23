"""Evaluation protocol of Voronoi Scissors (Qi et al.), Sec. 6.1.

1. Normalize each input shape to an area of 10,000 px^2 (the same scale is applied to its arrangement).
2. Sample 100 points uniformly on each shape. ASSUMPTION (logged in DEVIATIONS.md): "on each shape" is read as
   on the boundary, evenly spaced by arc length over all boundary rings (so internal gaps/holes of an
   arrangement contribute boundary points and are penalized).
3. Align each arrangement to its input by ICP with translation only (rotation fixed).
4. Report Chamfer and Hausdorff per shape, and their averages over A and B.
   ASSUMPTION: Chamfer = mean of the two directed mean nearest-neighbour distances; Hausdorff = max of the two
   directed max nearest-neighbour distances (the paper does not spell out its convention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.spatial import cKDTree

NORMALIZED_AREA = 10_000.0
N_SAMPLES = 100


def boundary_rings(geom) -> list[np.ndarray]:
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    rings = []
    for p in polys:
        if p.is_empty:
            continue
        rings.append(np.asarray(p.exterior.coords))
        rings += [np.asarray(r.coords) for r in p.interiors]
    return rings


def sample_boundary_uniform(geom, n: int = N_SAMPLES, offset: float = 0.0) -> np.ndarray:
    """n points evenly spaced by arc length over all boundary rings (closed polylines, first == last)."""
    segs = []
    for R in boundary_rings(geom):
        segs.append(np.stack([R[:-1], R[1:]], 1))
    S = np.concatenate(segs)  # (m, 2, 2)
    L = np.linalg.norm(S[:, 1] - S[:, 0], axis=-1)
    cum = np.concatenate([[0], np.cumsum(L)])
    total = cum[-1]
    s = (np.arange(n) + offset) / n * total
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(L) - 1)
    u = (s - cum[i]) / np.maximum(L[i], 1e-12)
    return S[i, 0] + u[:, None] * (S[i, 1] - S[i, 0])


def icp_translation(src: np.ndarray, dst: np.ndarray, iters: int = 100, tol: float = 1e-9) -> np.ndarray:
    """Translation t minimizing sum ||src + t - nn_dst(src + t)||^2 (point-to-point ICP, translation only)."""
    tree = cKDTree(dst)
    t = dst.mean(0) - src.mean(0)
    for _ in range(iters):
        _, j = tree.query(src + t)
        dt = (dst[j] - (src + t)).mean(0)
        t = t + dt
        if np.linalg.norm(dt) < tol:
            break
    return t


def chamfer_hausdorff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return 0.5 * (da.mean() + db.mean()), max(da.max(), db.max())


def px_scale(target_geom) -> float:
    return math.sqrt(NORMALIZED_AREA / target_geom.area)


@dataclass
class ShapeScore:
    chamfer: float
    hausdorff: float
    icp_shift_px: float


def score_arrangement(arrangement, target, n: int = N_SAMPLES) -> ShapeScore:
    """arrangement, target: shapely geometries in the same (world) units. Both are scaled so that the target has
    area 10,000 px^2."""
    s = px_scale(target)
    A = shapely.affinity.scale(arrangement, s, s, origin=(0, 0))
    B = shapely.affinity.scale(target, s, s, origin=(0, 0))
    pa = sample_boundary_uniform(A, n)
    pb = sample_boundary_uniform(B, n)
    t = icp_translation(pa, pb)
    c, h = chamfer_hausdorff(pa + t, pb)
    return ShapeScore(float(c), float(h), float(np.linalg.norm(t)))


def surface_scores(arrangement, target, n: int = 400, tol_px: float = 1.0) -> dict:
    """Surface-first scores (px at 10,000 px^2). The target surface is every boundary ring of the target (outline and
    real holes). Interior gaps between pieces are not penalized unless they reach the surface.
    - surface_coverage: fraction of target-surface samples within tol_px of the arrangement
    - surface_miss_mean / _max: distance from target-surface samples to the arrangement (0 where covered)
    - overhang_max / overhang_area: how far / how much the arrangement sticks out of the target"""
    s = px_scale(target)
    A = shapely.affinity.scale(arrangement, s, s, origin=(0, 0))
    B = shapely.affinity.scale(target, s, s, origin=(0, 0))
    # no ICP here: the arrangement is optimized against the target in place, and ICP would be pulled by the edges
    # of interior gaps, which these scores deliberately ignore
    pb = sample_boundary_uniform(B, n)
    d = shapely.distance(A, shapely.points(pb))
    out = A.difference(B)
    over = 0.0
    if not out.is_empty:
        po = sample_boundary_uniform(out, n)
        over = float(shapely.distance(B, shapely.points(po)).max())
    return {"surface_coverage": float((d <= tol_px).mean()), "surface_miss_mean": float(d.mean()),
            "surface_miss_max": float(d.max()), "overhang_max": over, "overhang_area_px2": float(out.area)}
