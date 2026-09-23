"""Exact(ish) polygon geometry for evaluation: pieces and targets as shapely polygons, arcs densely sampled."""

from __future__ import annotations

import math

import numpy as np
import shapely
import torch

from d4descent.objects.arclines import Shape

from srd_dissect.geom import loop_order, sample_boundary
from srd_dissect.state import Dissection

ARC_SAMPLES = 64  # per primitive; at 10,000 px^2 scale a 64-gon arc deviates < 0.01 px from the true arc


def shape_polygon(shape: Shape, n_per_prim: int = ARC_SAMPLES) -> shapely.Polygon:
    P = sample_boundary(loop_order(shape), n_per_prim).double().numpy()
    return shapely.Polygon(P)


def local_polygons(D: Dissection, n_per_prim: int = ARC_SAMPLES) -> dict[int, shapely.Polygon]:
    return {p.pid: shape_polygon(p.shape, n_per_prim) for p in D.pieces}


def _matrix64(pose) -> np.ndarray:
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    return np.array([[c, -s], [s, c]]) @ np.diag([1.0, float(pose.flip)])


def pose_affine(pose) -> list[float]:
    """shapely affine_transform params [a, b, d, e, xoff, yoff] for world = M @ local + t (float64)."""
    M = _matrix64(pose)
    return [M[0, 0], M[0, 1], M[1, 0], M[1, 1], pose.tx, pose.ty]


def pose_affine_inv(pose) -> list[float]:
    M = _matrix64(pose)
    Mi = np.linalg.inv(M)
    t = -Mi @ np.array([pose.tx, pose.ty])
    return [Mi[0, 0], Mi[0, 1], Mi[1, 0], Mi[1, 1], t[0], t[1]]
