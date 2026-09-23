"""Initializations.

- ``partition_init``: partition target A into k pieces by random valid chords; place the pieces in the other targets
  with random poses at random covered spots (spec default).
- ``growth_init``: a few tiny disks spawned at random covered spots in each target; CutPart/AddPart build from there.
"""

from __future__ import annotations

import math
import random

import torch

from d4descent.objects.arclines import Shape

from .geom import circle_piece_prims
from .render import RenderConfig
from .rewrites import Rewrite, apply_rewrite, valid_cut
from .state import Dissection, Piece, Pose


def _covered(targets: torch.Tensor, cfg: RenderConfig) -> list[torch.Tensor]:
    g = cfg.grid()
    return [g[targets[t] > 0.5] for t in range(targets.shape[0])]


def partition_init(target_a: Shape, targets: torch.Tensor, k: int, cfg: RenderConfig, rng: random.Random,
                   min_area: float = 0.02, tries: int = 200) -> Dissection:
    T = targets.shape[0]
    covered = _covered(targets, cfg)
    D = Dissection([Piece(0, target_a.clone(), [Pose(0.0, 0.0, 0.0, 1)] * T)], T, 1)
    D.pieces[0] = D.pieces[0].recentered()
    n = 0
    while len(D.pieces) < k and n < tries:
        n += 1
        areas = [p.area_centroid()[0] for p in D.pieces]
        p = rng.choices(D.pieces, weights=areas)[0]
        m = len(p.shape.primitives)
        if m < 2:
            continue
        a, b = rng.sample(range(m), 2)
        ta, tb = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
        if not valid_cut(p, a, ta, b, tb, min_area):
            continue
        removed, added = apply_rewrite(D, Rewrite("CutPart", pid=p.pid, args=(a, ta, b, tb)))
        D = D.replace_pieces(removed, added)
    # random poses in targets 1..T-1
    for p in D.pieces:
        for t in range(1, T):
            x, y = covered[t][rng.randrange(covered[t].shape[0])].tolist()
            p.poses[t] = Pose(rng.uniform(-math.pi, math.pi), x, y, 1)
    return D


def growth_init(targets: torch.Tensor, n: int, radius: float, cfg: RenderConfig, rng: random.Random) -> Dissection:
    T = targets.shape[0]
    covered = _covered(targets, cfg)
    pieces = []
    for i in range(n):
        poses = []
        for t in range(T):
            x, y = covered[t][rng.randrange(covered[t].shape[0])].tolist()
            poses.append(Pose(rng.uniform(-math.pi, math.pi), x, y, 1))
        pieces.append(Piece(i, Shape(circle_piece_prims(radius, 4)), poses))
    return Dissection(pieces, T, n)


# ---------------------------------------------------------------- overlay initialization


def _poly_to_piece(pid: int, poly, poses: list[Pose], tol: float) -> Piece:
    import shapely

    from .geom import polygon_prims

    poly = shapely.geometry.polygon.orient(poly.simplify(tol, preserve_topology=True), 1.0)
    pts = [tuple(map(float, p)) for p in list(poly.exterior.coords)[:-1]]
    return Piece(pid, Shape(polygon_prims(pts)), list(poses)).recentered()


def _pose_from_affine(M, t) -> Pose:
    """world = M @ local + t with M = R(theta) @ diag(1, flip)."""
    flip = 1 if M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0] > 0 else -1
    # first column of M is R(theta) @ e1
    theta = math.atan2(M[1, 0], M[0, 0])
    return Pose(theta, float(t[0]), float(t[1]), flip)


def overlay_init(shape_a: Shape, shape_b: Shape, targets: torch.Tensor, k: int, cfg: RenderConfig,
                 rng: random.Random, allow_flip: bool = False, n_angles: int = 72, top: int = 4,
                 min_frac: float = 0.004, tol_frac: float = 0.01, min_core: int = 1) -> Dissection:
    """Lay B over A in (one of the top few) best-overlap rigid placements, cut A along B's outline, and use the pieces:
    the overlap A∩B' is placed exactly in both targets; the leftover parts of A are moved into the uncovered parts
    of B. Then cut/merge to exactly k pieces. Different seeds pick different top placements and cuts."""
    import numpy as np
    import shapely
    from shapely import affinity

    from eval.geometry import shape_polygon

    A, B = shape_polygon(shape_a, 4), shape_polygon(shape_b, 4)
    area = A.area
    tol = tol_frac * area ** 0.5
    cands = []
    for f in ([1, -1] if allow_flip else [1]):
        Bf = affinity.scale(B, 1, f, origin=(0, 0))
        for i in range(n_angles):
            a = 2 * math.pi * i / n_angles
            Br = affinity.rotate(Bf, a, origin=(0, 0), use_radians=True)
            best = None
            for dx in np.linspace(-0.15, 0.15, 5):
                for dy in np.linspace(-0.15, 0.15, 5):
                    Bt = affinity.translate(Br, dx, dy)
                    iou = A.intersection(Bt).area / A.union(Bt).area
                    if best is None or iou > best[0]:
                        best = (iou, a, f, dx, dy)
            cands.append(best)
    cands.sort(reverse=True)
    iou, a, f, dx, dy = rng.choice(cands[:top])
    # B' = T(B) = R(a) diag(1,f) B + (dx, dy); a point p of A sits at T^-1(p) in B's frame
    M = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ np.diag([1.0, f])
    t = np.array([dx, dy])
    Mi = np.linalg.inv(M)
    ti = -Mi @ t
    pose_core_b = _pose_from_affine(Mi, ti)
    Bp = affinity.affine_transform(B, [M[0, 0], M[0, 1], M[1, 0], M[1, 1], dx, dy])

    parts = lambda g: [x for x in getattr(g, "geoms", [g]) if x.geom_type == "Polygon" and x.area > 0]
    core = parts(A.intersection(Bp))
    rest = parts(A.difference(Bp))
    # merge slivers (< min_frac of the area) into the neighbour with the longest shared boundary, then make sure the
    # shared core gets at least `min_core` of the k pieces by merging the smallest leftovers away
    polys = [(p, "core") for p in core] + [(p, "rest") for p in rest]

    def shared(p, q):
        return p.boundary.intersection(q.buffer(1e-7)).length

    def merge_smallest(pred) -> bool:
        order = sorted([i for i, (p, kind) in enumerate(polys) if pred(p, kind)], key=lambda i: polys[i][0].area)
        for i0 in order:
            small = polys[i0][0]
            nbrs = sorted([i for i in range(len(polys)) if i != i0], key=lambda i: -shared(small, polys[i][0]))
            for j in nbrs[:3]:
                if shared(small, polys[j][0]) <= 0:
                    break
                u = polys[j][0].union(small).buffer(1e-4).buffer(-1e-4)  # close hairline contacts
                if u.geom_type != "Polygon":  # drop zero-area specks left by the union
                    big = [g for g in getattr(u, "geoms", []) if g.geom_type == "Polygon" and g.area > 1e-6]
                    u = big[0] if len(big) == 1 else u
                if u.geom_type == "Polygon":
                    polys[j] = (u, polys[j][1])
                    polys.pop(i0)
                    return True
        return False

    while merge_smallest(lambda p, kind: p.area < min_frac * area):
        pass
    n_core_target = max(min_core, k - sum(kind == "rest" for _, kind in polys))
    while sum(kind == "rest" for _, kind in polys) > k - n_core_target and merge_smallest(lambda p, kind: kind == "rest"):
        pass
    # uncovered parts of B (in B's frame), largest first
    inv = [Mi[0, 0], Mi[0, 1], Mi[1, 0], Mi[1, 1], ti[0], ti[1]]
    holes_b = sorted(parts(B.difference(shapely.unary_union([affinity.affine_transform(p, inv)
                                                              for p, kind in polys if kind == "core"]))),
                     key=lambda g: -g.area)
    T = targets.shape[0]
    pieces: list[Piece] = []
    rest_polys = sorted([p for p, kind in polys if kind == "rest"], key=lambda g: -g.area)
    for p, kind in polys:
        if kind == "core":
            pieces.append(_poly_to_piece(len(pieces), p, [Pose(0, 0, 0, 1), pose_core_b], tol))
    for i, p in enumerate(rest_polys):
        dst = holes_b[i % len(holes_b)] if holes_b else B
        c, dc = np.array(p.centroid.coords[0]), np.array(dst.centroid.coords[0])
        best = None
        for j in range(8):  # orientation that best fits the hole
            th = 2 * math.pi * j / 8 + rng.uniform(-0.2, 0.2)
            moved = affinity.translate(affinity.rotate(p, th, origin=tuple(c), use_radians=True), *(dc - c))
            sc = moved.intersection(dst).area
            if best is None or sc > best[0]:
                best = (sc, th)
        th = best[1]
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        pose_b = _pose_from_affine(R, dc - R @ c)
        pieces.append(_poly_to_piece(len(pieces), p, [Pose(0, 0, 0, 1), pose_b], tol))
    D = Dissection(pieces, T, len(pieces))
    # exactly k pieces: cut the largest (exact: both halves keep the parent's poses), or merge the smallest pair
    from .rewrites import _cut_shape
    from .geom import shape_area_centroid

    tries = 0
    while len(D.pieces) < k and tries < 20:
        tries += 1
        p = max(D.pieces, key=lambda q: q.area_centroid()[0])
        m = len(p.shape.primitives)
        best = None
        for _ in range(60):  # among valid random chords, take the most even split
            a_, b_ = rng.sample(range(m), 2)
            ta, tb = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
            if not valid_cut(p, a_, ta, b_, tb, 0.02):
                continue
            s1, s2 = _cut_shape(p.shape, a_, ta, b_, tb)
            r = min(shape_area_centroid(s1)[0], shape_area_centroid(s2)[0])
            if best is None or r > best[0]:
                best = (r, (a_, ta, b_, tb))
        if best is not None:
            removed, added = apply_rewrite(D, Rewrite("CutPart", pid=p.pid, args=best[1]))
            D = D.replace_pieces(removed, added)
    while len(D.pieces) > k:
        D.pieces.sort(key=lambda q: q.area_centroid()[0])
        D.pieces.pop(0)  # drop the smallest leftover; its area becomes a hole the optimizer refills
    return D
