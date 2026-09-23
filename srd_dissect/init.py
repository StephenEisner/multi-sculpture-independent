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
