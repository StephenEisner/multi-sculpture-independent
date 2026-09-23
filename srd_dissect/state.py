"""Pieces, poses and the dissection state.

A piece is one closed Arc-Line loop (a d4descent ``Shape``) in its own local frame, plus one pose per target.
Geometry is shared by construction; only poses differ between targets.

Pose convention: world = R(theta) @ diag(1, flip) @ local + trans. Reflection is applied first, in the local frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch

from d4descent.objects.arclines import Arc, Line, Primitive, Shape

from .geom import loop_order, shape_area_centroid, translate_shape


@dataclass(frozen=True)
class Pose:
    theta: float
    tx: float
    ty: float
    flip: int = 1  # +1 or -1

    def R(self) -> torch.Tensor:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return torch.tensor([[c, -s], [s, c]])

    def matrix(self) -> torch.Tensor:
        """2x2 linear part R @ diag(1, flip)."""
        return self.R() @ torch.diag(torch.tensor([1.0, float(self.flip)]))

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        """p: (..., 2) local -> world."""
        return p @ self.matrix().T + torch.tensor([self.tx, self.ty])

    def with_centroid(self, c_local: torch.Tensor, w: torch.Tensor) -> "Pose":
        """Same rotation/flip, translated so that local point c_local lands on world point w."""
        t = w - self.matrix() @ c_local
        return replace(self, tx=float(t[0]), ty=float(t[1]))

    def wrapped(self) -> "Pose":
        th = (self.theta + math.pi) % (2 * math.pi) - math.pi
        return replace(self, theta=th)


@dataclass
class Piece:
    pid: int
    shape: Shape
    poses: list[Pose]

    def clone(self, pid: int | None = None) -> "Piece":
        return Piece(self.pid if pid is None else pid, self.shape.clone(), list(self.poses))

    def area_centroid(self) -> tuple[float, torch.Tensor]:
        return shape_area_centroid(self.shape)

    def world_centroid(self, t: int) -> torch.Tensor:
        _, c = self.area_centroid()
        return self.poses[t].apply(c)

    def recentered(self) -> "Piece":
        """Exact: move the local origin to the area centroid and compensate every pose."""
        _, c = self.area_centroid()
        shape = translate_shape(self.shape, -c)
        poses = [p.with_centroid(torch.zeros(2), p.apply(c)) for p in self.poses]
        return Piece(self.pid, shape, poses)

    def n_segments(self) -> int:
        return len(self.shape.primitives)


@dataclass
class Dissection:
    pieces: list[Piece]
    n_targets: int
    next_pid: int = 0

    def __post_init__(self):
        if self.pieces:
            self.next_pid = max(self.next_pid, max(p.pid for p in self.pieces) + 1)

    def new_pid(self) -> int:
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def by_pid(self) -> dict[int, Piece]:
        return {p.pid: p for p in self.pieces}

    def clone(self) -> "Dissection":
        return Dissection([p.clone() for p in self.pieces], self.n_targets, self.next_pid)

    def replace_pieces(self, removed: set[int], added: list[Piece]) -> "Dissection":
        kept = [p for p in self.pieces if p.pid not in removed]
        d = Dissection(kept + added, self.n_targets, self.next_pid)
        return d

    def n_segments(self) -> int:
        return sum(p.n_segments() for p in self.pieces)


def make_piece(pid: int, prims: list[Primitive], poses: list[Pose]) -> Piece:
    return Piece(pid, Shape(list(prims)), poses)


__all__ = ["Pose", "Piece", "Dissection", "make_piece", "Arc", "Line", "loop_order"]
