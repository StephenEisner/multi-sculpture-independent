"""Pack pieces into d4descent ShapeCollections and render them per target, differentiably.

Rendering a piece in target t transforms its control points by the pose and calls d4descent's unmodified
winding-number rasterizer (``ShapeCollection._rasterize``). Distances are invariant under rigid motions and
|winding number| under reflection, so this is exact; under reflection the bulge k is negated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from d4descent.objects.arclines import ShapeCollection, ShapeCollectionArgs

from .state import Piece, Pose


@dataclass
class RenderConfig:
    size: int = 128
    lim: tuple[float, float] = (-1.5, 1.5)
    blur: float = 2 ** -0.5  # ramp half-width in pixels (d4descent AL-F setting)
    ks_scale: float = 2 ** -0.5  # d4descent AL-F setting (parameterization of k for the optimizer)

    @property
    def pixel(self) -> float:
        return (self.lim[1] - self.lim[0]) / self.size

    def grid(self, device="cpu") -> torch.Tensor:
        lo, hi = self.lim
        b = (torch.arange(self.size, device=device) + 0.5) / self.size * (hi - lo) + lo
        return torch.stack([b.expand(self.size, -1), b.unsqueeze(-1).expand(-1, self.size)], dim=-1)  # (H, W, 2)

    def ramp(self, sdf: torch.Tensor) -> torch.Tensor:
        vlim = self.blur * self.pixel
        return (-sdf.clamp(-vlim, vlim) + vlim) / (2 * vlim)


@dataclass
class Packed:
    """Flat, differentiable parameters for a list of pieces."""

    sc: ShapeCollection  # local geometry; control_points and ks are the geometry parameters
    theta: torch.Tensor  # (P, T)
    trans: torch.Tensor  # (P, T, 2)
    flip: torch.Tensor  # (P, T) +-1, constant
    pt_piece: torch.Tensor  # (n_points,) piece index of each control point
    k_piece: torch.Tensor  # (n_ks,) piece index of each bulge parameter
    pids: list[int]
    cfg: RenderConfig

    @staticmethod
    def build(pieces: list[Piece], cfg: RenderConfig, requires_grad: bool = True) -> "Packed":
        assert len(pieces) > 0
        Coll = ShapeCollection.patch_args(ShapeCollectionArgs(ks_scale=cfg.ks_scale))
        colls = [Coll.from_shape(p.shape) for p in pieces]
        pt_piece = torch.cat([torch.full((c.control_points.shape[0],), i) for i, c in enumerate(colls)])
        k_piece = torch.cat([torch.full((c.ks.shape[0],), i, dtype=torch.long) for i, c in enumerate(colls)])
        sc = Coll.cat(colls)
        T = len(pieces[0].poses)
        theta = torch.tensor([[q.theta for q in p.poses] for p in pieces], dtype=torch.float32)
        trans = torch.tensor([[[q.tx, q.ty] for q in p.poses] for p in pieces], dtype=torch.float32)
        flip = torch.tensor([[float(q.flip) for q in p.poses] for p in pieces], dtype=torch.float32)
        pk = Packed(sc, theta.reshape(-1, T), trans.reshape(-1, T, 2), flip.reshape(-1, T),
                    pt_piece.long(), k_piece, [p.pid for p in pieces], cfg)
        if requires_grad:
            pk.requires_grad_()
        return pk

    def requires_grad_(self) -> "Packed":
        self.sc.control_points = self.sc.control_points.detach().clone().requires_grad_()
        self.sc.ks = self.sc.ks.detach().clone().requires_grad_()
        self.theta = self.theta.detach().clone().requires_grad_()
        self.trans = self.trans.detach().clone().requires_grad_()
        return self

    @property
    def n_pieces(self) -> int:
        return len(self.pids)

    def param_groups(self) -> dict[str, torch.Tensor]:
        return {"vertices": self.sc.control_points, "bulges": self.sc.ks, "rotation": self.theta,
                "translation": self.trans}

    def target_collection(self, t: int) -> ShapeCollection:
        """Pieces placed in target t (world frame), sharing autograd with the parameters."""
        th = self.theta[:, t][self.pt_piece]
        c, s = torch.cos(th), torch.sin(th)
        f = self.flip[:, t][self.pt_piece]
        x, y = self.sc.control_points.unbind(-1)
        y = y * f
        wx = c * x - s * y + self.trans[:, t, 0][self.pt_piece]
        wy = s * x + c * y + self.trans[:, t, 1][self.pt_piece]
        ks = self.sc.ks * self.flip[:, t][self.k_piece] if self.sc.ks.numel() > 0 else self.sc.ks
        return ShapeCollection(
            control_points=torch.stack([wx, wy], -1), ks=ks, lines=self.sc.lines, arcs=self.sc.arcs,
            shapes=self.sc.shapes, shape_ids=self.sc.shape_ids, shape_payloads=self.sc.shape_payloads,
            args=self.sc.args,
        )

    def occupancy(self, t: int, grid: torch.Tensor | None = None) -> torch.Tensor:
        """Soft occupancy of each piece in target t: (P, H, W) in [0, 1]."""
        grid = self.cfg.grid() if grid is None else grid
        sdf, _ = self.target_collection(t)._rasterize(grid)
        return self.cfg.ramp(sdf)

    def local_areas(self) -> torch.Tensor:
        """Signed local area per piece (P,) — CCW positive. Differentiable."""
        return self.sc.compute_area()

    def segment_lengths(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Chord length of every primitive and its piece index."""
        cp = self.sc.control_points
        idx = torch.cat([self.sc.lines, self.sc.arcs[:, :2]], 0)
        L = (cp[idx[:, 1]] - cp[idx[:, 0]]).norm(dim=-1)
        return L, self.pt_piece[idx[:, 0]]

    def unpack(self, template: list[Piece]) -> list[Piece]:
        """Read parameters back into Pieces (same order and pids as ``template``)."""
        out = []
        with torch.no_grad():
            th, tr, fl = self.theta.detach(), self.trans.detach(), self.flip
            for i, p in enumerate(template):
                shape = self.sc.get_shape(i, detach=True)
                poses = [Pose(float(th[i, t]), float(tr[i, t, 0]), float(tr[i, t, 1]), int(fl[i, t])).wrapped()
                         for t in range(th.shape[1])]
                out.append(Piece(p.pid, shape, poses))
        return out


def render_target_image(shape_or_pieces, cfg: RenderConfig) -> torch.Tensor:
    """Render a target silhouette (a single d4descent Shape, possibly multi-loop) to a soft image (H, W)."""
    Coll = ShapeCollection.patch_args(ShapeCollectionArgs(ks_scale=cfg.ks_scale))
    sc = Coll.from_shape(shape_or_pieces)
    sdf, _ = sc._rasterize(cfg.grid())
    return cfg.ramp(sdf)[0].detach()
