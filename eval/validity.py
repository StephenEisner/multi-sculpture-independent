"""Validity checker and polygon-level clean-up for a multi-target dissection.

Per arrangement: pairwise overlap area, internal gap area (holes enclosed by the union that the target does not
have), uncovered target area, overhang area, and whether every piece is a simple, connected polygon.

Clean-up (``remove_overlaps``): an isometry-preserving polygon clean-up. Pieces are processed in a fixed order; each
piece is trimmed, in its local frame, by every earlier piece mapped into its frame from *every* arrangement. Trimming
a piece changes it in all arrangements (that is what keeps it rigid), so this can only remove material: the result
has zero overlap in every arrangement, and the lost material shows up as extra gap/uncovered area, which is reported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import shapely
from shapely import affinity

from srd_dissect.state import Dissection

from .geometry import local_polygons, pose_affine, pose_affine_inv

EPS_AREA = 1e-12


@dataclass
class ArrangementValidity:
    overlap_area: float  # sum over pairs of intersection area
    internal_gap_area: float  # holes of the union (targets are simply connected)
    uncovered_area: float  # target minus union
    overhang_area: float  # union minus target
    union_area: float
    target_area: float
    n_components: int  # connected components of the union


@dataclass
class PieceValidity:
    pid: int
    simple: bool
    connected: bool
    area: float


def world_polys(local: dict[int, shapely.Geometry], D: Dissection, t: int) -> dict[int, shapely.Geometry]:
    """World-frame pieces of arrangement t. Invalid (self-intersecting) pieces are made valid for the area
    computations; ``piece_validity`` reports them on the raw geometry."""
    P = D.by_pid()
    return {pid: shapely.make_valid(affinity.affine_transform(g, pose_affine(P[pid].poses[t])))
            for pid, g in local.items()}


def arrangement_validity(polys: dict[int, shapely.Geometry], target: shapely.Geometry) -> ArrangementValidity:
    items = list(polys.values())
    tree = shapely.STRtree(items)
    ov = 0.0
    for i, g in enumerate(items):
        for j in tree.query(g):
            if j > i:
                ov += g.intersection(items[j]).area
    U = shapely.unary_union(items)
    comps = list(U.geoms) if hasattr(U, "geoms") else [U]
    holes = sum(shapely.Polygon(r).area for c in comps for r in c.interiors)
    return ArrangementValidity(
        overlap_area=ov, internal_gap_area=holes, uncovered_area=target.difference(U).area,
        overhang_area=U.difference(target).area, union_area=U.area, target_area=target.area,
        n_components=len([c for c in comps if c.area > EPS_AREA]),
    )


def piece_validity(local: dict[int, shapely.Geometry]) -> list[PieceValidity]:
    out = []
    for pid, g in local.items():
        connected = g.geom_type == "Polygon"
        simple = connected and g.is_valid and g.exterior.is_simple and len(g.interiors) == 0
        out.append(PieceValidity(pid, bool(simple), bool(connected), float(g.area)))
    return out


def remove_overlaps(local: dict[int, shapely.Geometry], D: Dissection, keep_largest: bool = True
                    ) -> dict[int, shapely.Geometry]:
    """Isometric clean-up (see module docstring). Returns new local-frame polygons (same pids; empty pieces dropped)."""
    P = D.by_pid()
    order = sorted(local, key=lambda pid: -local[pid].area)  # larger pieces keep their material
    done: dict[int, shapely.Geometry] = {}
    for pid in order:
        g = shapely.make_valid(local[pid])
        for t in range(D.n_targets):
            inv = pose_affine_inv(P[pid].poses[t])
            for q, h in done.items():
                hw = affinity.affine_transform(h, pose_affine(P[q].poses[t]))
                g = g.difference(affinity.affine_transform(hw, inv))
        if keep_largest and hasattr(g, "geoms"):
            parts = [x for x in g.geoms if x.geom_type == "Polygon"]
            g = max(parts, key=lambda x: x.area) if parts else shapely.Polygon()
        if hasattr(g, "interiors") and len(g.interiors):
            g = shapely.Polygon(g.exterior)  # cannot gain area: a hole in a trimmed piece is enclosed by the piece
        if g.area > EPS_AREA:
            done[pid] = g
    return done


def full_report(D: Dissection, targets: list[shapely.Geometry], local: dict[int, shapely.Geometry] | None = None
                ) -> dict:
    local = local_polygons(D) if local is None else local
    arr = [arrangement_validity(world_polys(local, D, t), targets[t]) for t in range(D.n_targets)]
    pcs = piece_validity(local)
    return {"arrangements": [asdict(a) for a in arr], "pieces": [asdict(p) for p in pcs],
            "all_pieces_simple_connected": all(p.simple and p.connected for p in pcs),
            "n_pieces": len(pcs)}
