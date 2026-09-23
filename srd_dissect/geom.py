"""Geometry helpers on d4descent Arc-Line shapes (local frame, no autograd)."""

from __future__ import annotations

from typing import Callable

import torch

from d4descent.objects.arclines import (
    Arc,
    Line,
    Primitive,
    Shape,
    split_arc,
    vectorized_sample_arc,
)


def map_points(shape: Shape, f: Callable[[torch.Tensor], torch.Tensor], k_mul: float = 1.0) -> Shape:
    """Apply f to every control point, keeping shared-endpoint identity (connectivity). For f a similarity with
    scale s, pass k_mul = s (times -1 if f reflects): the bulge is a length and flips with the chord's normal."""
    new: dict[int, torch.Tensor] = {}

    def g(p: torch.Tensor) -> torch.Tensor:
        q = new.get(id(p))
        if q is None:
            q = f(p.detach()).detach().clone()
            new[id(p)] = q
        return q

    prims: list[Primitive] = []
    for pr in shape.primitives:
        if isinstance(pr, Arc):
            prims.append(Arc(g(pr.start), g(pr.end), (pr.k.detach() * k_mul).clone()))
        else:
            prims.append(Line(g(pr.start), g(pr.end)))
    return Shape(prims, payload=shape.payload)


def translate_shape(shape: Shape, d: torch.Tensor) -> Shape:
    return map_points(shape, lambda p: p + d)


def transform_shape(shape: Shape, M: torch.Tensor, t: torch.Tensor) -> Shape:
    """world = M @ p + t for a similarity M. If det(M) < 0 the bulge sign flips (the chord's left normal is reflected)."""
    det = float(torch.det(M))
    k_mul = (1.0 if det > 0 else -1.0) * abs(det) ** 0.5
    return map_points(shape, lambda p: M @ p + t, k_mul=k_mul)


def loop_order(shape: Shape) -> Shape:
    """Return the shape with primitives in loop order. Requires a single closed loop."""
    loops = shape.find_loops()
    if len(loops) != 1 or len(loops[0][0]) != len(shape.primitives):
        raise ValueError(f"expected a single closed loop, got {len(loops)} loops")
    return Shape([shape.primitives[i] for i in loops[0][0]], id=shape.id, payload=shape.payload)


def sample_boundary(shape: Shape, n_per_prim: int = 16) -> torch.Tensor:
    """Dense boundary polygon (loop order assumed), (n_prims * n_per_prim, 2). Excludes each primitive's end point."""
    t = torch.arange(n_per_prim, dtype=torch.float32) / n_per_prim
    pts = []
    for pr in shape.primitives:
        s, e = pr.start.detach(), pr.end.detach()
        if isinstance(pr, Arc) and pr.k.abs() > 1e-9:
            pts.append(vectorized_sample_arc(s, e, pr.k.detach(), t))
        else:
            pts.append(s + t[:, None] * (e - s))
    return torch.cat(pts, dim=0)


def polygon_area_centroid(P: torch.Tensor) -> tuple[float, torch.Tensor]:
    x, y = P[:, 0].double(), P[:, 1].double()
    x1, y1 = x.roll(-1), y.roll(-1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2
    if a.abs() < 1e-14:
        return 0.0, P.mean(0)
    cx = ((x + x1) * cr).sum() / (6 * a)
    cy = ((y + y1) * cr).sum() / (6 * a)
    return float(a), torch.stack([cx, cy]).float()


def shape_area_centroid(shape: Shape, n_per_prim: int = 32) -> tuple[float, torch.Tensor]:
    """Signed area (CCW > 0) and area centroid, from a dense boundary sampling."""
    return polygon_area_centroid(sample_boundary(loop_order(shape), n_per_prim))


def split_prim_at(pr: Primitive, t: float) -> tuple[Primitive, Primitive]:
    """Exact split of a line (any t) or arc (angle-uniform parameter t, exact up to float)."""
    if isinstance(pr, Arc):
        return split_arc(pr, t)
    # d4descent's split_line computes t*(start+end), which is only right at t=0.5
    m = (pr.start + t * (pr.end - pr.start)).detach().clone()
    return Line(pr.start, m), Line(m, pr.end)


def point_on_prim(pr: Primitive, t: float) -> torch.Tensor:
    if isinstance(pr, Arc):
        return vectorized_sample_arc(pr.start, pr.end, pr.k, torch.tensor([t]))[0]
    return pr.start + t * (pr.end - pr.start)


def circle_piece_prims(radius: float, n: int = 4) -> list[Primitive]:
    """CCW circle of n arcs centred at the origin."""
    from d4descent.objects.arclines import PrimitiveHelper

    return PrimitiveHelper.create_circle_arcs(n, (0.0, 0.0), radius)


def polygon_prims(points: list[tuple[float, float]]) -> list[Primitive]:
    from d4descent.objects.arclines import PrimitiveHelper

    return PrimitiveHelper.create_polygon(points)


def simplified(shape: Shape, rel_tol: float = 0.01) -> Shape:
    """Douglas-Peucker simplified copy of a line-only loop (tolerance relative to sqrt(area)); used to seed the
    initial partition with few segments, while the target keeps its full detail."""
    import shapely

    P = sample_boundary(loop_order(shape), 1).double().numpy()
    poly = shapely.Polygon(P)
    poly = shapely.geometry.polygon.orient(poly.simplify(rel_tol * poly.area ** 0.5, preserve_topology=True), 1.0)
    return Shape(polygon_prims([tuple(map(float, p)) for p in list(poly.exterior.coords)[:-1]]))
