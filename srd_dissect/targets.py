"""Target silhouettes as d4descent Shapes (world frame), normalized to a common area and centred at the origin."""

from __future__ import annotations

import math

import torch

from d4descent.objects.arclines import Shape

from .geom import circle_piece_prims, map_points, polygon_prims, shape_area_centroid

DEFAULT_AREA = 1.5  # world units^2; the render window is [-1.5, 1.5]^2


def normalize(shape: Shape, area: float = DEFAULT_AREA) -> Shape:
    a, c = shape_area_centroid(shape)
    s = math.sqrt(area / abs(a))
    return map_points(shape, lambda p: (p - c) * s, k_mul=s)


def square(area: float = DEFAULT_AREA) -> Shape:
    return normalize(Shape(polygon_prims([(-1, -1), (1, -1), (1, 1), (-1, 1)])), area)


def equilateral_triangle(area: float = DEFAULT_AREA) -> Shape:
    pts = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in (-90, 30, 150)]
    return normalize(Shape(polygon_prims(pts)), area)


def disk(area: float = DEFAULT_AREA, n: int = 4) -> Shape:
    return normalize(Shape(circle_piece_prims(1.0, n)), area)


def polygon(points: list[tuple[float, float]], area: float = DEFAULT_AREA) -> Shape:
    return normalize(Shape(polygon_prims(points)), area)


BUILTIN = {"square": square, "triangle": equilateral_triangle, "disk": disk}


def get(name: str, area: float = DEFAULT_AREA) -> Shape:
    return BUILTIN[name](area)


def fit_pair(shapes: list[Shape], lim: float = 1.5, fill: float = 0.85) -> list[Shape]:
    """Scale all targets to one common area, as large as possible such that every target (centred on its area
    centroid) fits inside fill * [-lim, lim]^2. Equal areas are required for a dissection."""
    from .geom import sample_boundary

    exts = []
    for s in shapes:
        P = sample_boundary(normalize(s, DEFAULT_AREA), 16)
        exts.append(float(P.abs().max()))
    k = min(fill * lim / e for e in exts)
    return [normalize(s, DEFAULT_AREA * k * k) for s in shapes]
