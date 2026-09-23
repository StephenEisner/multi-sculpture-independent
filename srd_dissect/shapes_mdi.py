"""Stand-in silhouettes from Material Design Icons (Apache-2.0, vendored in data/mdi/).

The Duncan et al. 2017 / Voronoi Scissors inputs are not reachable from this environment, so benchmark pairs are run on
these stand-ins. Results on them are NOT comparable to Voronoi Scissors Table 1.

Pipeline: SVG path -> subpaths sampled densely -> union of filled subpaths -> largest component, holes filled (a
silhouette) -> closing by a small buffer to seal hairline slits -> simplified polygon -> d4descent Shape of lines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely
from svgpathtools import svg2paths

from d4descent.objects.arclines import Shape

from .geom import polygon_prims
from .targets import DEFAULT_AREA, normalize

DATA = Path(__file__).resolve().parents[1] / "data" / "mdi"


def _subpath_polygons(path, n_per_seg: int = 24) -> list[np.ndarray]:
    polys = []
    for sub in path.continuous_subpaths():
        pts = []
        for seg in sub:
            t = np.linspace(0, 1, n_per_seg, endpoint=False)
            pts.append(np.array([seg.point(x) for x in t]))
        P = np.concatenate(pts)
        polys.append(np.stack([P.real, P.imag], -1))
    return polys


def silhouette_polygon(name: str, close_px: float = 0.15, simplify: float = 0.05) -> shapely.Polygon:
    """Filled outer silhouette of the icon, in SVG units (24x24 viewbox), y pointing down."""
    paths, _ = svg2paths(str(DATA / f"{name}.svg"))
    region = shapely.Polygon()
    for p in paths:
        for P in _subpath_polygons(p):
            if len(P) < 3:
                continue
            region = shapely.union(region, shapely.make_valid(shapely.Polygon(P)))  # silhouette: holes are filled anyway
    region = region.buffer(close_px).buffer(-close_px)  # seal hairline gaps
    comps = list(region.geoms) if hasattr(region, "geoms") else [region]
    big = max(comps, key=lambda g: g.area)
    sil = shapely.Polygon(big.exterior)
    return sil.simplify(simplify, preserve_topology=True)


# icons whose parts are separated by thin gaps (hat band, butterfly body, ladybug split) need a larger closing
CLOSE = {"hat-fedora": 1.5, "butterfly": 0.9, "ladybug": 0.9, "bug": 0.6}


def mdi_shape(name: str, area: float = DEFAULT_AREA, **kw) -> Shape:
    kw.setdefault("close_px", CLOSE.get(name, 0.15))
    poly = silhouette_polygon(name, **kw)
    # positive signed area in (x, y), the convention of shape_area_centroid. SVG y points down and so do our image
    # rows (row index = y), so y is kept as is and the icon renders upright.
    poly = shapely.geometry.polygon.orient(poly, sign=1.0)
    xy = np.asarray(poly.exterior.coords)[:-1]
    return normalize(Shape(polygon_prims([tuple(map(float, p)) for p in xy])), area)


# Stand-ins for the Voronoi Scissors / Duncan et al. pairs (None = no stand-in available).
PAIRS = {
    "dog-bone": ("dog-side", "bone"),
    "bunny-egg": ("rabbit", "egg"),
    "cat-bear": ("cat", "teddy-bear"),
    "serpent-apple": ("snake", "apple"),
    "caterpillar-butterfly": ("bug", "butterfly"),  # weak stand-in: no caterpillar icon
    "hat-ghost": ("hat-fedora", "ghost"),
    "dog-duck": ("dog-side", "duck"),
    "trump-map": None,
}
