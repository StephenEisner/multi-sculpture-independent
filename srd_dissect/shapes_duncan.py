"""Duncan et al. 2017 input silhouettes, traced from the figure images of the paper PDF
(see scripts/extract_duncan_shapes.py). Traced inputs: close to, not identical with, the originals."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely

from d4descent.objects.arclines import Shape

from .geom import polygon_prims
from .targets import DEFAULT_AREA, normalize

DATA = Path(__file__).resolve().parents[1] / "data" / "duncan" / "shapes.json"

# Voronoi Scissors Table 1 pairs available from the figures (dog-bone is not among them)
PAIRS = {
    "bunny-egg": ("bunny", "egg", 6),
    "cat-bear": ("cat", "bear", 6),
    "serpent-apple": ("serpent", "apple", 5),
    "caterpillar-butterfly": ("caterpillar", "butterfly", 6),
    "trump-map": ("trump", "us_map", 5),
}


@lru_cache(None)
def _shapes() -> dict:
    return json.loads(DATA.read_text())["shapes"]


def duncan_polygon(name: str) -> shapely.Polygon:
    return shapely.geometry.polygon.orient(shapely.Polygon(_shapes()[name]["exterior"]), sign=1.0)


def duncan_shape(name: str, area: float = DEFAULT_AREA) -> Shape:
    xy = np.asarray(duncan_polygon(name).exterior.coords)[:-1]
    return normalize(Shape(polygon_prims([tuple(map(float, p)) for p in xy])), area)
