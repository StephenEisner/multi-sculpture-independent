"""IoU restricted to the regions where two forms disagree (M6: the dog's legs and ears, the duck's bill and tail).

Regions are boxes in the icon's SVG coordinates (24x24 viewbox, y down). They are mapped to the world frame of the
fitted target through its bounding box: normalization is a translation plus uniform scale, so the bbox map is exact.
"""

from __future__ import annotations

import shapely

from srd_dissect.shapes_mdi import CLOSE, silhouette_polygon

REGIONS = {
    "dog-side": {
        "legs": [(3.5, 14.5, 6.5, 21.5), (14.5, 14.5, 17.5, 21.5)],
        "ears_head": [(17.5, 2.5, 22.5, 8.5)],
    },
    "duck": {
        "bill": [(1.5, 6.0, 6.0, 10.5)],
        "tail": [(17.0, 11.0, 22.5, 15.0)],
    },
}


def region_geoms(icon: str, target_world: shapely.Geometry) -> dict[str, shapely.Geometry]:
    svg = silhouette_polygon(icon, close_px=CLOSE.get(icon, 0.15))
    sx0, sy0, sx1, sy1 = svg.bounds
    wx0, wy0, wx1, wy1 = target_world.bounds
    s = (wx1 - wx0) / (sx1 - sx0)
    f = lambda x, y: (wx0 + (x - sx0) * s, wy0 + (y - sy0) * s)
    out = {}
    for name, boxes in REGIONS.get(icon, {}).items():
        out[name] = shapely.unary_union([shapely.box(*f(x0, y0), *f(x1, y1)) for x0, y0, x1, y1 in boxes])
    return out


def region_iou(union: shapely.Geometry, target: shapely.Geometry, region: shapely.Geometry) -> float:
    inter = union.intersection(target).intersection(region).area
    uni = union.union(target).intersection(region).area
    return inter / uni if uni > 0 else float("nan")


def iou(union: shapely.Geometry, target: shapely.Geometry) -> float:
    return union.intersection(target).area / union.union(target).area
