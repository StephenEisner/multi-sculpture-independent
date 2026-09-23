"""Full evaluation of a dissection: validity (raw and after clean-up) and Chamfer/Hausdorff per target.

The headline numbers are computed on the *cleaned* (overlap-free, hence valid) dissection; raw numbers are reported
alongside so the effect of the clean-up is visible.
"""

from __future__ import annotations

import shapely

from d4descent.objects.arclines import Shape

from srd_dissect.state import Dissection

from .geometry import local_polygons, shape_polygon
from .protocol import px_scale, score_arrangement
from .validity import full_report, remove_overlaps, world_polys


def _scores(local, D, targets):
    out = []
    for t, tg in enumerate(targets):
        U = shapely.unary_union(list(world_polys(local, D, t).values()))
        s = score_arrangement(U, tg)
        out.append({"chamfer": s.chamfer, "hausdorff": s.hausdorff, "icp_shift_px": s.icp_shift_px})
    return out


def _to_px2(report, targets):
    """Add px^2 versions of the area residuals (target normalized to 10,000 px^2)."""
    for a, tg in zip(report["arrangements"], targets):
        s2 = px_scale(tg) ** 2
        for k in ("overlap_area", "internal_gap_area", "uncovered_area", "overhang_area"):
            a[k + "_px2"] = a[k] * s2
    return report


def evaluate(D: Dissection, target_shapes: list[Shape]) -> dict:
    targets = [shape_polygon(s) for s in target_shapes]
    local = local_polygons(D)
    raw = _to_px2(full_report(D, targets, local), targets)
    raw["scores"] = _scores(local, D, targets)
    clean_local = remove_overlaps(local, D)
    clean = _to_px2(full_report(D, targets, clean_local), targets)
    clean["scores"] = _scores(clean_local, D, targets)
    avg = lambda rep, k: sum(s[k] for s in rep["scores"]) / len(rep["scores"])
    return {
        "raw": raw, "clean": clean,
        "avg_chamfer": avg(clean, "chamfer"), "avg_hausdorff": avg(clean, "hausdorff"),
        "avg_chamfer_raw": avg(raw, "chamfer"), "avg_hausdorff_raw": avg(raw, "hausdorff"),
        "n_pieces": len(D.pieces), "n_pieces_clean": clean["n_pieces"],
    }
