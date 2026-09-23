"""M2 tests: evaluation protocol and validity checker on synthetic cases with known answers."""

import math

import numpy as np
import pytest
import shapely
import torch
from shapely import affinity

from d4descent.objects.arclines import Shape

from eval.evaluate import evaluate
from eval.geometry import local_polygons
from eval.protocol import chamfer_hausdorff, icp_translation, px_scale, sample_boundary_uniform, score_arrangement
from eval.validity import arrangement_validity, full_report, remove_overlaps, world_polys
from srd_dissect.geom import polygon_prims
from srd_dissect.state import Dissection, Piece, Pose
from srd_dissect import targets as TG


def rect_piece(pid, w, h, poses):
    return Piece(pid, Shape(polygon_prims([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])), poses)


# ---------------------------------------------------------------- protocol


def test_sampling_is_uniform_and_on_boundary():
    sq = shapely.box(0, 0, 1, 1)
    P = sample_boundary_uniform(sq, 100)
    assert P.shape == (100, 2)
    d = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    assert np.allclose(d, 0.04, atol=1e-9)  # perimeter 4 / 100
    assert np.allclose(sq.boundary.distance(shapely.points(P)), 0, atol=1e-12)


def test_icp_recovers_translation_and_scores_zero():
    g = TG.disk(1.0)
    from eval.geometry import shape_polygon

    poly = shape_polygon(g)
    moved = affinity.translate(poly, 0.3, -0.2)
    s = score_arrangement(moved, poly)
    assert s.chamfer < 1e-6 and s.hausdorff < 1e-6
    assert s.icp_shift_px == pytest.approx(math.hypot(0.3, -0.2) * px_scale(poly), rel=1e-4)


def test_concentric_circles_known_distance():
    """Circle of area 10,000 px^2 vs concentric circle 2 px larger in radius: Chamfer = Hausdorff ~ 2 px."""
    r = math.sqrt(10_000 / math.pi)
    A = shapely.Point(0, 0).buffer(r, 256)
    B = shapely.Point(0, 0).buffer(r + 2.0, 256)
    s = score_arrangement(B, A)  # target A has area ~10,000 so scale ~1
    assert s.chamfer == pytest.approx(2.0, abs=0.05)
    assert s.hausdorff == pytest.approx(2.0, abs=0.3)


def test_hausdorff_of_a_bump():
    """Square with a thin 5 px spike: Hausdorff ~ spike length, Chamfer small."""
    A = shapely.box(0, 0, 100, 100)
    B = shapely.union(A, shapely.Polygon([(49, 100), (51, 100), (50, 105)]))
    s = score_arrangement(B, A)
    assert 3.0 < s.hausdorff <= 5.0 + 1e-6
    assert s.chamfer < 1.5  # sample spacing is ~4 px, so identical boundaries already score ~1 px


# ---------------------------------------------------------------- validity


def test_exact_two_piece_dissection():
    """Square (2x2... area 4) cut in two 2x1 rectangles; target B is the 4x1 rectangle made by placing them end to
    end. Zero overlap, zero gaps, near-zero Chamfer/Hausdorff in both arrangements."""
    a = rect_piece(0, 2, 1, [Pose(0, 0, 0.5), Pose(0, -1, 0)])
    b = rect_piece(1, 2, 1, [Pose(0, 0, -0.5), Pose(math.pi, 1, 0)])
    D = Dissection([a, b], 2, 2)
    A = Shape(polygon_prims([(-1, -1), (1, -1), (1, 1), (-1, 1)]))
    B = Shape(polygon_prims([(-2, -0.5), (2, -0.5), (2, 0.5), (-2, 0.5)]))
    r = evaluate(D, [A, B])
    for rep in (r["raw"], r["clean"]):
        for arr in rep["arrangements"]:
            assert arr["overlap_area"] < 1e-9
            assert arr["internal_gap_area"] < 1e-9
            assert arr["uncovered_area"] < 1e-9
            assert arr["overhang_area"] < 1e-9
        assert rep["all_pieces_simple_connected"]
    # boundary samples of union vs target: same geometry, samples only differ by phase along the boundary
    assert r["avg_chamfer"] < 0.5 * math.sqrt(10_000 / 4) * 4 * 2 / 100  # < half the sample spacing in px
    assert r["avg_hausdorff"] < math.sqrt(10_000 / 4) * 8 / 100


def test_known_overlap_and_cleanup():
    """Two unit squares overlapping by 0.5 x 1 in arrangement 0 (disjoint in arrangement 1)."""
    a = rect_piece(0, 1, 1, [Pose(0, 0, 0), Pose(0, 0, 0)])
    b = rect_piece(1, 1, 1, [Pose(0, 0.5, 0), Pose(0, 3, 0)])
    D = Dissection([a, b], 2, 2)
    local = local_polygons(D)
    tg = [shapely.box(-0.5, -0.5, 1.0, 0.5), shapely.box(-0.5, -0.5, 3.5, 0.5)]
    v0 = arrangement_validity(world_polys(local, D, 0), tg[0])
    v1 = arrangement_validity(world_polys(local, D, 1), tg[1])
    assert v0.overlap_area == pytest.approx(0.5, abs=1e-9)
    assert v1.overlap_area == pytest.approx(0.0, abs=1e-12)
    clean = remove_overlaps(local, D)
    for t in range(2):
        v = arrangement_validity(world_polys(clean, D, t), tg[t])
        assert v.overlap_area < 1e-12
    # the trimmed piece lost exactly the overlap: in arrangement 1 that material is now a gap/uncovered area
    assert sum(g.area for g in clean.values()) == pytest.approx(1.5, abs=1e-9)
    v1c = arrangement_validity(world_polys(clean, D, 1), tg[1])
    assert v1c.uncovered_area - v1.uncovered_area == pytest.approx(0.5, abs=1e-9)


def test_cleanup_respects_isometry_across_targets():
    """The trim of piece i is applied in its local frame, so every arrangement shows the same trimmed piece."""
    a = rect_piece(0, 2, 1, [Pose(0, 0, 0), Pose(0.7, 5, 5)])
    b = rect_piece(1, 1, 1, [Pose(0, 0.8, 0), Pose(-0.3, 5.2, 5.4)])
    D = Dissection([a, b], 2, 2)
    clean = remove_overlaps(local_polygons(D), D)
    for t in range(2):
        W = world_polys(clean, D, t)
        assert W[0].intersection(W[1]).area < 1e-12
    areas = {pid: g.area for pid, g in clean.items()}
    for t in range(2):
        W = world_polys(clean, D, t)
        for pid in W:
            assert W[pid].area == pytest.approx(areas[pid], rel=1e-9)


def test_internal_gap():
    """Four squares ringing a square hole: internal gap = hole area."""
    poses = [Pose(0, x, y) for x, y in [(-1, -1), (0, -1), (1, -1)]]
    pieces = [rect_piece(0, 3, 1, [Pose(0, 0, -1)]), rect_piece(1, 3, 1, [Pose(0, 0, 1)]),
              rect_piece(2, 1, 1, [Pose(0, -1, 0)]), rect_piece(3, 1, 1, [Pose(0, 1, 0)])]
    D = Dissection(pieces, 1, 4)
    rep = full_report(D, [shapely.box(-1.5, -1.5, 1.5, 1.5)])
    a = rep["arrangements"][0]
    assert a["internal_gap_area"] == pytest.approx(1.0, abs=1e-9)
    assert a["uncovered_area"] == pytest.approx(1.0, abs=1e-9)
    assert a["overlap_area"] < 1e-9


def test_piece_validity_flags_self_intersection():
    bow = Shape(polygon_prims([(0, 0), (1, 1), (1, 0), (0, 1)]))  # bow-tie
    D = Dissection([Piece(0, bow, [Pose(0, 0, 0)])], 1, 1)
    from eval.geometry import shape_polygon

    rep = full_report(D, [shapely.box(0, 0, 1, 1)], {0: shapely.Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])})
    assert not rep["all_pieces_simple_connected"]


def test_surface_scores():
    """Interior gap: surface fully covered. Missing bite at the edge: miss distance ~ bite depth. Spike: overhang."""
    from eval.protocol import surface_scores

    T = shapely.box(0, 0, 100, 100)
    gap = T.difference(shapely.box(40, 40, 60, 60))
    s = surface_scores(gap, T)
    assert s["surface_coverage"] == 1.0 and s["overhang_max"] == 0.0
    bite = T.difference(shapely.box(40, 90, 60, 100.1))
    s = surface_scores(bite, T)
    assert s["surface_coverage"] < 0.97 and 8.0 < s["surface_miss_max"] <= 10.0 + 1e-6
    spike = shapely.union(T, shapely.box(49, 100, 51, 106))
    s = surface_scores(spike, T)
    assert s["surface_coverage"] == 1.0 and 5.0 < s["overhang_max"] <= 6.0 + 1e-6
