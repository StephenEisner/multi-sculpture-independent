"""M1 tests: exact shared rewrites leave every target's render unchanged; per-target rewrites leave the other
targets' renders unchanged; locks; gradient routing; scoring baseline."""

import math
import random

import pytest
import torch

from d4descent.objects.arclines import Arc, Line, Shape, ShapeRewriteArgs, ShapeRewriteSpec, ShapeRewriteType

from srd_dissect.geom import polygon_prims, transform_shape
from srd_dissect.loss import LossConfig, image_terms
from srd_dissect.render import Packed, RenderConfig, render_target_image
from srd_dissect.rewrites import (ProposalConfig, RepairConfig, Rewrite, apply_many, apply_rewrite, conflicts,
                                  propose, repair, valid_cut)
from srd_dissect.srd import SRDConfig, score_rewrites, select_greedy
from srd_dissect.state import Dissection, Piece, Pose
from srd_dissect import targets as TG

CFG = RenderConfig(size=128)


def blob_shape() -> Shape:
    """Asymmetric CCW loop with two arcs (one convex, one concave)."""
    prims = polygon_prims([(-0.45, -0.30), (0.50, -0.35), (0.55, 0.30), (0.05, 0.50), (-0.40, 0.35)])
    prims[1] = Arc(prims[1].start, prims[1].end, torch.tensor(0.12))
    prims[3] = Arc(prims[3].start, prims[3].end, torch.tensor(-0.06))
    return Shape(prims)


def make_dissection() -> Dissection:
    p0 = Piece(0, blob_shape(), [Pose(0.3, -0.2, -0.1, 1), Pose(-1.1, 0.3, 0.2, -1)]).recentered()
    p1 = Piece(1, TG.disk(0.2), [Pose(0.0, 0.8, 0.7, 1), Pose(2.0, -0.8, -0.6, 1)])
    return Dissection([p0, p1], 2, 2)


def union(D: Dissection) -> torch.Tensor:
    pk = Packed.build(D.pieces, CFG, requires_grad=False)
    with torch.no_grad():
        return torch.stack([pk.occupancy(t).sum(0) for t in range(D.n_targets)])  # (T, H, W)


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Per-target max abs difference and total L1 difference in pixel units."""
    d = (a - b).abs()
    return float(d.max()), float(d.sum())


def assert_render_equal(a: torch.Tensor, b: torch.Tensor, l1: float = 0.5, flips: int = 1):
    """Near-exact render equality, per target: total L1 difference below `l1` pixels, ignoring at most `flips`
    isolated flipped pixels (d4descent's rasterizer regularizes k=0 arcs with a radius ~(chord/2)^2/1e-4, and in
    float32 a pixel centre lying within ~1e-4 of such a chord's line can get the wrong winding number)."""
    for t in range(a.shape[0]):
        d = (a[t] - b[t]).abs()
        big = d > 0.5
        assert int(big.sum()) <= flips, (t, int(big.sum()))
        assert float(d[~big].sum()) < l1, (t, float(d[~big].sum()))


# ---------------------------------------------------------------- rendering


def test_pose_render_matches_transformed_shape():
    """Rendering via transformed control points equals rendering the explicitly transformed shape, incl. flips."""
    D = make_dissection()
    p = D.pieces[0]
    pk = Packed.build([p], CFG, requires_grad=False)
    for t in range(2):
        q = p.poses[t]
        world = transform_shape(p.shape, q.matrix(), torch.tensor([q.tx, q.ty]))
        ref = render_target_image(world, CFG)
        with torch.no_grad():
            got = pk.occupancy(t)[0]
        assert float((got - ref).abs().max()) < 1e-5, t


def test_flip_mirrors_image():
    """A piece with flip=-1 and theta=0 at the origin renders as the mirror (y -> -y) of the unflipped piece."""
    s = blob_shape()
    a = Packed.build([Piece(0, s, [Pose(0, 0, 0, 1)])], CFG, requires_grad=False)
    b = Packed.build([Piece(0, s, [Pose(0, 0, 0, -1)])], CFG, requires_grad=False)
    with torch.no_grad():
        ia, ib = a.occupancy(0)[0], b.occupancy(0)[0]
    assert float((ia.flip(0) - ib).abs().max()) < 1e-5


def test_recenter_is_exact():
    D = make_dissection()
    D2 = Dissection([p.recentered() for p in D.pieces], 2, 2)
    m, _ = diff_stats(union(D), union(D2))
    assert m < 1e-4


# ---------------------------------------------------------------- exact shared rewrites


def _grammar(D, pid, typ, args):
    return Rewrite({ShapeRewriteType.Split: "Split", ShapeRewriteType.ToArc: "ToArc",
                    ShapeRewriteType.ToLine: "ToLine", ShapeRewriteType.Merge: "Merge"}[typ],
                   pid=pid, spec=ShapeRewriteSpec(typ, args))


@pytest.mark.parametrize("prim", [0, 1, 2, 3, 4])
def test_split_exact_in_every_target(prim):
    """Split on a line is exact; on an arc it is exact up to float error."""
    D = make_dissection()
    before = union(D)
    D2 = apply_many(D, [_grammar(D, 0, ShapeRewriteType.Split, (prim,))])
    assert len(D2.by_pid()[0].shape.primitives) == 6
    m, s = diff_stats(before, union(D2))
    assert m < 2e-3 and s < 5e-2, (m, s)  # line: ~1e-6; arc: ~1.5e-3 max, ~0.016 px total


@pytest.mark.parametrize("prim", [0, 2, 4])
def test_to_arc_exact_in_every_target(prim):
    D = make_dissection()
    before = union(D)
    D2 = apply_many(D, [_grammar(D, 0, ShapeRewriteType.ToArc, (prim,))])
    assert_render_equal(before, union(D2))


def test_merge_after_split_and_to_line_on_flat_arc():
    """Conditional rewrites are exact when their condition holds: Merge undoing a Split of a line, ToLine on k=0."""
    D = make_dissection()
    before = union(D)
    D2 = apply_many(D, [_grammar(D, 0, ShapeRewriteType.Split, (0,))])
    D3 = apply_many(D2, [_grammar(D2, 0, ShapeRewriteType.Merge, (0, 1))])
    assert len(D3.by_pid()[0].shape.primitives) == 5
    assert diff_stats(before, union(D3))[0] < 1e-5
    D4 = apply_many(D, [_grammar(D, 0, ShapeRewriteType.ToArc, (0,))])
    D5 = apply_many(D4, [_grammar(D4, 0, ShapeRewriteType.ToLine, (0,))])
    assert diff_stats(before, union(D5))[0] < 1e-5  # back to a line: bit-for-bit the original geometry


def test_multiple_grammar_rewrites_on_one_piece():
    D = make_dissection()
    before = union(D)
    rws = [_grammar(D, 0, ShapeRewriteType.Split, (0,)), _grammar(D, 0, ShapeRewriteType.ToArc, (2,)),
           _grammar(D, 0, ShapeRewriteType.Split, (1,))]
    held = []
    for rw in rws:
        assert not conflicts(rw.locks(2), held)
        held += rw.locks(2)
    D2 = apply_many(D, rws)
    assert len(D2.by_pid()[0].shape.primitives) == 7
    assert_render_equal(before, union(D2))


def test_cutpart_exact_in_every_target():
    """CutPart: both halves inherit every pose, so the union is unchanged in every target. The only residual is at
    the two chord endpoints, where the ramps of three boundaries meet (a few pixels, bounded below)."""
    D = make_dissection()
    before = union(D)
    rng = random.Random(0)
    n_ok = 0
    for _ in range(200):
        p = D.by_pid()[0]
        a, b = rng.sample(range(5), 2)
        ta, tb = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
        if not valid_cut(p, a, ta, b, tb, 1e-3):
            continue
        rw = Rewrite("CutPart", pid=0, args=(a, ta, b, tb))
        D2 = D.clone()
        removed, added = apply_rewrite(D2, rw)
        D2 = D2.replace_pieces(removed, added)
        assert len(D2.pieces) == 3
        after = union(D2)
        for t in range(2):
            m, s = diff_stats(before[t], after[t])
            assert s < 2.0, (t, m, s)  # < 2 pixels' worth of occupancy in total
        # no overlap introduced
        cov, ov = image_terms(after[None], torch.zeros_like(after))
        cov0, ov0 = image_terms(before[None], torch.zeros_like(before))
        assert float((ov - ov0).abs().max()) < 1e-6
        # halves are CCW with positive area summing to the parent's
        a0 = p.area_centroid()[0]
        a12 = sum(q.area_centroid()[0] for q in added)
        assert all(q.area_centroid()[0] > 0 for q in added)
        assert abs(a12 - a0) < 1e-4
        n_ok += 1
        if n_ok >= 10:
            break
    assert n_ok >= 10


def test_repair_is_exact_on_clean_dissection():
    D = make_dissection()
    D2, stats = repair(D, RepairConfig())
    assert len(D2.pieces) == 2
    assert diff_stats(union(D), union(D2))[0] < 1e-3


def test_repair_splits_multiloop_piece_exactly():
    """A piece whose outline became two loops is split into two pieces, both inheriting all poses."""
    D = make_dissection()
    p = D.by_pid()[0]
    two = Shape(list(p.shape.primitives) + TG.disk(0.05).primitives)  # add a disjoint second loop far away?
    # translate the extra disk inside the render window but outside the blob
    from srd_dissect.geom import translate_shape
    disk = translate_shape(TG.disk(0.03), torch.tensor([0.9, -0.9]))
    two = Shape(list(p.shape.primitives) + list(disk.primitives))
    D.pieces[0] = Piece(0, two, p.poses)
    before = union(D)
    D2, stats = repair(D, RepairConfig(remove_area=1e-4))
    assert len(D2.pieces) == 3 and stats["split_loops"] == 1
    assert diff_stats(before, union(D2))[0] < 1e-3


# ---------------------------------------------------------------- per-target rewrites


@pytest.mark.parametrize("kind,t", [("Rotate", 0), ("Rotate", 1), ("Flip", 0), ("Flip", 1), ("SwapPoses", 0),
                                    ("SwapPoses", 1), ("Relocate", 0), ("Relocate", 1)])
def test_per_target_rewrite_leaves_other_target_unchanged(kind, t):
    D = make_dissection()
    before = union(D)
    args = {"Rotate": (math.pi / 2,), "Relocate": (0.4, -0.5)}.get(kind, ())
    rw = Rewrite(kind, pid=0, t=t, pid2=1 if kind == "SwapPoses" else None, args=args)
    assert rw.scope == "per-target"
    removed, added = apply_rewrite(D, rw)
    D2 = D.replace_pieces(removed, added)
    after = union(D2)
    other = 1 - t
    assert torch.equal(before[other], after[other]), kind  # bit-identical
    assert float((before[t] - after[t]).abs().max()) > 0.5, kind  # and the target itself changed
    # geometry untouched
    for q in D2.pieces:
        orig = D.by_pid()[q.pid]
        assert all(torch.equal(x.start, y.start) and torch.equal(x.end, y.end)
                   for x, y in zip(orig.shape.primitives, q.shape.primitives))


@pytest.mark.parametrize("kind", ["Rotate", "Flip"])
def test_rotate_flip_keep_world_centroid(kind):
    D = make_dissection()
    p = D.by_pid()[0]
    w = p.world_centroid(1)
    rw = Rewrite(kind, pid=0, t=1, args=(1.234,) if kind == "Rotate" else ())
    _, added = apply_rewrite(D, rw)
    assert float((added[0].world_centroid(1) - w).norm()) < 1e-5


def test_swap_exchanges_centroids():
    D = make_dissection()
    P = D.by_pid()
    w0, w1 = P[0].world_centroid(0), P[1].world_centroid(0)
    _, added = apply_rewrite(D, Rewrite("SwapPoses", pid=0, pid2=1, t=0))
    A = {q.pid: q for q in added}
    assert float((A[0].world_centroid(0) - w1).norm()) < 1e-5
    assert float((A[1].world_centroid(0) - w0).norm()) < 1e-5
    assert A[0].poses[0].theta == pytest.approx(P[1].poses[0].theta)


# ---------------------------------------------------------------- locks


def test_locks():
    split0 = Rewrite("Split", pid=0, spec=ShapeRewriteSpec(ShapeRewriteType.Split, (0,)))
    split0b = Rewrite("Split", pid=0, spec=ShapeRewriteSpec(ShapeRewriteType.Split, (1,)))
    merge01 = Rewrite("Merge", pid=0, spec=ShapeRewriteSpec(ShapeRewriteType.Merge, (0, 1)))
    cut0 = Rewrite("CutPart", pid=0, args=(0, 0.5, 2, 0.5))
    rot00 = Rewrite("Rotate", pid=0, t=0, args=(1.0,))
    rot01 = Rewrite("Rotate", pid=0, t=1, args=(1.0,))
    rot10 = Rewrite("Rotate", pid=1, t=0, args=(1.0,))
    swap010 = Rewrite("SwapPoses", pid=0, pid2=1, t=0)
    swap231 = Rewrite("SwapPoses", pid=2, pid2=3, t=1)
    add_a = Rewrite("AddPart", args=(0.04, 4, 0, 0, 0, 0, 0, 0))
    add_b = Rewrite("AddPart", args=(0.04, 4, 0, 1, 1, 0, 1, 1))
    L = lambda r: r.locks(2)
    assert not conflicts(L(split0), L(split0b))  # disjoint primitives of the same piece
    assert conflicts(L(split0), L(merge01))  # shared primitive
    assert conflicts(L(split0), L(cut0))  # shared rewrite holds (i, t) for every t
    assert conflicts(L(split0), L(rot01))
    assert conflicts(L(cut0), L(rot01))
    assert not conflicts(L(rot00), L(rot01))  # same piece, different targets
    assert not conflicts(L(rot00), L(rot10))
    assert conflicts(L(swap010), L(rot10))  # swap holds (j, t)
    assert conflicts(L(swap010), L(rot00))
    assert not conflicts(L(swap010), L(swap231))
    assert not conflicts(L(swap010), L(rot01))
    assert conflicts(L(add_a), L(add_b))  # one spawn per round
    assert not conflicts(L(add_a), L(cut0))


def test_select_greedy_respects_locks_and_order():
    rws = [Rewrite("Rotate", pid=0, t=0, args=(1.0,)), Rewrite("Rotate", pid=0, t=0, args=(2.0,)),
           Rewrite("Rotate", pid=0, t=1, args=(1.0,)), Rewrite("CutPart", pid=1, args=(0, 0.5, 1, 0.5))]
    chosen = select_greedy(rws, [1e-3, 2e-3, 5e-4, -1.0], 2, 1e-8)
    assert chosen == [rws[1], rws[2]]


# ---------------------------------------------------------------- gradient routing


def test_gradient_routing():
    """Geometry gets the sum of per-target gradients; each pose only its own target's."""
    D = make_dissection()
    tg = torch.stack([render_target_image(TG.square(), CFG), render_target_image(TG.equilateral_triangle(), CFG)])
    pk = Packed.build(D.pieces, CFG)
    params = [pk.sc.control_points, pk.sc.ks, pk.theta, pk.trans]
    per_t = []
    for t in range(2):
        cov, ov = image_terms(pk.occupancy(t).sum(0)[None, None], tg[t : t + 1])
        per_t.append(torch.autograd.grad((cov + ov).sum(), params))
    for t in range(2):
        o = 1 - t
        assert float(per_t[t][2][:, o].abs().max()) == 0.0  # theta of the other target
        assert float(per_t[t][3][:, o].abs().max()) == 0.0  # trans of the other target
        assert float(per_t[t][2][:, t].abs().max()) > 0
    cov0, ov0 = image_terms(pk.occupancy(0).sum(0)[None, None], tg[0:1])
    cov1, ov1 = image_terms(pk.occupancy(1).sum(0)[None, None], tg[1:2])
    g = torch.autograd.grad((cov0 + ov0 + cov1 + ov1).sum(), params)
    for i in (0, 1):
        assert torch.allclose(g[i], per_t[0][i] + per_t[1][i], atol=1e-8)


# ---------------------------------------------------------------- scoring


def test_scoring_baseline_noop_is_zero_and_exact_rewrites_are_near_zero():
    """A rewrite that changes nothing scores exactly against 'same pieces, same local step'. Exact rewrites (Split,
    ToArc) score ~0 up to the simplicity cost of the extra segment."""
    D = make_dissection()
    tg = torch.stack([render_target_image(TG.square(), CFG), render_target_image(TG.equilateral_triangle(), CFG)])
    # Rotate by 0 is a no-op per-target rewrite
    rws = [Rewrite("Rotate", pid=0, t=0, args=(0.0,)), Rewrite("Rotate", pid=1, t=1, args=(0.0,)),
           _grammar(D, 0, ShapeRewriteType.ToArc, (0,)), _grammar(D, 0, ShapeRewriteType.Split, (0,))]
    # with the local step: no-ops score exactly 0 against the same-pieces-stepped baseline
    deltas, failed = score_rewrites(D, rws, tg, SRDConfig(), 1.0)
    assert not failed
    assert abs(deltas[0]) < 1e-7 and abs(deltas[1]) < 1e-7, deltas
    # without the local step, exact rewrites score their simplicity change only (Split: -w_seg; ToArc: 0)
    cfg0 = SRDConfig(local_steps=0)
    deltas0, _ = score_rewrites(D, rws, tg, cfg0, 1.0)
    assert abs(deltas0[2]) < 1e-5, deltas0
    assert abs(deltas0[3] + cfg0.loss.w_seg) < 1e-6, deltas0


def test_propose_produces_valid_scoped_rewrites():
    D = make_dissection()
    tg = torch.stack([render_target_image(TG.square(), CFG), render_target_image(TG.equilateral_triangle(), CFG)])
    grid = CFG.grid()
    res = [grid[tg[t] > 0.5] for t in range(2)]
    rws = propose(D, res, ProposalConfig(allow_flip=True, n_proposals=64), random.Random(0))
    assert 0 < len(rws) <= 64
    kinds = {r.kind for r in rws}
    assert {"CutPart", "Rotate", "Split"} <= kinds, kinds
    for r in rws:
        removed, added = apply_rewrite(D.clone(), r)
        if r.scope == "per-target":
            assert r.t is not None


def test_cropped_rasterization_matches_full():
    """Per-piece cropped rendering equals the full-grid rendering (values and gradients), incl. major arcs/flips."""
    D = make_dissection()
    # add a piece with a major arc (|k| > chord/2)
    prims = polygon_prims([(-0.2, -0.1), (0.2, -0.1), (0.0, 0.2)])
    prims[0] = Arc(prims[0].start, prims[0].end, torch.tensor(0.35))
    D.pieces.append(Piece(2, Shape(prims), [Pose(0.4, 0.1, 0.9, -1), Pose(-2.0, 0.9, 0.1, 1)]))
    tg = torch.stack([render_target_image(TG.square(), CFG), render_target_image(TG.equilateral_triangle(), CFG)])
    outs = []
    for crop in (False, True):
        pk = Packed.build(D.pieces, CFG)
        pk.crop = crop
        occ = [pk.occupancy(t) for t in range(2)]
        cov, ov = image_terms(torch.stack([o.sum(0) for o in occ])[None], tg)
        g = torch.autograd.grad((cov + ov).sum(), [pk.sc.control_points, pk.sc.ks, pk.theta, pk.trans])
        outs.append((torch.stack(occ).detach(), g))
    assert float((outs[0][0] - outs[1][0]).abs().max()) < 1e-6
    for a, b in zip(outs[0][1], outs[1][1]):
        assert torch.allclose(a, b, atol=1e-7, rtol=1e-4)
