"""Rewrites on a multi-target dissection: kinds, scopes, locks, application, proposal sampling, repair.

Scope rule: a rewrite's scope is what it writes to. Rewrites that edit piece geometry are *shared* (they change the
piece in every target at once); rewrites that edit one pose are *per-target*. Scoring is always on the loss summed
over all targets, whatever the scope.

Locks (greedy apply step). A lock is (key, mode) with mode "X" (exclusive) or "S" (shared); two rewrites conflict
if they hold the same key and at least one holds it exclusively.
- shared rewrite on piece i:      ("piece", i, t) X for every t
- d4descent grammar rewrite on i: ("piece", i, t) S for every t, plus ("prim", i, j) X for the primitives it edits.
  Several Split/Merge/ToArc/ToLine on disjoint primitives of the same piece can therefore be applied together
  (exactly as d4descent does within one shape), but they still conflict with every other rewrite on piece i.
- per-target rewrite on (i, t):   ("piece", i, t) X
- SwapPoses(i, j, t):             ("piece", i, t) X and ("piece", j, t) X
- AddPart:                        ("add",) X  (one spawn per round)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

import numpy as np
import shapely
import torch

from d4descent.objects.arclines import Line, Shape, ShapeRewriteArgs, ShapeRewriteSpec, ShapeRewriteType

from .geom import circle_piece_prims, loop_order, point_on_prim, sample_boundary, shape_area_centroid, split_prim_at
from .state import Dissection, Piece, Pose

# ---------------------------------------------------------------- kinds and scopes

GRAMMAR = {"Split", "Merge", "MergeClose", "ToArc", "ToLine"}
SHARED = GRAMMAR | {"CutPart", "AddPart", "RemoveSmallPart", "FusePart"}
PER_TARGET = {"Rotate", "Flip", "SwapPoses", "Relocate"}
HYBRID = {"TrimOverlap"}
ALL_KINDS = SHARED | PER_TARGET | HYBRID

_GRAMMAR_TYPE = {
    ShapeRewriteType.Split: "Split",
    ShapeRewriteType.Merge: "Merge",
    ShapeRewriteType.MergeClose: "MergeClose",
    ShapeRewriteType.ToArc: "ToArc",
    ShapeRewriteType.ToLine: "ToLine",
}


@dataclass(frozen=True)
class Rewrite:
    kind: str
    pid: int | None = None
    t: int | None = None
    pid2: int | None = None
    args: tuple = ()
    spec: ShapeRewriteSpec | None = field(default=None, compare=False)  # d4descent spec for grammar rewrites

    @property
    def scope(self) -> str:
        if self.kind in SHARED:
            return "shared"
        if self.kind in PER_TARGET:
            return "per-target"
        return "hybrid"

    def locks(self, n_targets: int) -> list[tuple[tuple, str]]:
        k = self.kind
        if k == "AddPart":
            return [(("add",), "X")]
        if k in GRAMMAR:
            assert self.spec is not None
            return [(("piece", self.pid, t), "S") for t in range(n_targets)] + [
                (("prim", self.pid, int(j)), "X") for j in self.spec.args
            ]
        if k == "SwapPoses":
            return [(("piece", self.pid, self.t), "X"), (("piece", self.pid2, self.t), "X")]
        if k in PER_TARGET:
            return [(("piece", self.pid, self.t), "X")]
        if k == "TrimOverlap":  # edits piece geometry -> holds the piece everywhere
            return [(("piece", self.pid, t), "X") for t in range(n_targets)]
        if k == "FusePart":
            return [(("piece", p, t), "X") for p in (self.pid, self.pid2) for t in range(n_targets)]
        return [(("piece", self.pid, t), "X") for t in range(n_targets)]

    def touched_pids(self) -> set[int]:
        return {p for p in (self.pid, self.pid2) if p is not None}

    def __repr__(self) -> str:
        bits = [self.kind, f"p{self.pid}"]
        if self.pid2 is not None:
            bits.append(f"p{self.pid2}")
        if self.t is not None:
            bits.append(f"t{self.t}")
        if self.spec is not None:
            bits.append(str(tuple(int(a) for a in self.spec.args)))
        elif self.args and self.kind != "AddPart":
            bits.append(",".join(f"{a:.3g}" if isinstance(a, float) else str(a) for a in self.args))
        return "Rewrite(" + " ".join(bits) + ")"


def conflicts(a: list[tuple[tuple, str]], b: list[tuple[tuple, str]]) -> bool:
    bm = {}
    for key, mode in b:
        bm[key] = "X" if mode == "X" or bm.get(key) == "X" else "S"
    return any(key in bm and (mode == "X" or bm[key] == "X") for key, mode in a)


# ---------------------------------------------------------------- application


def _cut_shape(shape: Shape, a: int, ta: float, b: int, tb: float) -> tuple[Shape, Shape]:
    """Cut a single CCW loop along the straight chord from point(a, ta) to point(b, tb). Both halves stay CCW."""
    prims = loop_order(shape).primitives
    if a > b:
        a, ta, b, tb = b, tb, a, ta
    assert a != b
    a1, a2 = split_prim_at(prims[a], ta)
    b1, b2 = split_prim_at(prims[b], tb)
    A, B = a2.start, b1.end
    loop1 = [a2, *prims[a + 1 : b], b1, Line(B, A)]
    loop2 = [b2, *prims[b + 1 :], *prims[:a], a1, Line(A, B)]
    return Shape(loop1), Shape(loop2)


def _pose_op(piece: Piece, rw: Rewrite, other: Piece | None = None) -> list[Piece]:
    t = rw.t
    _, c = piece.area_centroid()
    pose = piece.poses[t]
    w = pose.apply(c)
    if rw.kind == "Rotate":
        new = replace(pose, theta=pose.theta + rw.args[0]).with_centroid(c, w).wrapped()
    elif rw.kind == "Flip":
        new = replace(pose, flip=-pose.flip).with_centroid(c, w)
    elif rw.kind == "Relocate":
        new = pose.with_centroid(c, torch.tensor(rw.args[:2], dtype=torch.float32))
    elif rw.kind == "SwapPoses":
        assert other is not None
        _, c2 = other.area_centroid()
        q = other.poses[t]
        w2 = q.apply(c2)
        new = Pose(q.theta, 0.0, 0.0, q.flip).with_centroid(c, w2)
        new2 = Pose(pose.theta, 0.0, 0.0, pose.flip).with_centroid(c2, w)
        o = other.clone()
        o.poses[t] = new2
        p = piece.clone()
        p.poses[t] = new
        return [p, o]
    else:
        raise ValueError(rw.kind)
    p = piece.clone()
    p.poses[t] = new
    return [p]


def apply_rewrite(D: Dissection, rw: Rewrite, new_pid=None) -> tuple[set[int], list[Piece]]:
    """Returns (pids removed, pieces added). Pieces whose pid is kept are returned as removed+added.
    new_pid: callable returning fresh pids (defaults to D.new_pid)."""
    new_pid = new_pid or D.new_pid
    P = D.by_pid()
    k = rw.kind
    if k in GRAMMAR:
        p = P[rw.pid]
        return {p.pid}, [Piece(p.pid, p.shape.apply_rewrite(rw.spec), list(p.poses))]
    if k == "CutPart":
        p = P[rw.pid]
        a, ta, b, tb = rw.args
        s1, s2 = _cut_shape(p.shape, int(a), ta, int(b), tb)
        return {p.pid}, [Piece(new_pid(), s1, list(p.poses)).recentered(), Piece(new_pid(), s2, list(p.poses)).recentered()]
    if k == "AddPart":
        r, n = rw.args[0], int(rw.args[1])
        T = D.n_targets
        poses = [Pose(rw.args[2 + 3 * t], rw.args[3 + 3 * t], rw.args[4 + 3 * t], 1) for t in range(T)]
        return set(), [Piece(new_pid(), Shape(circle_piece_prims(r, n)), poses)]
    if k == "RemoveSmallPart":
        return {rw.pid}, []
    if k == "SwapPoses":
        return {rw.pid, rw.pid2}, _pose_op(P[rw.pid], rw, P[rw.pid2])
    if k in PER_TARGET:
        return {rw.pid}, _pose_op(P[rw.pid], rw)
    raise NotImplementedError(k)


def apply_many(D: Dissection, rws: list[Rewrite]) -> Dissection:
    """Apply a lock-compatible set. Grammar rewrites on the same piece are combined with d4descent's
    ``do_multiple_rewrites`` (they touch disjoint primitives)."""
    D = D.clone()
    grammar: dict[int, list[ShapeRewriteSpec]] = {}
    for rw in rws:
        if rw.kind in GRAMMAR:
            grammar.setdefault(rw.pid, []).append(rw.spec)
    P = D.by_pid()
    for pid, specs in grammar.items():
        p = P[pid]
        p.shape = p.shape.do_multiple_rewrites(specs) if len(specs) > 1 else p.shape.apply_rewrite(specs[0])
    for rw in rws:
        if rw.kind in GRAMMAR:
            continue
        removed, added = apply_rewrite(D, rw)
        D = D.replace_pieces(removed, added)
    return D


# ---------------------------------------------------------------- proposals


@dataclass
class ProposalConfig:
    n_proposals: int = 64
    family_weights: dict[str, float] = field(default_factory=lambda: {
        "grammar": 0.40, "CutPart": 0.15, "AddPart": 0.10, "RemoveSmallPart": 0.05,
        "Rotate": 0.10, "Flip": 0.05, "SwapPoses": 0.10, "Relocate": 0.0,
    })
    allow_flip: bool = False
    lossy_threshold: float = 0.03  # d4descent default (world units)
    cut_tries: int = 20
    cut_t_range: tuple[float, float] = (0.2, 0.8)
    add_radius: float = 0.04
    add_segments: int = 4
    remove_area: float = 0.01  # RemoveSmallPart is proposed for pieces smaller than this (world units^2)
    min_piece_area: float = 1e-3  # halves of a cut must be at least this large
    swap_area_ratio: float = 2.0  # SwapPoses only between pieces within this area ratio
    rotate_angles: tuple[float, ...] = (math.pi / 2, -math.pi / 2, math.pi)


def _piece_polygon(p: Piece) -> shapely.Polygon:
    return shapely.Polygon(sample_boundary(loop_order(p.shape), 16).numpy())


def valid_cut(p: Piece, a: int, ta: float, b: int, tb: float, min_area: float) -> bool:
    prims = loop_order(p.shape).primitives
    if a == b:
        return False
    A = point_on_prim(prims[a], ta).detach().numpy()
    B = point_on_prim(prims[b], tb).detach().numpy()
    if np.linalg.norm(A - B) < 1e-3:
        return False
    poly = _piece_polygon(p)
    if not poly.is_valid:
        return False
    d = B - A
    chord = shapely.LineString([A + 0.02 * d, B - 0.02 * d])
    if not poly.contains(chord):
        return False
    s1, s2 = _cut_shape(p.shape, a, ta, b, tb)
    a1, _ = shape_area_centroid(s1)
    a2, _ = shape_area_centroid(s2)
    return a1 > min_area and a2 > min_area


def propose(D: Dissection, residual_px: list[torch.Tensor], cfg: ProposalConfig, rng: random.Random) -> list[Rewrite]:
    """Sample up to cfg.n_proposals valid rewrites. residual_px[t]: (n, 2) world positions of uncovered target
    pixels in target t (used by AddPart)."""
    T = D.n_targets
    pieces = D.pieces
    areas = {p.pid: p.area_centroid()[0] for p in pieces}
    gargs = ShapeRewriteArgs(add_holes=False, remove_holes=False, lossy_threshold=cfg.lossy_threshold)
    grammar_pool = [
        Rewrite(_GRAMMAR_TYPE[s.type], pid=p.pid, spec=s)
        for p in pieces for s in p.shape.generate_rewrite_specs(gargs) if s.type in _GRAMMAR_TYPE
    ]
    rng.shuffle(grammar_pool)

    def sample_cut() -> Rewrite | None:
        cand = [p for p in pieces if areas[p.pid] > 2 * cfg.min_piece_area and len(p.shape.primitives) >= 2]
        if not cand:
            return None
        for _ in range(cfg.cut_tries):
            p = rng.choices(cand, weights=[areas[q.pid] for q in cand])[0]
            m = len(p.shape.primitives)
            a, b = rng.sample(range(m), 2)
            ta, tb = rng.uniform(*cfg.cut_t_range), rng.uniform(*cfg.cut_t_range)
            if valid_cut(p, a, ta, b, tb, cfg.min_piece_area):
                return Rewrite("CutPart", pid=p.pid, args=(a, ta, b, tb))
        return None

    def sample_add() -> Rewrite | None:
        args: list[float] = [cfg.add_radius, cfg.add_segments]
        for t in range(T):
            if residual_px[t].shape[0] == 0:
                return None
            x, y = residual_px[t][rng.randrange(residual_px[t].shape[0])].tolist()
            args += [rng.uniform(-math.pi, math.pi), x, y]
        return Rewrite("AddPart", args=tuple(args))

    def sample_remove() -> Rewrite | None:
        small = [p for p in pieces if areas[p.pid] < cfg.remove_area]
        return Rewrite("RemoveSmallPart", pid=rng.choice(small).pid) if small else None

    def sample_rotate() -> Rewrite | None:
        if not pieces:
            return None
        ang = rng.choice(list(cfg.rotate_angles) + ["random"])
        ang = rng.uniform(-math.pi, math.pi) if ang == "random" else ang
        return Rewrite("Rotate", pid=rng.choice(pieces).pid, t=rng.randrange(T), args=(ang,))

    def sample_flip() -> Rewrite | None:
        if not cfg.allow_flip or not pieces:
            return None
        return Rewrite("Flip", pid=rng.choice(pieces).pid, t=rng.randrange(T))

    def sample_swap() -> Rewrite | None:
        pairs = [(p, q) for i, p in enumerate(pieces) for q in pieces[i + 1 :]
                 if max(areas[p.pid], areas[q.pid]) <= cfg.swap_area_ratio * max(min(areas[p.pid], areas[q.pid]), 1e-9)]
        if not pairs:
            return None
        p, q = rng.choice(pairs)
        return Rewrite("SwapPoses", pid=p.pid, pid2=q.pid, t=rng.randrange(T))

    def sample_relocate() -> Rewrite | None:
        if not pieces:
            return None
        t = rng.randrange(T)
        if residual_px[t].shape[0] == 0:
            return None
        x, y = residual_px[t][rng.randrange(residual_px[t].shape[0])].tolist()
        return Rewrite("Relocate", pid=rng.choice(pieces).pid, t=t, args=(x, y))

    samplers = {"CutPart": sample_cut, "AddPart": sample_add, "RemoveSmallPart": sample_remove,
                "Rotate": sample_rotate, "Flip": sample_flip, "SwapPoses": sample_swap, "Relocate": sample_relocate}
    fams = [f for f, w in cfg.family_weights.items() if w > 0]
    wts = [cfg.family_weights[f] for f in fams]
    out: list[Rewrite] = []
    seen: set = set()
    dead: set[str] = set()
    g_iter = iter(grammar_pool)
    tries = 0
    while len(out) < cfg.n_proposals and tries < 20 * cfg.n_proposals and len(dead) < len(fams):
        tries += 1
        f = rng.choices(fams, weights=wts)[0]
        if f in dead:
            continue
        if f == "grammar":
            rw = next(g_iter, None)
        else:
            rw = samplers[f]()
        if rw is None:
            if f in ("grammar", "RemoveSmallPart", "SwapPoses", "Flip") or tries > 5 * cfg.n_proposals:
                dead.add(f)
            continue
        key = (rw.kind, rw.pid, rw.pid2, rw.t, rw.args, None if rw.spec is None else (rw.spec.type, rw.spec.args))
        if key in seen:
            continue
        seen.add(key)
        out.append(rw)
    return out


# ---------------------------------------------------------------- repair


@dataclass
class RepairConfig:
    remove_area: float = 2e-3  # pieces smaller than this are dropped (RemoveSmallPart as a repair)


def repair(D: Dissection, cfg: RepairConfig) -> tuple[Dissection, dict[str, int]]:
    """Within-piece repair: resolve self-intersections, canonicalize orientation, split a piece whose outline has
    become several loops into separate pieces (exact: each inherits all poses), drop holes and tiny pieces,
    recentre local frames, wrap angles."""
    stats = {"split_loops": 0, "dropped_holes": 0, "removed_small": 0, "failed": 0}
    out: list[Piece] = []
    for p in D.pieces:
        try:
            s = p.shape.resolve_intersections().canonicalize_loops()
            loops = s.find_loops()
        except Exception:
            stats["failed"] += 1
            out.append(p)
            continue
        kept = []
        for ids, area in sorted(loops, key=lambda x: -x[1]):
            if area <= 0:
                stats["dropped_holes"] += 1
                continue
            if area < cfg.remove_area:
                stats["removed_small"] += 1
                continue
            kept.append(Shape([s.primitives[i] for i in ids]))
        if len(kept) > 1:
            stats["split_loops"] += 1
        for j, shape in enumerate(kept):
            pid = p.pid if j == 0 else D.new_pid()
            out.append(Piece(pid, shape, [q.wrapped() for q in p.poses]).recentered())
    return Dissection(out, D.n_targets, D.next_pid), stats
