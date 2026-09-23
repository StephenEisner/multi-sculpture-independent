"""Stochastic Rewrite Descent over a multi-target dissection.

Each round: continuous phase -> repair -> propose K rewrites -> score each (one local step) -> greedy apply.

Scoring baseline (see DEVIATIONS.md). d4descent scores every proposal against the unmodified object after the *same*
local step. Here a proposal only re-renders and locally steps the pieces it touches (the rest of the arrangement is
held fixed from a cache), so the matching baseline for a proposal that touches pieces S is the current state with
exactly the pieces in S stepped once. Baselines are computed once per distinct S and shared. Proposals that touch no
existing piece (AddPart) are compared with the unstepped current loss.
"""

from __future__ import annotations

import copy
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from .loss import LossConfig, dissection_loss, image_terms, piece_regularizers, simplicity
from .render import Packed, RenderConfig
from .rewrites import ProposalConfig, RepairConfig, Rewrite, apply_many, apply_rewrite, conflicts, propose, repair
from .state import Dissection, Piece

PARAM_GROUPS = ("vertices", "bulges", "rotation", "translation")


@dataclass
class SRDConfig:
    n_rounds: int = 200
    steps_per_round: int = 25  # d4descent AL-F: propose every 25 steps
    optimizer: str = "sgd"  # "sgd" (d4descent default) or "adam"
    lr: dict[str, float] = field(default_factory=lambda: {
        "vertices": 0.5, "bulges": 0.5, "rotation": 1.0, "translation": 0.5})
    adam_lr: dict[str, float] = field(default_factory=lambda: {
        "vertices": 2e-3, "bulges": 2e-3, "rotation": 1e-2, "translation": 2e-3})
    clip_grad: float = 2.0
    # AdaptiveLR (d4descent): a global multiplier on all groups
    lr_factor: float = 0.5
    lr_reduce_patience: int = 2
    lr_increase_patience: int = 2
    lr_min_scale: float = 2e-4
    local_steps: int = 1
    better_abs_eps: float = 1e-8
    w_ov_start: float = 0.1
    w_ov_end: float = 10.0
    max_batch_prims: int = 800
    stopping_patience: int | None = 25  # rounds without improving the best loss
    time_budget_s: float | None = None
    fixed_k: int | None = None  # hold the piece count fixed (disables CutPart/AddPart/RemoveSmallPart)
    seed: int = 0
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    render: RenderConfig = field(default_factory=RenderConfig)


# ---------------------------------------------------------------- optimizer helpers


def _make_opt(pk: Packed, cfg: SRDConfig, scale: float) -> torch.optim.Optimizer:
    lrs = cfg.lr if cfg.optimizer == "sgd" else cfg.adam_lr
    groups = [{"params": [pk.param_groups()[g]], "lr": lrs[g] * scale, "name": g} for g in PARAM_GROUPS]
    return torch.optim.SGD(groups) if cfg.optimizer == "sgd" else torch.optim.Adam(groups)


def _clip(pk: Packed, cfg: SRDConfig):
    if cfg.clip_grad is not None:
        torch.nn.utils.clip_grad_value_(list(pk.param_groups().values()), cfg.clip_grad)


class AdaptiveScale:
    """d4descent's AdaptiveLRScheduler logic, as one multiplier shared by all parameter groups."""

    def __init__(self, cfg: SRDConfig):
        self.cfg = cfg
        self.scale = 1.0
        self.best = float("inf")
        self.bad = 0
        self.good = 0

    def new_round(self):
        """d4descent recreates its scheduler (keeping the current lr) after every rewrite round, so the plateau
        tracker starts fresh: the loss legitimately jumps across rounds (rewrites, w_ov annealing)."""
        self.best, self.bad, self.good = float("inf"), 0, 0

    def step(self, loss: float):
        c = self.cfg
        self.good = self.good + 1 if not loss > self.best * (1 + 1e-4) else 0
        if loss < self.best * (1 - 1e-4):
            self.best, self.bad = loss, 0
        else:
            self.bad += 1
        if self.bad > c.lr_reduce_patience:
            self.scale = max(self.scale * c.lr_factor, c.lr_min_scale)
            self.bad = self.good = 0
        if self.good > c.lr_increase_patience:
            self.scale = min(self.scale * (1 / c.lr_factor) ** 0.5, 1.0)
            self.bad = self.good = 0


# ---------------------------------------------------------------- scoring


@dataclass
class _Variant:
    removed: frozenset[int]
    added: list[Piece]
    loss: float = float("nan")  # after local step, incl. simplicity


def _occ_cache(D: Dissection, cfg: SRDConfig, T: int):
    pk = Packed.build(D.pieces, cfg.render, requires_grad=False)
    with torch.no_grad():
        occ = torch.stack([pk.occupancy(t) for t in range(T)], 1)  # (P, T, H, W)
        reg = piece_regularizers(pk, cfg.loss)  # (P,)
    row = {pid: i for i, pid in enumerate(pk.pids)}
    return occ, reg, row


def _eval_variants(variants: list[_Variant], D: Dissection, targets: torch.Tensor, cfg: SRDConfig, lr_scale: float,
                   occ: torch.Tensor, reg: torch.Tensor, row: dict[int, int]) -> None:
    """Fill v.loss for each variant: loss (all targets, incl. simplicity) after cfg.local_steps local steps on the
    variant's added pieces, with every other piece held fixed at its cached occupancy."""
    T = targets.shape[0]
    base_sum = occ.sum(0)  # (T, H, W)
    reg_total = reg.sum()
    n_pieces = len(D.pieces)
    n_segs = D.n_segments()
    P = D.by_pid()
    lc = cfg.loss

    def const_part(v: _Variant):
        rows = [row[p] for p in v.removed]
        rem = occ[rows].sum(0) if rows else torch.zeros_like(base_sum)
        g = simplicity(n_pieces - len(v.removed) + len(v.added),
                       n_segs - sum(P[p].n_segments() for p in v.removed) + sum(a.n_segments() for a in v.added), lc)
        return base_sum - rem, reg_total - (reg[rows].sum() if rows else 0.0), g

    # variants without pieces to optimize
    for v in variants:
        if not v.added:
            s, r, g = const_part(v)
            cov, ov = image_terms(s[None], targets)
            v.loss = float((lc.w_cov * cov + lc.w_ov * ov).sum() + r) + g

    todo = [v for v in variants if v.added]
    chunks, cur, n = [], [], 0
    for v in todo:
        m = sum(len(a.shape.primitives) for a in v.added)
        if cur and n + m > cfg.max_batch_prims:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(v)
        n += m
    if cur:
        chunks.append(cur)

    for chunk in chunks:
        V = len(chunk)
        pieces = [a for v in chunk for a in v.added]
        vid = torch.tensor([i for i, v in enumerate(chunk) for _ in v.added])
        consts = [const_part(v) for v in chunk]
        rest = torch.stack([c[0] for c in consts])  # (V, T, H, W)
        rest_reg = torch.tensor([float(c[1]) for c in consts])
        gs = torch.tensor([c[2] for c in consts])
        pk = Packed.build(pieces, cfg.render)
        opt = _make_opt(pk, cfg, lr_scale)

        def losses() -> torch.Tensor:
            new = torch.stack([torch.zeros(V, *targets.shape[1:]).index_add(0, vid, pk.occupancy(t))
                               for t in range(T)], 1)  # (V, T, H, W)
            cov, ov = image_terms(rest + new, targets)
            r = torch.zeros(V).index_add(0, vid, piece_regularizers(pk, lc))
            return (lc.w_cov * cov + lc.w_ov * ov).sum(-1) + rest_reg + r  # (V,)

        for _ in range(cfg.local_steps):
            opt.zero_grad()
            losses().sum().backward()
            _clip(pk, cfg)
            opt.step()
        with torch.no_grad():
            Ls = losses() + gs
        for v, L in zip(chunk, Ls.tolist()):
            v.loss = L


class _PidGen:
    def __init__(self, start: int):
        self.n = start

    def __call__(self) -> int:
        self.n += 1
        return self.n


def score_rewrites(D: Dissection, rws: list[Rewrite], targets: torch.Tensor, cfg: SRDConfig, lr_scale: float
                   ) -> tuple[list[float], list[Rewrite]]:
    """Returns (delta L per rewrite, rewrites that failed to apply). delta L > 0 means improvement over the
    baseline 'same pieces, same local step'."""
    T = targets.shape[0]
    occ, reg, row = _occ_cache(D, cfg, T)
    P = D.by_pid()
    gen = _PidGen(10**9)  # placeholder pids for pieces created during scoring
    props: list[_Variant | None] = []
    failed = []
    for rw in rws:
        try:
            removed, added = apply_rewrite(D, rw, new_pid=gen)
            props.append(_Variant(frozenset(removed), added))
        except Exception:
            props.append(None)
            failed.append(rw)
    baselines: dict[frozenset[int], _Variant] = {}
    for v in props:
        if v is not None and v.removed not in baselines:
            baselines[v.removed] = _Variant(v.removed, [P[p].clone() for p in v.removed])
    _eval_variants([v for v in props if v is not None] + list(baselines.values()), D, targets, cfg, lr_scale,
                   occ, reg, row)
    deltas = [float("-inf") if v is None else baselines[v.removed].loss - v.loss for v in props]
    return deltas, failed


def select_greedy(rws: list[Rewrite], deltas: list[float], n_targets: int, eps: float) -> list[Rewrite]:
    order = sorted([i for i, d in enumerate(deltas) if d > eps], key=lambda i: -deltas[i])
    chosen: list[Rewrite] = []
    held: list[tuple[tuple, str]] = []
    for i in order:
        lk = rws[i].locks(n_targets)
        if conflicts(lk, held):
            continue
        chosen.append(rws[i])
        held += lk
    return chosen


# ---------------------------------------------------------------- diagnostics


@torch.no_grad()
def arrangement_stats(D: Dissection, targets: torch.Tensor, cfg: RenderConfig) -> dict[str, list[float]]:
    pk = Packed.build(D.pieces, cfg, requires_grad=False)
    px2 = cfg.pixel**2
    out = defaultdict(list)
    for t in range(targets.shape[0]):
        s = pk.occupancy(t).sum(0)
        U, Tg = s > 0.5, targets[t] > 0.5
        out["iou"].append(float((U & Tg).sum() / (U | Tg).sum().clamp(min=1)))
        out["overlap_area"].append(float(torch.relu(s - 1).sum() * px2))
        out["uncovered_area"].append(float((Tg & ~U).sum() * px2))
        out["overhang_area"].append(float((U & ~Tg).sum() * px2))
    return dict(out)


def gradient_conflict(D: Dissection, targets: torch.Tensor, cfg: SRDConfig) -> list[float]:
    """Per piece: cosine between the target-0 and target-1 gradients on the shared geometry parameters."""
    if targets.shape[0] < 2:
        return []
    pk = Packed.build(D.pieces, cfg.render)
    lc = copy.copy(cfg.loss)
    grads = []
    occ = [pk.occupancy(t) for t in range(2)]
    for t in range(2):
        cov, ov = image_terms(occ[t].sum(0)[None, None], targets[t : t + 1])
        L = (lc.w_cov * cov + lc.w_ov * ov).sum()
        gcp, gk = torch.autograd.grad(L, [pk.sc.control_points, pk.sc.ks], retain_graph=True, allow_unused=True)
        grads.append((gcp if gcp is not None else torch.zeros_like(pk.sc.control_points),
                      gk if gk is not None else torch.zeros_like(pk.sc.ks)))
    cos = []
    for i in range(pk.n_pieces):
        m, mk = pk.pt_piece == i, pk.k_piece == i
        a = torch.cat([grads[0][0][m].flatten(), grads[0][1][mk]])
        b = torch.cat([grads[1][0][m].flatten(), grads[1][1][mk]])
        na, nb = a.norm(), b.norm()
        cos.append(float((a @ b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else float("nan"))
    return cos


# ---------------------------------------------------------------- main loop


@dataclass
class RoundLog:
    round: int
    time_s: float
    loss: float  # final objective (w_ov = w_ov_end), after repair
    cov: list[float]
    ov: list[float]
    n_pieces: int
    n_segments: int
    arc_fraction: float
    lr_scale: float
    w_ov: float
    n_proposals: int
    accepted: list[str]
    stats: dict[str, list[float]]
    grad_conflict: list[float]
    repair: dict[str, int]


def _arc_fraction(D: Dissection) -> float:
    from d4descent.objects.arclines import Arc

    n = sum(len(p.shape.primitives) for p in D.pieces)
    a = sum(isinstance(q, Arc) and abs(float(q.k)) > 1e-6 for p in D.pieces for q in p.shape.primitives)
    return a / max(n, 1)


def run_srd(D: Dissection, targets: torch.Tensor, cfg: SRDConfig, log_fn=None) -> tuple[Dissection, float, dict]:
    """Optimize D towards targets (T, H, W). Returns (best dissection, its loss, history)."""
    cfg = copy.deepcopy(cfg)  # w_ov is annealed in place
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    T = targets.shape[0]
    pcfg = copy.deepcopy(cfg.proposal)
    if cfg.fixed_k is not None:
        for f in ("CutPart", "AddPart", "RemoveSmallPart"):
            pcfg.family_weights[f] = 0.0
    sched = AdaptiveScale(cfg)
    accept = defaultdict(lambda: [0, 0])  # kind -> [proposed, accepted]
    history: list[RoundLog] = []
    best = (float("inf"), D.clone())
    stall = 0
    t0 = time.time()
    grid = cfg.render.grid()

    for r in range(cfg.n_rounds):
        frac = r / max(cfg.n_rounds - 1, 1)
        cfg.loss.w_ov = cfg.w_ov_start * (cfg.w_ov_end / cfg.w_ov_start) ** frac

        # 1. continuous phase
        conflict = gradient_conflict(D, targets, cfg) if D.pieces else []
        if D.pieces:
            pk = Packed.build(D.pieces, cfg.render)
            opt = _make_opt(pk, cfg, sched.scale)
            sched.new_round()
            for _ in range(cfg.steps_per_round):
                opt.zero_grad()
                L, bd = dissection_loss(pk, targets, cfg.loss, D.n_segments())
                L.backward()
                _clip(pk, cfg)
                opt.step()
                sched.step(bd.total)
                for g in opt.param_groups:
                    g["lr"] = (cfg.lr if cfg.optimizer == "sgd" else cfg.adam_lr)[g["name"]] * sched.scale
            D = Dissection(pk.unpack(D.pieces), T, D.next_pid)

        # 2. repair
        D, rstats = repair(D, cfg.repair)

        # track best (after repair, before rewrites) under the *final* objective (w_ov = w_ov_end), so rounds
        # with different annealed w_ov are comparable
        eval_loss = copy.copy(cfg.loss)
        eval_loss.w_ov = cfg.w_ov_end
        with torch.no_grad():
            if D.pieces:
                pk0 = Packed.build(D.pieces, cfg.render, requires_grad=False)
                _, cur = dissection_loss(pk0, targets, eval_loss, D.n_segments())
            else:
                cov, ov = image_terms(torch.zeros(1, *targets.shape), targets)
                cur = type("B", (), {"total": float(cov.sum()), "cov": cov[0].tolist(), "ov": [0.0] * T})()
        if cur.total < best[0] * (1 - 1e-4):
            best, stall = (cur.total, D.clone()), 0
        else:
            stall += 1

        # 3. propose
        with torch.no_grad():
            if D.pieces:
                s = torch.stack([Packed.build(D.pieces, cfg.render, requires_grad=False).occupancy(t).sum(0)
                                 for t in range(T)])
            else:
                s = torch.zeros_like(targets)
            residual = [grid[(targets[t] - s[t].clamp(0, 1)) > 0.5] for t in range(T)]
        rws = propose(D, residual, pcfg, rng)

        # 4. score, 5. apply
        chosen: list[Rewrite] = []
        if rws:
            deltas, _ = score_rewrites(D, rws, targets, cfg, sched.scale) if D.pieces else _score_empty(D, rws, targets, cfg)
            chosen = select_greedy(rws, deltas, T, cfg.better_abs_eps)
            for rw in rws:
                accept[rw.kind][0] += 1
            for rw in chosen:
                accept[rw.kind][1] += 1
            if chosen:
                D = apply_many(D, chosen)

        entry = RoundLog(r, time.time() - t0, cur.total, list(cur.cov), list(cur.ov), len(D.pieces), D.n_segments(),
                         _arc_fraction(D), sched.scale, cfg.loss.w_ov, len(rws), [repr(c) for c in chosen],
                         arrangement_stats(D, targets, cfg.render) if D.pieces else {}, conflict, rstats)
        history.append(entry)
        if log_fn:
            log_fn(entry)
        if cfg.stopping_patience is not None and stall >= cfg.stopping_patience:
            break
        if cfg.time_budget_s is not None and time.time() - t0 > cfg.time_budget_s:
            break

    hist = {"rounds": [e.__dict__ for e in history],
            "accept": {k: {"proposed": v[0], "accepted": v[1],
                           "scope": Rewrite(k).scope if k else "", "rate": v[1] / max(v[0], 1)}
                       for k, v in accept.items()}}
    return best[1], best[0], hist


def _score_empty(D: Dissection, rws: list[Rewrite], targets: torch.Tensor, cfg: SRDConfig):
    """Scoring when there are no pieces yet (only AddPart applies): baseline is the empty arrangement."""
    cov, ov = image_terms(torch.zeros(1, *targets.shape), targets)
    base = float((cfg.loss.w_cov * cov).sum())
    gen = _PidGen(10**9)
    vs = []
    for rw in rws:
        _, added = apply_rewrite(D, rw, new_pid=gen)
        vs.append(_Variant(frozenset(), added))
    occ = torch.zeros(0, *targets.shape)
    _eval_variants(vs, D, targets, cfg, 1.0, occ, torch.zeros(0), {})
    return [base - v.loss for v in vs], []
