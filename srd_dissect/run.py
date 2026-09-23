"""Runner: one SRD dissection run between built-in targets, with logging of seed/config/history and a figure.

  python -m srd_dissect.run --pair square triangle --k 4 --init partition --rounds 200 --seed 0 --out runs/sq_tri
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import random
import time
from pathlib import Path

import torch

from . import targets as TG
from .init import growth_init, partition_init
from .render import render_target_image
from .srd import SRDConfig, run_srd
from .viz import plot_dissection


def save_dissection(D, path):
    torch.save({"n_targets": D.n_targets, "next_pid": D.next_pid,
                "pieces": [{"pid": p.pid, "prims": [(type(q).__name__, q.start.tolist(), q.end.tolist(),
                                                     float(q.k) if hasattr(q, "k") else 0.0)
                                                    for q in p.shape.primitives],
                            "poses": [dataclasses.asdict(x) for x in p.poses]} for p in D.pieces]}, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, default=["square", "triangle"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--init", choices=["partition", "growth"], default="partition")
    ap.add_argument("--fixed-k", action="store_true")
    ap.add_argument("--allow-flip", action="store_true")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)

    cfg = SRDConfig(n_rounds=a.rounds, seed=a.seed, time_budget_s=a.time_budget,
                    fixed_k=a.k if a.fixed_k else None)
    cfg.render.size = a.size
    cfg.proposal.allow_flip = a.allow_flip
    shapes = [TG.get(n) for n in a.pair]
    targets = torch.stack([render_target_image(s, cfg.render) for s in shapes])
    rng = random.Random(a.seed)
    if a.init == "partition":
        D0 = partition_init(shapes[0], targets, a.k, cfg.render, rng)
    else:
        D0 = growth_init(targets, a.k, 0.08, cfg.render, rng)

    meta = {"args": {k: str(v) for k, v in vars(a).items()}, "config": dataclasses.asdict(cfg),
            "hardware": {"platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
                         "torch": torch.__version__}}
    (a.out / "config.json").write_text(json.dumps(meta, indent=1, default=str))
    plot_dissection(D0, targets, cfg.render, a.out / "init.png", "init", a.pair)

    log_f = open(a.out / "log.txt", "w")

    def log(e):
        s = (f"r{e.round} {e.time_s:.0f}s L={e.loss:.4g} cov={[round(c, 4) for c in e.cov]} "
             f"ov={[round(c, 5) for c in e.ov]} P={e.n_pieces} S={e.n_segments} arc={e.arc_fraction:.2f} "
             f"lr={e.lr_scale:.2g} w_ov={e.w_ov:.3g} iou={[round(x, 3) for x in e.stats.get('iou', [])]} "
             f"acc={len(e.accepted)}")
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    t0 = time.time()
    best, L, hist = run_srd(D0, targets, cfg, log)
    hist["wall_s"] = time.time() - t0
    hist["best_loss"] = L
    (a.out / "history.json").write_text(json.dumps(hist, default=str))
    save_dissection(best, a.out / "best.pt")
    plot_dissection(best, targets, cfg.render, a.out / "best.png",
                    f"{a.pair[0]} <-> {a.pair[1]}  k={a.k} init={a.init} seed={a.seed}  "
                    f"L={L:.4g}  pieces={len(best.pieces)}", a.pair)
    print(json.dumps(hist["accept"], indent=1))


if __name__ == "__main__":
    main()
