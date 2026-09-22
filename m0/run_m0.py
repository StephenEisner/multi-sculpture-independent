"""M0: reproduce d4descent Arc-Line single-image fitting (AL-F Full) on a subset of its benchmark.

Runs the unmodified `scripts/optimize_shc.py` from d4descent with the exact flags produced by
`runs/_rungen_arclines.py` (AL-F, AdaptiveLR, w=1e-5), one process per shape, and aggregates
PSNR / primitive count / wall time. Only deviation from the generated scripts: batch_param_count
is lowered (memory only -- batches are evaluated independently, so results are unaffected) and
each process is pinned to a few threads so several shapes can run in parallel on CPU.

Usage:
  python m0/run_m0.py run  --d4d ../d4descent --n-one 8 --n-donut 3 --n-two 3 --jobs 4 --seed 0
  python m0/run_m0.py aggregate --d4d ../d4descent
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASETS = {
    "OneComp": ("data/arclines/bench128.shc", 128),
    "Donut": ("data/arclines/donut25.shc", 25),
    "TwoComp": ("data/arclines/twocomp23.shc", 23),
}

# Reference values: Kodnongbua et al. 2025, Tables 2/3, row "AL-F Full".
PAPER = {
    "OneComp": {"psnr": 44.3, "prims": 9, "time_s": 80},
    "Donut": {"psnr": 48.1, "prims": 10, "time_s": 106},
    "TwoComp": {"psnr": 49.3, "prims": 11, "time_s": 107},
}

ALF_FLAGS = [
    "--render.blur", str(1 / math.sqrt(2)),
    "---loss", "configs/losses/raster.yaml",
    "---task", "configs/tasks/arclines.yaml",
    "--task.rewrite_args.add_hole_random", "True",
    "--task.arclines_args.ks_scale", str(1 / math.sqrt(2)),
    "--optim.proposal_trigger", "step",
    "--optim.propose_every", "25",
    "--optim.proposal_size", "64",
    "--optim.n_steps", "5000",
    "--optim.stopping_patience", "25",
    "--optim.proposal_criterion", "loss",
    "--optim.proposal_steps", "1",
    "--restart", "False",
    "--optim.scheduler", "AdaptiveLR",
    "--optim.lr", "0.5",
    "--task.line_weight", "1e-05",
    "--task.arc_weight", "1e-05",
]

OUT_ROOT = "output/M0"


def pick_indices(seed: int, counts: dict[str, int]) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    jobs = []
    for name, n in counts.items():
        total = DATASETS[name][1]
        for i in sorted(rng.sample(range(total), min(n, total))):
            jobs.append((name, i))
    return jobs


def run_one(d4d: Path, name: str, idx: int, threads: int, batch_param_count: int) -> tuple[str, int, int]:
    save = f"{OUT_ROOT}/{name}/{idx:03d}"
    log = d4d / save / "run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "python", "scripts/optimize_shc.py",
        "--save_path", save,
        *ALF_FLAGS,
        "--optim.batch_param_count", str(batch_param_count),
        "--target_points_path", DATASETS[name][0],
        "--skip", str(idx), "--until", str(idx + 1),
    ]
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=d4d, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    # optimize_shc.py saves all results before writing video.mp4; without ffmpeg only that last step fails.
    print(f"[{name} {idx}] exit {rc}", flush=True)
    return name, idx, rc


def aggregate(d4d: Path, out_csv: Path) -> None:
    code = r"""
import json, math, sys, torch
from pathlib import Path
from d4descent.util import torch_load
from d4descent.objects.arclines import ShapeCollection
root = Path(sys.argv[1]); rows = []
for mp in sorted(root.glob('*/*/*/metrics.pt')):
    ds, idx, sid = mp.parts[-4], mp.parts[-3], mp.parts[-2]
    m = torch.load(mp, weights_only=False)
    top = torch_load(mp.parent / 'topshape.objc', ShapeCollection)[0]
    mse = m['$loss_cont'][-1]
    res = torch.load(mp.parent / 'results.bsr', weights_only=False)
    summ = res[2].summarize(0.8)
    mse_best = min(m['$loss_cont'])
    rows.append(dict(dataset=ds, index=int(idx), shape_id=sid, mse=mse, psnr=10*math.log10(1/max(mse,1e-12)),
                     mse_best=mse_best,
                     prims=len(top.primitives), steps=len(m['$loss_cont']), time_s=m['$timestamp'][-1],
                     pct_matches=summ.pct_matches, pct_perfect=summ.pct_perfect_matches))
print(json.dumps(rows))
"""
    out = subprocess.run(
        ["uv", "run", "python", "-c", code, str(d4d / OUT_ROOT)], cwd=d4d, capture_output=True, text=True, check=True
    ).stdout
    rows = json.loads(out.strip().splitlines()[-1])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # PSNR of the dataset-mean MSE (exact fits give MSE ~1e-14, so a mean of per-shape PSNRs is meaningless);
    # also the median per-shape PSNR capped at 60 dB, and d4descent's own primitive-recovery metric.
    # d4descent's optimize() returns the *final* state, not the best; "best" is min over the trajectory.
    hdr = f"{'dataset':9s} {'n':>3s} {'PSNR(meanMSE)':>13s} {'best':>5s} {'paper':>6s} {'medPSNR<=60':>11s} {'#prim':>6s} {'paper':>6s} {'match%':>7s} {'time_s':>7s} {'paper':>6s}"
    print(hdr)
    for ds in DATASETS:
        rs = [r for r in rows if r["dataset"] == ds]
        if not rs:
            continue
        mean = lambda k: sum(r[k] for r in rs) / len(rs)
        med = sorted(min(r["psnr"], 60.0) for r in rs)[len(rs) // 2]
        p = PAPER[ds]
        print(f"{ds:9s} {len(rs):3d} {10 * math.log10(1 / mean('mse')):13.1f} {10 * math.log10(1 / mean('mse_best')):5.1f} {p['psnr']:6.1f} {med:11.1f} "
              f"{mean('prims'):6.1f} {p['prims']:6d} {100 * mean('pct_matches'):7.0f} {mean('time_s'):7.0f} {p['time_s']:6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "aggregate"])
    ap.add_argument("--d4d", type=Path, default=Path(__file__).resolve().parents[2] / "d4descent")
    ap.add_argument("--n-one", type=int, default=8)
    ap.add_argument("--n-donut", type=int, default=3)
    ap.add_argument("--n-two", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch-param-count", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "m0_results.csv")
    a = ap.parse_args()
    d4d = a.d4d.resolve()
    if a.cmd == "run":
        jobs = pick_indices(a.seed, {"OneComp": a.n_one, "Donut": a.n_donut, "TwoComp": a.n_two})
        print("jobs:", jobs, flush=True)
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(lambda j: run_one(d4d, j[0], j[1], a.threads, a.batch_param_count), jobs))
    aggregate(d4d, a.out)


if __name__ == "__main__":
    sys.exit(main())
