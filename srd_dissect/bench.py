"""Restart driver: N independent restarts of one configuration in parallel (one process per restart), each with a
wall-clock budget; the restart with the smallest final objective ("average L2" in Voronoi Scissors' protocol) is
kept. Appends one row per configuration to a CSV.

  python -m srd_dissect.bench --name sq-tri --pair square triangle --k 4 --restarts 4 --jobs 4 \\
      --time-budget 900 --out runs/m3 --csv results/m3.csv -- --init partition --mode fixed
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pair", nargs=2, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--time-budget", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = [x for x in a.extra if x != "--"]
    root = a.out / a.name
    root.mkdir(parents=True, exist_ok=True)

    def one(seed):
        out = root / f"seed{seed}"
        cmd = [sys.executable, "-m", "srd_dissect.run", "--pair", *a.pair, "--k", str(a.k), "--seed", str(seed),
               "--time-budget", str(a.time_budget), "--rounds", "100000", "--threads", "1", "--out", str(out), *extra]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        with open(root / f"seed{seed}.log", "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        ev = out / "eval.json"
        return seed, rc, (json.loads(ev.read_text()) if ev.exists() else None)

    seeds = list(range(a.seed0, a.seed0 + a.restarts))
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(one, seeds))
    ok = [(s, e) for s, rc, e in results if e is not None]
    if not ok:
        print("all restarts failed", results)
        return 1
    # prefer restarts that reached exactly k pieces, then the smallest final objective
    s_best, e_best = min(ok, key=lambda x: (not x[1].get("reached_k", True), x[1]["best_loss"]))
    ch = [e["avg_chamfer"] for _, e in ok]
    row = {
        "name": a.name, "label": a.label, "pair": "-".join(a.pair), "k": a.k, "config": " ".join(extra),
        "restarts": len(ok), "budget_s": a.time_budget, "best_seed": s_best,
        "pieces": e_best["n_pieces"], "pieces_clean": e_best["n_pieces_clean"], "reached_k": e_best.get("reached_k"),
        "chamfer_A": e_best["clean"]["scores"][0]["chamfer"], "chamfer_B": e_best["clean"]["scores"][1]["chamfer"],
        "hausdorff_A": e_best["clean"]["scores"][0]["hausdorff"], "hausdorff_B": e_best["clean"]["scores"][1]["hausdorff"],
        "avg_chamfer": e_best["avg_chamfer"], "avg_hausdorff": e_best["avg_hausdorff"],
        "avg_chamfer_raw": e_best["avg_chamfer_raw"], "avg_hausdorff_raw": e_best["avg_hausdorff_raw"],
        "overlap_raw_px2": sum(x["overlap_area_px2"] for x in e_best["raw"]["arrangements"]),
        "gap_clean_px2": sum(x["internal_gap_area_px2"] for x in e_best["clean"]["arrangements"]),
        "uncovered_clean_px2": sum(x["uncovered_area_px2"] for x in e_best["clean"]["arrangements"]),
        "overhang_clean_px2": sum(x["overhang_area_px2"] for x in e_best["clean"]["arrangements"]),
        "simple_connected": e_best["clean"]["all_pieces_simple_connected"],
        "best_loss": e_best["best_loss"], "median_restart_chamfer": sorted(ch)[len(ch) // 2],
        "all_restart_chamfer": ";".join(f"{c:.2f}" for c in ch),
    }
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    new = not a.csv.exists()
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
