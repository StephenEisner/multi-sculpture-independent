"""Build results/table1.csv and results/table1.md from results/table1_standin.csv, with Voronoi Scissors Table 1
alongside. All our rows are on stand-in inputs and are NOT comparable to Voronoi Scissors."""

import csv
from pathlib import Path

VS = {  # Voronoi Scissors Table 1: (avg Chamfer, avg Hausdorff)
    ("dog-bone", 4): (2.54, 8.89), ("dog-bone", 5): (3.85, 13.69), ("trump-map", 5): (3.17, 11.18),
    ("bunny-egg", 6): (3.26, 9.74), ("caterpillar-butterfly", 6): (4.61, 13.71), ("cat-bear", 6): (2.81, 8.92),
    ("serpent-apple", 5): (6.60, 29.36),
}

ROOT = Path(__file__).resolve().parents[1]


def main():
    src = ROOT / "results" / "table1_standin.csv"
    rows = list(csv.DictReader(open(src)))
    out = []
    for r in rows:
        name = r["name"]
        flip = name.endswith("_flip")
        base = name.replace("_flip", "")
        pair, k = base.rsplit("_k", 1)
        vs = VS.get((pair, int(k)), (None, None))
        out.append({
            "pair": pair, "k": k, "reflections": flip, "inputs": "stand-in (MDI), not comparable",
            "chamfer_A": r["chamfer_A"], "chamfer_B": r["chamfer_B"], "avg_chamfer": r["avg_chamfer"],
            "hausdorff_A": r["hausdorff_A"], "hausdorff_B": r["hausdorff_B"], "avg_hausdorff": r["avg_hausdorff"],
            "vs_avg_chamfer": vs[0], "vs_avg_hausdorff": vs[1],
            "raw_overlap_px2": r["overlap_raw_px2"], "clean_gap_px2": r["gap_clean_px2"],
            "clean_uncovered_px2": r["uncovered_clean_px2"], "clean_overhang_px2": r["overhang_clean_px2"],
            "pieces": r["pieces_clean"], "simple_connected": r["simple_connected"],
            "runtime_s_per_restart": r["budget_s"], "restarts": r["restarts"], "seed": r["best_seed"],
            "median_restart_chamfer": r["median_restart_chamfer"], "config": r["config"],
        })
    with open(ROOT / "results" / "table1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    f2 = lambda x: "—" if x in (None, "") else f"{float(x):.2f}"
    lines = ["| Pair | k | Reflections | Avg Chamfer | Avg Hausdorff | VS Chamfer | VS Hausdorff | Uncovered px² (A+B) | "
             "Overhang px² | Gap px² | Raw overlap px² | Pieces | Seed |", "|" + "---|" * 13]
    for r in out:
        lines.append(f"| {r['pair']} | {r['k']} | {'yes' if r['reflections'] else 'no'} | {f2(r['avg_chamfer'])} | "
                     f"{f2(r['avg_hausdorff'])} | {f2(r['vs_avg_chamfer'])} | {f2(r['vs_avg_hausdorff'])} | "
                     f"{float(r['clean_uncovered_px2']):.0f} | {float(r['clean_overhang_px2']):.0f} | "
                     f"{float(r['clean_gap_px2']):.0f} | {float(r['raw_overlap_px2']):.1f} | {r['pieces']} | {r['seed']} |")
    (ROOT / "results" / "table1.md").write_text(
        "Stand-in inputs (Material Design Icons silhouettes), NOT comparable to Voronoi Scissors Table 1 (VS columns "
        "are their numbers on the Duncan et al. inputs). Our numbers: cleaned (overlap-free) dissection, "
        "Chamfer/Hausdorff in px at 10,000 px^2 per shape.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
