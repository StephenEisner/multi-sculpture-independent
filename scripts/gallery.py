"""Gallery of the kept restart for every benchmark row: targets (grey) with the cleaned arrangement on top,
pieces coloured consistently across the two targets.  PYTHONPATH=. uv run python scripts/gallery.py"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from eval.geometry import local_polygons, shape_polygon
from eval.validity import remove_overlaps, world_polys
from srd_dissect import targets as TG
from srd_dissect.run import get_shape, load_dissection

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = list(csv.DictReader(open(ROOT / "results" / "table1_standin.csv")))
    fig, axes = plt.subplots(len(rows), 2, figsize=(5.2, 2.6 * len(rows)), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(rows):
        run = ROOT / "runs" / "bench" / r["name"] / f"seed{r['best_seed']}"
        cfg = json.loads((run / "config.json").read_text())
        pair = eval(cfg["args"]["pair"])
        D = load_dissection(run / "best.pt")
        shapes = TG.fit_pair([get_shape(n) for n in pair])
        clean = remove_overlaps(local_polygons(D), D)
        col = {pid: cmap(j % 10) for j, pid in enumerate(sorted(clean))}
        for t in range(2):
            ax = axes[i, t]
            tg = shape_polygon(shapes[t])
            ax.add_patch(MplPolygon(list(tg.exterior.coords), closed=True, fc="0.8", ec="k", lw=0.8))
            for pid, g in world_polys(clean, D, t).items():
                for gg in getattr(g, "geoms", [g]):
                    if gg.geom_type == "Polygon":
                        ax.add_patch(MplPolygon(list(gg.exterior.coords), closed=True, fc=col[pid], ec="k", lw=0.4,
                                                alpha=0.9))
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(1.5, -1.5)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{pair[t].replace('mdi:', '')}" + (f"   [{r['name']}: Ch {float(r['avg_chamfer']):.2f}, "
                         f"Hd {float(r['avg_hausdorff']):.1f}]" if t == 0 else ""), fontsize=7, loc="left")
    fig.suptitle("Stand-in inputs (MDI icons), cleaned dissections; not comparable to Voronoi Scissors", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "gallery.png", dpi=100)


if __name__ == "__main__":
    main()
