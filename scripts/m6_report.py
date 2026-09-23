"""M6 report for dog <-> duck: IoU per target, IoU in the disagreement regions, overlap/gap, pieces, Chamfer/Hausdorff.

  PYTHONPATH=. uv run python scripts/m6_report.py runs/bench/dog-duck_k6_flip
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shapely
from matplotlib.patches import Polygon as MplPolygon

from eval.geometry import local_polygons, shape_polygon
from eval.regions import iou, region_geoms, region_iou
from eval.validity import remove_overlaps, world_polys
from srd_dissect import targets as TG
from srd_dissect.run import get_shape, load_dissection

ICONS = ["dog-side", "duck"]


def main(root: Path):
    seeds = sorted(root.glob("seed*/eval.json"))
    best = min(seeds, key=lambda p: (not json.loads(p.read_text()).get("reached_k", True),
                                     json.loads(p.read_text())["best_loss"]))
    run = best.parent
    ev = json.loads(best.read_text())
    D = load_dissection(run / "best.pt")
    shapes = TG.fit_pair([get_shape("mdi:" + n) for n in ICONS])
    tgs = [shape_polygon(s) for s in shapes]
    out = {"run": str(run), "n_pieces": ev["n_pieces"], "avg_chamfer": ev["avg_chamfer"],
           "avg_hausdorff": ev["avg_hausdorff"], "targets": {}}
    for tag, local in (("raw", local_polygons(D)), ("clean", None)):
        if local is None:
            local = remove_overlaps(local_polygons(D), D)
        for t, (icon, tg) in enumerate(zip(ICONS, tgs)):
            U = shapely.unary_union(list(world_polys(local, D, t).values()))
            d = out["targets"].setdefault(icon, {})
            d[f"iou_{tag}"] = iou(U, tg)
            for name, R in region_geoms(icon, tg).items():
                d[f"iou_{name}_{tag}"] = region_iou(U, tg, R)
            arr = ev[tag]["arrangements"][t]
            d[f"overlap_px2_{tag}"] = arr["overlap_area_px2"]
            d[f"gap_px2_{tag}"] = arr["internal_gap_area_px2"]
            d[f"uncovered_px2_{tag}"] = arr["uncovered_area_px2"]
            d[f"overhang_px2_{tag}"] = arr["overhang_area_px2"]
            d["chamfer"] = ev["clean"]["scores"][t]["chamfer"]
            d["hausdorff"] = ev["clean"]["scores"][t]["hausdorff"]
    flips = sum(q.flip < 0 for p in D.pieces for q in p.poses)
    out["n_flipped_poses"] = int(flips)
    (root / "m6_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    # figure with the regions outlined
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    clean = remove_overlaps(local_polygons(D), D)
    cmap = plt.get_cmap("tab20")
    col = {pid: cmap(i % 20) for i, pid in enumerate(sorted(clean))}
    for t, (ax, icon, tg) in enumerate(zip(axes, ICONS, tgs)):
        ax.add_patch(MplPolygon(list(tg.exterior.coords), closed=True, fc="0.85", ec="k", lw=1))
        for pid, g in world_polys(clean, D, t).items():
            for gg in getattr(g, "geoms", [g]):
                if gg.geom_type == "Polygon":
                    ax.add_patch(MplPolygon(list(gg.exterior.coords), closed=True, fc=col[pid], ec="k", lw=0.5, alpha=0.85))
        for name, R in region_geoms(icon, tg).items():
            for rr in getattr(R, "geoms", [R]):
                ax.add_patch(MplPolygon(list(rr.exterior.coords), closed=True, fill=False, ec="r", ls="--", lw=1))
            ax.text(*R.centroid.coords[0], name, color="r", fontsize=7, ha="center")
        d = out["targets"][icon]
        ax.set_title(f"{icon}: IoU {d['iou_clean']:.3f}", fontsize=9)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(1.5, -1.5)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(f"dog <-> duck, k={ev['n_pieces']}, flips on (stand-in MDI silhouettes); cleaned dissection", fontsize=9)
    fig.tight_layout()
    fig.savefig(root / "m6_regions.png", dpi=110)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
