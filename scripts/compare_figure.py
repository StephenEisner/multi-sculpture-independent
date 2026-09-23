"""Side-by-side figure per pair: targets | our cleaned arrangement in A and B | Duncan et al. 2017's published result
(cropped from their paper's figure). PYTHONPATH=. uv run python scripts/compare_figure.py <paper.pdf>"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from matplotlib.patches import Polygon as MplPolygon

from eval.geometry import local_polygons, shape_polygon
from eval.validity import remove_overlaps, world_polys
from srd_dissect import targets as TG
from srd_dissect.run import get_shape, load_dissection

ROOT = Path(__file__).resolve().parents[1]
VS = {"bunny-egg": (3.26, 9.74), "cat-bear": (2.81, 8.92), "serpent-apple": (6.60, 29.36),
      "caterpillar-butterfly": (4.61, 13.71), "trump-map": (3.17, 11.18)}
# Duncan et al. result figures: list of (page, image) whose right 3/4 shows their pieces for this pair
DUNCAN_FIG = {"bunny-egg": [(11, 5), (11, 4)], "cat-bear": [(10, 2)], "serpent-apple": [(10, 3)],
              "caterpillar-butterfly": [(10, 4)], "trump-map": [(10, 7)]}


def page_image(doc, page, idx):
    pix = pymupdf.Pixmap(doc, doc[page].get_images(full=True)[idx][0])
    if pix.n - pix.alpha >= 4:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    return a[:, int(a.shape[1] * 0.22):]


def draw(ax, tg, polys, cols):
    ax.add_patch(MplPolygon(list(tg.exterior.coords), closed=True, fc="0.85", ec="k", lw=0.8))
    for pid, g in polys.items():
        for gg in getattr(g, "geoms", [g]):
            if gg.geom_type == "Polygon":
                ax.add_patch(MplPolygon(list(gg.exterior.coords), closed=True, fc=cols[pid], ec="k", lw=0.4))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(1.5, -1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def main(pdf):
    from PIL import Image

    doc = pymupdf.open(pdf)
    rows = list(csv.DictReader(open(ROOT / "results" / "table1_duncan_raw.csv")))
    cmap = plt.get_cmap("tab10")
    panels = []
    for r in rows:
        pair = r["name"].rsplit("_k", 1)[0]
        run = ROOT / "runs" / "duncan" / r["name"] / f"seed{r['best_seed']}"
        names = eval(json.loads((run / "config.json").read_text())["args"]["pair"])
        shapes = TG.fit_pair([get_shape(n) for n in names])
        D = load_dissection(run / "best.pt")
        clean = remove_overlaps(local_polygons(D), D)
        cols = {pid: cmap(j % 10) for j, pid in enumerate(sorted(clean))}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 2]})
        for t in range(2):
            draw(axes[t], shape_polygon(shapes[t]), world_polys(clean, D, t), cols)
            axes[t].set_title(f"ours: {names[t].split(':')[1]}", fontsize=11)
        imgs = [page_image(doc, p, j) for p, j in DUNCAN_FIG[pair]]
        w = min(im.shape[1] for im in imgs)
        axes[2].imshow(np.concatenate([im[:, :w] for im in imgs], 0))
        axes[2].axis("off")
        axes[2].set_title("Duncan et al. 2017, published result", fontsize=11)
        vs = VS[pair]
        fig.suptitle(f"{pair}, {r['k']} pieces.   Ours: Chamfer {float(r['avg_chamfer']):.2f}, Hausdorff "
                     f"{float(r['avg_hausdorff']):.1f}.   Voronoi Scissors: Chamfer {vs[0]}, Hausdorff {vs[1]}.",
                     fontsize=12, weight="bold")
        fig.tight_layout()
        path = ROOT / "results" / f"compare_{pair}.png"
        fig.savefig(path, dpi=80)
        plt.close(fig)
        panels.append(Image.open(path).convert("RGB"))
    W = max(p.width for p in panels)
    out = Image.new("RGB", (W, sum(p.height for p in panels)), "white")
    y = 0
    for p in panels:
        out.paste(p, (0, y))
        y += p.height
    out.save(ROOT / "results" / "compare_duncan.png")


if __name__ == "__main__":
    main(sys.argv[1])
