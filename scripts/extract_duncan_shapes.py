"""Extract the Duncan et al. 2017 input silhouettes from the figure images embedded in the paper PDF.

Source: github.com/noah-duncan/noah-duncan.github.io, pdf/dt-final-paper-dissections.pdf (the published
"Approximate Dissections" paper, ACM TOG 36(6) 2017). Its results figures show every input shape with a flat fill
(224,224,144) and a red outline (200,48,48); results use other colours. We take pixels of those two colours in the
input column, fill holes, and trace the outline at sub-pixel precision (marching squares), then simplify by 0.5 px.

These are TRACED inputs: close to, but not identical with, the original shape files.

  uv run python scripts/extract_duncan_shapes.py /path/to/dt-final-paper-dissections.pdf
writes data/duncan/shapes.json (polygons in image pixel coordinates, y down) and data/duncan/preview.png.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pymupdf
import shapely
from scipy import ndimage
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
FILL = np.array([224, 224, 144])
LINE = np.array([200, 48, 48])

# (page, image index on page, names of the input shapes top-to-bottom in the left column)
FIGURES = [
    (10, 0, ["cat_fish_cat", "fish"]),
    (10, 1, ["dove", "bomb"]),
    (10, 2, ["bear", "cat"]),
    (10, 3, ["serpent", "apple"]),
    (10, 4, ["caterpillar", "butterfly"]),
    (10, 5, ["island", "x"]),
    (10, 6, ["nike_shoe", "nike"]),
    (10, 7, ["us_map", "trump"]),
    (11, 4, ["egg"]),
    (11, 5, ["bunny"]),
]


def image(doc, page, idx) -> np.ndarray:
    xref = doc[page].get_images(full=True)[idx][0]
    pix = pymupdf.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return a[..., :3].astype(int)


def input_mask(img: np.ndarray) -> np.ndarray:
    near = lambda c: np.abs(img - c).max(-1) < 24
    return near(FILL) | near(LINE)


def trace(mask: np.ndarray) -> shapely.Polygon:
    m = ndimage.binary_fill_holes(mask)
    cs = measure.find_contours(np.pad(m, 1).astype(float), 0.5)
    c = max(cs, key=len) - 1  # (row, col)
    poly = shapely.Polygon(np.stack([c[:, 1], c[:, 0]], -1))  # (x, y) with y down
    return shapely.make_valid(poly).buffer(0).simplify(0.5, preserve_topology=True)


def main(pdf):
    doc = pymupdf.open(pdf)
    out = {}
    for page, idx, names in FIGURES:
        img = image(doc, page, idx)
        m = input_mask(img)
        m[:, int(img.shape[1] * 0.25):] = False  # the inputs live in the left column
        lab, n = ndimage.label(ndimage.binary_closing(m, iterations=2))
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.argsort(sizes)[::-1][: len(names)] + 1
        keep = sorted(keep, key=lambda k: ndimage.center_of_mass(lab == k)[0])  # top to bottom
        for name, k in zip(names, keep):
            poly = trace(lab == k)
            if poly.geom_type != "Polygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            out[name] = {"page": page, "image": idx, "n_vertices": len(poly.exterior.coords) - 1,
                         "area_px": poly.area, "exterior": np.asarray(poly.exterior.coords)[:-1].round(2).tolist()}
            print(f"{name:14s} {out[name]['n_vertices']:4d} vertices, area {poly.area:9.0f} px^2")
    dst = ROOT / "data" / "duncan"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "shapes.json").write_text(json.dumps(
        {"source": "Duncan et al. 2017, Approximate Dissections, figure images of the paper PDF "
                   "(github.com/noah-duncan/noah-duncan.github.io/pdf/dt-final-paper-dissections.pdf)",
         "note": "traced inputs, not the original shape files", "units": "image pixels, y down", "shapes": out}))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 6, figsize=(14, 7))
    for ax, (name, d) in zip(axes.flat, out.items()):
        P = np.array(d["exterior"])
        ax.fill(P[:, 0], P[:, 1], fc="#e0e090", ec="#c03030", lw=0.8)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(f"{name} ({d['n_vertices']}v)", fontsize=8)
    for ax in list(axes.flat)[len(out):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(dst / "preview.png", dpi=80)


if __name__ == "__main__":
    main(sys.argv[1])
