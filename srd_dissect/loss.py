"""Multi-target dissection loss.

L = sum_t [ w_cov * mean((clamp(sum_i occ_i^t, 0, 1) - target_t)^2) + w_ov * mean(relu(sum_i occ_i^t - 1)) ]
    + w_fold * sum_i relu(-area_i) + w_short * sum_seg relu(l_min - len)^2
    + g,   g = w_part * #pieces + w_seg * #segments   (non-differentiable simplicity)

The union is the clamped *sum* of soft occupancies rather than a min-SDF. For two pieces sharing an edge the
ramps are complementary (occ_A + occ_B = 1 across the seam), so a CutPart leaves the union unchanged and a
perfect tiling has zero overlap penalty. A min-SDF union would show a half-intensity seam along every cut.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .render import Packed


@dataclass
class LossConfig:
    w_cov: float = 1.0
    w_ov: float = 1.0  # annealed by the optimizer (see SRDConfig.w_ov_start / w_ov_end)
    w_fold: float = 1.0
    w_short: float = 1e-4  # per segment at zero length
    short_px: float = 1.0  # segments shorter than this many pixels are penalized
    w_part: float = 1e-4
    w_seg: float = 1e-5
    # optional per-pixel weights on the coverage term, (T, H, W); see surface_weights
    cov_weights: torch.Tensor | None = None


def image_terms(sum_occ: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """sum_occ: (B, T, H, W), targets: (T, H, W), weights: optional (T, H, W) on the coverage term.
    Returns coverage and overlap per (B, T)."""
    err = (sum_occ.clamp(0, 1) - targets).square()
    if weights is not None:
        err = err * weights
    cov = err.flatten(-2).mean(-1)
    ov = torch.relu(sum_occ - 1).flatten(-2).mean(-1)
    return cov, ov


def piece_regularizers(pk: Packed, cfg: LossConfig) -> torch.Tensor:
    """Per-piece regularizer (P,): fold-over (negative local area) + short-segment penalty."""
    fold = torch.relu(-pk.local_areas())
    L, seg_piece = pk.segment_lengths()
    lmin = cfg.short_px * pk.cfg.pixel
    short = torch.zeros(pk.n_pieces).index_add(0, seg_piece, torch.relu(lmin - L).square() / pk.cfg.pixel**2)
    return cfg.w_fold * fold + cfg.w_short * short


def simplicity(n_pieces: int, n_segments: int, cfg: LossConfig) -> float:
    return cfg.w_part * n_pieces + cfg.w_seg * n_segments


@dataclass
class LossBreakdown:
    total: float
    cov: list[float]
    ov: list[float]
    reg: float
    simp: float


def dissection_loss(pk: Packed, targets: torch.Tensor, cfg: LossConfig, n_segments: int,
                    occ: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, LossBreakdown]:
    """Differentiable loss (without g) for a whole packed dissection; also returns a breakdown incl. g."""
    T = targets.shape[0]
    if occ is None:
        occ = [pk.occupancy(t) for t in range(T)]
    sum_occ = torch.stack([o.sum(0) for o in occ], 0)[None]  # (1, T, H, W)
    cov, ov = image_terms(sum_occ, targets, cfg.cov_weights)
    reg = piece_regularizers(pk, cfg).sum()
    L = (cfg.w_cov * cov + cfg.w_ov * ov).sum() + reg
    g = simplicity(pk.n_pieces, n_segments, cfg)
    return L, LossBreakdown(L.item() + g, cov[0].tolist(), ov[0].tolist(), reg.item(), g)


def surface_weights(targets: torch.Tensor, pixel: float, band_px: float = 3.0, w_surface: float = 1.0,
                    w_deep: float = 0.1, w_out: float = 3.0) -> torch.Tensor:
    """Surface-first coverage weights (T, H, W). The target *surface* is every edge of the shape: its outline and the
    edges of any real holes. Missing material within band_px of the surface costs w_surface; missing material deeper
    inside costs only w_deep (hidden interior gaps are acceptable); material outside the target costs w_out."""
    import numpy as np
    from scipy import ndimage

    out = []
    for t in range(targets.shape[0]):
        inside = (targets[t] > 0.5).numpy()
        depth = ndimage.distance_transform_edt(inside)  # pixels to the nearest non-target pixel (the surface)
        w = np.where(inside, np.where(depth <= band_px, w_surface, w_deep), w_out)
        out.append(torch.tensor(w, dtype=torch.float32))
    return torch.stack(out)
