# Deviations and assumptions

Anything assumed or changed relative to `TASK.md` is logged here.

## Environment

- **No GPU.** The session container is 4 CPU cores and 15 GB RAM, with no CUDA. All timings are CPU timings and are not comparable to the paper's wall-clock (the paper reports GPU times).
- **Egress is restricted.** GitHub and PyPI are reachable. The paper PDF (computationaldesign.group), arXiv and DeepWiki are blocked. The user supplied the D4Descent PDF directly. A probe on 2026-09-22 also found dl.acm.org, cs.ubc.ca, csail.mit.edu, stanford.edu, Google Scholar, Semantic Scholar and Hugging Face blocked; only GitHub (incl. raw.githubusercontent.com) and PyPI work. The Duncan et al. 2017 input shapes therefore can't be fetched from the ACM supplement or a project page in this environment, and GitHub code search is outside this session's repo scope. This matters for M4/M5: the inputs must be supplied by the user or traced.
- **Files referenced by the spec are missing from the repo:** `SRD_BASELINE_TASK.md`, `tri_grammar_srd.py`, `two_target_srd.py`, `dog_duck_snapshot_iter30.png`. I took the prototype findings from the spec text only.

## M0

- **Subset, not the full benchmark.** On CPU, one OneComp shape takes roughly 30–60 min (vs 80 s on the paper's GPU). M0 runs a seeded random subset instead: 8 OneComp, 3 Donut and 3 TwoComp shapes (seed 0, see `m0/run_m0.py`), 14 of 176.
- **`--optim.batch_param_count 1024` instead of 8192.** At 8192 the proposal batch is OOM-killed at about 14 GB RSS. Batches are evaluated independently, so the change affects memory and speed but not results.
- **Single-threaded processes** (`OMP_NUM_THREADS=1`), run 4 in parallel. Under the default 4 threads, sys time was larger than user time (thread contention).
- **PSNR** is computed as `10·log10(1/MSE)`, where MSE is d4descent's final raster loss (`$loss_cont`, mean squared error of the 256² soft rasterization vs the target in [0,1]). The paper doesn't spell out its formula; this is the natural reading.

## SRD scoring baseline (applies to M1+)

- **The paper and the code differ.** The paper (§3.3) writes ΔL_ρ ≈ L(s,p) − L(s′, p̂), i.e. it compares against the *current* loss. The released code (`optimizer.py`, `proposal_criterion="loss"`) concatenates the unmodified shape into the proposal batch, gives it the same `proposal_steps` local optimization step, and compares every proposal against that stepped original (`og_loss = all_losses_[-1]`). Acceptance additionally needs an improvement above `better_abs_eps=1e-8` or `better_rel_eps=1e-2 × base_loss`. **Decision:** follow the code, comparing against the current state after the same local step, as the spec asks.

## M1: multi-arrangement wrapper

- **Union = clamped sum of per-piece soft occupancies**, not a soft raster of the union outline (min-SDF). With a min-SDF union every cut shows a half-intensity seam (both pieces are at their boundary on the chord). With the sum, the two ramps are complementary across a shared edge (occ_A + occ_B = 1), so CutPart leaves the union unchanged. The overlap term `mean(relu(Σ occ − 1))` is then exactly zero for a perfect tiling. The per-piece soft raster is d4descent's, unmodified.
- **Lock refinement.** The spec has any shared rewrite on piece i hold (i, t) for every t. I kept that for all shared rewrites except d4descent's own grammar rewrites (Split/Merge/MergeClose/ToArc/ToLine). Those hold (i, t) in shared mode plus their primitive indices exclusively, so several of them on *disjoint primitives of the same piece* can be applied together. That is exactly what d4descent does within one shape, and it preserves its convergence speed. They still conflict with every other rewrite on piece i (CutPart, pose moves, …).
- **Scoring baseline per touched set.** A proposal re-renders and steps only the pieces it touches. Every other piece is held fixed at its cached occupancy, which makes each proposal O(touched pieces). The matching baseline is the current state with *the same pieces* stepped once, computed once per distinct touched set. AddPart touches no existing piece, so it is compared with the unstepped current loss. RemoveSmallPart's baseline is the piece stepped once while the proposal has nothing to step, a slight bias against removal (same as d4descent, where the original is always stepped).
- **Optimizer.** The default is SGD with per-group learning rates (vertices 0.5, bulges 0.5, rotation 1.0, translation 0.5), gradient value clipping at 2.0, and d4descent's AdaptiveLR logic as one global multiplier whose plateau tracker restarts each round (as d4descent's recreated scheduler does). Adam is available (`optimizer="adam"`). With Adam, *all* state is reset every round rather than only for the touched parameters; not yet compared.
- **Near-exactness, measured** (128², tests in `tests/test_m1.py`):
  - Split(line) and Merge-after-Split: ≤ 1e-5.
  - Split(arc): ≤ 1.5e-3 max occupancy error, about 0.016 px total.
  - ToArc: ≤ 0.013 max and ≤ 0.17 px total. Rarely a *single* pixel also flips, because d4descent's rasterizer regularizes a k = 0 arc to radius ≈ (chord/2)²/1e-4 and float32 then misclassifies pixel centres within ~1e-4 of the chord's line. This is upstream behaviour and is not patched.
  - CutPart: < 2 px of occupancy in total, all at the two chord endpoints where three ramps meet. Overlap unchanged to 1e-6.
- **Pieces are simple regions.** Repair drops CW (hole) loops. It splits a piece whose outline became several CCW loops into separate pieces; this is exact, and each keeps all poses. Local frames are re-centred on the area centroid after every repair (exact).
- **CutPart chords are straight lines** between points at t ∈ [0.2, 0.8] of two distinct primitives. Validity: the chord (shortened 2% at each end) lies inside the piece polygon (shapely, 16 samples per primitive), and both halves have area > 1e-3. A chord can be bent later by ToArc.
- **AddPart** spawns a 4-arc disk of radius 0.04 (about 1.7 px at 128²) at a uniformly sampled uncovered target pixel in *each* target, with independent random θ.
- **Best state.** The loop tracks and returns the best state under the *final* objective (w_ov = w_ov_end). This follows the M0 finding that d4descent returns the last state.
- **Not yet implemented** (post-v0 per the spec): FusePart, TrimOverlap. Relocate is implemented but off (family weight 0).
- **Units.** The world window is [−1.5, 1.5]², rendered at 128². Built-in targets are normalized to area 1.5 world units² (≈ 2,700 render px). The M2 evaluation will rescale to 10,000 px².
