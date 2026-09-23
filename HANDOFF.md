# Handoff

The spec is in `TASK.md`. Assumptions and deviations are in `DEVIATIONS.md`.

## Status

| Milestone | State |
|---|---|
| M0 Reproduce SRD | **Done** on a 14-shape subset (CPU). Reproduces the paper's range; see below. Per-shape rows in `m0/m0_results.csv`. |
| M1 Multi-arrangement wrapper | **Done.** `srd_dissect/`, v0 rewrites with scopes and locks, 32 unit tests passing (`uv run pytest`). Integration smoke runs in `runs/m1_smoke/`. |
| M2 Harness | **Done.** `eval/`: Voronoi Scissors 6.1 protocol, validity checker, isometric overlap clean-up. 9 synthetic tests with known answers (`tests/test_m2.py`). |
| M3 Sanity dissection | **Done, negative result.** SRD does not recover Dudeney's exact dissection: best of 28 restarts reaches Chamfer 6.5 / Hausdorff ~25 px. See below. |
| M4 Dog–Bone k=4 | **Done on stand-in inputs** (not comparable): Chamfer 3.96 / Hausdorff 24.1 vs VS 2.54 / 8.89 on the real inputs. |
| M5 Full table | Running overnight (`scripts/run_benchmarks.sh`); `results/table1.csv`, `results/table1.md`. |
| M6 Dog ↔ duck | **Done on stand-in inputs**, k = 6, flips on: IoU 0.886 (dog) / 0.922 (duck). |

## Setup

```bash
uv sync                               # this repo's env; pulls d4descent@a66b729 from GitHub
uv run pytest                         # M1 unit tests
PYTHONPATH=. uv run python -m srd_dissect.run --pair square triangle --k 4 --init growth --rounds 40 --out runs/x
scripts/setup_d4descent.sh            # clones d4descent@a66b729 to ../d4descent and runs uv sync
python3 m0/run_m0.py run --jobs 4     # M0 subset (seed 0), then aggregates
python3 m0/run_m0.py aggregate        # re-aggregate only
```

d4descent core files are **not** modified. M0 calls its `scripts/optimize_shc.py` unchanged. `srd_dissect` imports d4descent's `Shape` grammar (Split/Merge/ToArc/ToLine, `resolve_intersections`, `canonicalize_loops`) and its rasterizer as-is.

## M0: reproduce d4descent Arc–Line fitting

**Reference** (paper Tables 2/3, AL-F Full, GPU):

| Dataset | PSNR | #prims | time/shape |
|---|---|---|---|
| OneComp (128) | 44.3 | 9 | 80 s |
| Donut (25) | 48.1 | 10 | 106 s |
| TwoComp (23) | 49.3 | 11 | 107 s |

**Run:** flags come verbatim from `runs/_rungen_arclines.py` (AL-F, AdaptiveLR lr 0.5, w_line = w_arc = 1e-5, 64 proposals every 25 steps, 1 local step, stopping patience 25), with `batch_param_count` 1024 (memory only). Subset: seed 0, OneComp {10, 66, 77, 98, 103, 107, 122, 124}, Donut {6, 11, 18}, TwoComp {4, 9, 16}.

**Results** (seed 0; 4 single-thread processes in parallel on a 4-core CPU, no GPU):

| Dataset | n | PSNR final | PSNR best | Paper | Median per-shape PSNR (cap 60) | #prims | Paper | Prim. match % | Time/shape |
|---|---|---|---|---|---|---|---|---|---|
| OneComp | 8 | 34.2 | 54.8 | 44.3 | 59.1 | 6.0 | 9 | 96 | 1314 s |
| Donut | 3 | 34.4 | 50.0 | 48.1 | 48.0 | 12.3 | 10 | 83 | 4323 s |
| TwoComp | 3 | 52.1 | 52.1 | 49.3 | 50.6 | 11.0 | 11 | 86 | 3273 s |

PSNR = 10·log10(1 / mean MSE over the shapes). "Final" is the state `optimize()` returns; "best" is the minimum MSE along the trajectory.

**Verdict: reproduced.** On the best state every dataset meets or beats the paper's PSNR, with primitive counts in the same range. 13 of the 14 shapes end at MSE ≤ 1.6e-5; two are near-exact (≤ 1.6e-11).

**Caveat: late divergence.** Two shapes (OneComp 122, Donut 6) reach a low loss (6.5e-6 and 3.6e-6), then diverge in the last few steps after a rewrite round, to 3e-3 and 1e-3. d4descent's `optimize()` returns the *final* state, not the best, so one such shape dominates the mean-MSE PSNR of its set. Whether the paper's numbers include such cases is unknown. **Consequence for M1:** track and return the best state; after a rewrite, keep the optimizer from taking a jolt from a stale step size.

Runtime is 12–60× slower than the paper's GPU timings, which is expected on one CPU thread at 256² resolution with 64 proposals per round.

## M1: multi-arrangement wrapper

**Layout (`srd_dissect/`).**
- `state.py`: `Pose(θ, tx, ty, flip)`, `Piece` (stable pid, d4descent `Shape` in the local frame, one pose per target), `Dissection`.
- `render.py`: packs pieces into a d4descent `ShapeCollection`. It renders target t by transforming the control points (and negating k under a flip), then calling the unmodified rasterizer.
- `loss.py`: coverage over the union (clamped sum of occupancies), overlap, fold-over, short segments, and simplicity g.
- `rewrites.py`: all v0 kinds with scope and locks, `apply_rewrite` / `apply_many`, proposal sampling, repair.
- `srd.py`: the loop, batched scoring against the same-pieces-stepped baseline, greedy apply under locks, best-state tracking, diagnostics.
- `init.py`: partition and growth initializations.
- `viz.py`, `run.py`: figures and runner (seed, config and hardware logged to `config.json`).
- The v0 set is implemented: Split/Merge/MergeClose, ToArc/ToLine, CutPart, AddPart/RemoveSmallPart, Rotate, Flip (gated), SwapPoses, soft overlap penalty with annealed w_ov. Relocate is implemented but off. FusePart and TrimOverlap are not yet implemented.

**Tests (`tests/test_m1.py`, 32 passing).**
- Pose render equals the explicitly transformed shape, including flips.
- Exact shared rewrites leave *both* targets' renders unchanged: Split, ToArc, Merge-after-Split, ToLine at k = 0, multiple grammar rewrites on one piece, CutPart, recentre, and repair, including splitting a multi-loop piece. The measured tolerances are in DEVIATIONS.
- Every per-target rewrite (Rotate, Flip, SwapPoses, Relocate) leaves the other target's render *bit-identical* and the geometry untouched. Rotate and Flip keep the world centroid; Swap exchanges centroids.
- The lock conflict matrix and greedy selection.
- Gradient routing: poses get only their own target's gradient, and geometry gets exactly the sum.
- A no-op rewrite scores exactly 0 against the baseline.

**Integration smoke** (square ↔ triangle, 40 rounds, seed 0, free-k, 2 threads each, two runs in parallel, ~12 s/round):

| init | best L | IoU sq / tri | overlap (mean relu) | pieces | arc fraction |
|---|---|---|---|---|---|
| partition (k=4) | 0.0115 | 0.956 / 0.957 | 9e-5 / 6e-5 | 13 | 0.83 |
| growth (4 disks) | 0.0084 | 0.967 / 0.958 | 1e-5 / 1e-5 | 11 | 0.91 |

Accept rates (proposed → accepted), growth run:

| Rewrite | Scope | Proposed | Accepted | Rate |
|---|---|---|---|---|
| ToArc | shared | 57 | 28 | 0.49 |
| ToLine | shared | 363 | 26 | 0.07 |
| Merge | shared | 176 | 11 | 0.06 |
| Split | shared | 607 | 13 | 0.02 |
| AddPart | shared | 271 | 5 | 0.02 |
| Rotate | per-target | 306 | 3 | 0.01 |
| CutPart | shared | 450 | 3 | 0.01 |
| RemoveSmallPart | shared | 76 | 1 | 0.01 |
| SwapPoses | per-target | 254 | 0 | 0.00 |

**What the diagnostics say (inputs to M3):**
- **Cross-target gradient conflict is strongly negative.** The mean cosine is −0.44 (partition) and −0.62 (growth); 58% and 74% of piece-rounds are below −0.5. Yet **CutPart is accepted only 1% of the time.** CutPart is exact, so after one local step the two halves have barely moved apart and rarely beat the baseline. This is the same failure mode the prototype showed for SwapPoses. The fix to try first: score CutPart and SwapPoses with a few more local steps (the scorer already supports `local_steps`; it needs to be per-kind).
- **Free-k overshoots.** The piece count grows to 11–13 for a k = 4 problem, and several leftovers are tiny AddPart disks. M3 needs the finish-at-k step (free-k) or fixed-k, plus a stronger or annealed `w_part`.
- **Overlap annealing works.** The mean overlap goes from ~2e-2 to ~1e-5 as w_ov goes 0.1 → 10, at a cost of about 1–2% IoU. Gaps between pieces are visible in the figures; the M2 validity pass will quantify them.
- **ToLine does not dominate here, unlike the prototype.** The arc fraction stays at 0.83–0.91.

## M2: evaluation harness

- `eval/protocol.py` scales each target to 10,000 px² and applies the same scale to its arrangement. It samples 100 points evenly spaced along all boundary rings, aligns with translation-only ICP, and reports Chamfer (mean of the two directed means) and Hausdorff (max of the two directed maxima). The conventions are logged in DEVIATIONS.
- `eval/validity.py` reports, per arrangement: pairwise overlap area, internal gap area (holes of the union), uncovered target area, and overhang area. It also checks that every piece is simple and connected. `remove_overlaps` is the polygon-level clean-up: each piece is trimmed in its local frame by earlier (larger) pieces from every arrangement. That keeps the result isometric and overlap-free, and turns the removed material into reported gaps.
- `eval/evaluate.py` runs everything. **Headline numbers are on the cleaned, valid dissection**, with raw numbers alongside.
- Tests: uniform sampling; ICP recovers a translation (score 0); concentric circles 2 px apart (Chamfer ≈ Hausdorff ≈ 2); a 5 px spike (Hausdorff ≈ 5); an exact 2-piece dissection (zero residuals); known overlap 0.5 and its clean-up; clean-up isometry across targets; a known internal gap; a self-intersecting piece is flagged.
- Applied to the M1 smoke runs (40 rounds): Chamfer 8.3–8.7, Hausdorff 31–32 px, 4–5% of the target uncovered, overlap ≤ 6 px² raw and 0 after clean-up.

## M3: square ↔ equilateral triangle, k = 4 (Dudeney)

The targets are fitted to the window at a common area. Every configuration was run with 4 restarts × 10 min, 1 CPU thread each (`results/m3_sweep.csv`, runs in `runs/m3/`). All numbers are on the cleaned, valid dissection.

| Config | Chamfer of kept restart | Hausdorff of kept restart | Per-restart Chamfer | Mean Chamfer | Mean Hausdorff |
|---|---|---|---|---|---|
| partition init, **fixed-k** | 6.81 | 24.6 | 7.41 / 9.36 / 6.85 / 6.81 | **7.61** | **26.0** |
| partition init, free-k (finish at k from 70%) | 12.17 | 34.7 | 12.2 / 11.3 / 12.0 / 12.3 | 11.9 | — |
| growth init, free-k | 9.43 | 39.0 | 11.8 / 11.6 / 9.4 / 14.4 | 11.8 | — |
| fixed-k, w_ov_end = 3 | 8.02 | 38.3 | 6.50 / 7.42 / 10.2 / 8.02 | 8.03 | 32.8 |
| fixed-k, per-round step-size floor 0.25 | 9.60 | 37.3 | 7.61 / 9.60 / 10.4 / 7.67 | 8.81 | 33.1 |
| fixed-k, both | — | — | — | 8.75 | 34.2 |

**Dudeney is not recovered.** 0 of 28 restarts come near the exact solution. The best is Chamfer 6.5, and every restart leaves 3–7% of each target uncovered plus 1–3% overhang. Overlap is driven to ≤ 7 px² raw (0 after clean-up).

**Initialization and piece count.** **Fixed-k from a random-chord partition of A works best.** Free-k overgrows: about 17 pieces for k = 4, many of them small AddPart disks. The finishing phase then has to delete about 13 pieces in the last 30% of the budget, and the holes are never refilled. Growth vs partition under free-k is a wash. Free-k would need a cap on the piece count during growth (for example ≤ 2k) and an earlier finish; not done.

**Tuning** was done on this pair only, and every benchmark pair uses the resulting hyperparameters. The tweaks did not beat the defaults on Chamfer/Hausdorff; they only reduce area error slightly. Defaults kept: w_ov annealed 0.1 → 10 geometrically over the run, no step-size floor.

**Why it stalls** (from the logs):
1. The adaptive step-size multiplier decays to its floor (2e-4) by the last third of a run. The pieces are then effectively frozen while w_ov is still rising.
2. Fixed-k disables CutPart, so the piece *topology* is whatever the random initial chords gave. The optimizer can only bend those pieces, and Dudeney needs specific cut lines.
3. The spread across restarts (Chamfer 6.5–10.4) is larger than the differences between configs. Selecting the restart by final loss does not always pick the lowest-Chamfer restart; the protocol says to select by L2, so I follow it.

## Benchmark setup (M4–M6)

- **Inputs:** stand-in silhouettes from Material Design Icons (`srd_dissect/shapes_mdi.py`, vendored under `data/mdi/`). **Not comparable to Voronoi Scissors Table 1.** Both targets are fitted to the render window at a common area (`targets.fit_pair`).
- **One configuration for every pair**, chosen on square ↔ triangle only: partition init, fixed-k, SGD+AdaptiveLR, w_ov 0.1 → 10, 64 proposals every 25 steps, per-kind local steps (CutPart/Swap 5, pose moves/AddPart 3, rest 1), 128² render.
- **Budget:** 4 restarts in parallel, one per core, **40 min each, for every k.** Voronoi Scissors used 1/4/6 h for k = 4/5/6 on 56 cores, which is not reproducible here. The kept restart is chosen by final objective, as the protocol says ("smallest average L2").
- **Hardware:** 4-core cloud container, no GPU, 1 thread per restart.
- `scripts/run_benchmarks.sh` runs everything, `srd_dissect/bench.py` drives the restarts, and `scripts/make_table1.py` writes `results/table1.csv`/`.md`.

## M4: Dog–Bone, k = 4 (stand-in inputs)

The kept restart (seed 2) scores avg Chamfer **3.96**, avg Hausdorff **24.1**. Per-restart Chamfer: 8.91 / 8.79 / 3.96 / 5.98. The arrangements are recognizable, and the dog's head piece is reused as a bone knob (`runs/bench/dog-bone_k4/seed2/best.png`). Residuals after clean-up (A+B): uncovered 1,259 px², overhang 772 px², internal gaps 39 px²; raw overlap 6 px². The large Hausdorff comes from under-filled extremities (thin legs, the bone's round knobs). With these inputs and this budget it does not beat Voronoi Scissors' 2.54 / 8.89, which were measured on different inputs anyway.

## M6: Dog ↔ Duck, k = 6, flips on (stand-in inputs)

Report: `runs/bench/dog-duck_k6_flip/m6_report.json`. Figure with the regions: `m6_regions.png`. Kept restart: seed 2.

| | IoU | Region IoU | Chamfer | Hausdorff | Uncovered px² | Overhang px² | Internal gap px² | Overlap px² raw → clean |
|---|---|---|---|---|---|---|---|---|
| dog | 0.886 | legs 0.884, ears/head 0.854 | 6.86 | 23.7 | 790 | 391 | 82 | 1.4 → 0 |
| duck | 0.922 | bill 0.892, tail 0.846 | 7.85 | 31.8 | 598 | 199 | 218 | 4.7 → 0 |

6 pieces, one pose flipped. Parts are reused across the forms: the dog's ear/head piece becomes the duck's bill, the dog's back foot becomes the duck's tail, and one large piece is the dog's belly and legs and the duck's body. The disagreement regions score 3–5 IoU points below the whole-shape IoU, as expected: each is a piece that has to serve two different outlines.

## Findings from reading d4descent (inputs to M1)

- **Scoring baseline.** The code scores proposals against the unmodified shape after the *same* `proposal_steps` local step (it is concatenated into the proposal batch). The paper text says "vs current loss". We follow the code (see DEVIATIONS).
- **The optimizer is SGD, not Adam**, with gradient clipping (abs 2.0) and the custom `AdaptiveLRScheduler`, recreated after each rewrite round. The paper's Limitations section says Adam's state after rewrites is an open question. The spec asks for Adam with per-group LRs and state reset on touched params. That is feasible, but SGD+AdaptiveLR is the proven path, so M1 should support both and compare.
- **Rasterizer.** `ShapeCollection._rasterize(positions)` returns an unsigned distance to the nearest primitive, signed by the winding number (|wn| > π means inside). `render01` maps it through a ±blur·pixel ramp, so gradient exists only in a ~1 px band around boundaries. Distance is invariant under rigid motions, and |wn| under reflection too, so a piece in target t can be rendered by **transforming its control points** (`p' = R(θ)·diag(1, s)·p + x`, and `k' = s·k`, because reflection flips the chord's left normal) and calling the unmodified rasterizer. Pose gradients then flow through autograd with no rasterizer changes.
- **Grammar reuse.** `Shape` (list of `Line`/`Arc` with shared endpoint tensors) holds per-piece topology. `apply_rewrite`, `do_multiple_rewrites`, `generate_rewrite_specs`, `resolve_intersections`, `canonicalize_loops` and `remove_all_holes` all operate in the piece's local frame, so Split/Merge/ToArc/ToLine are shared-scope and exact in every target by construction. `ShapeCollection.from_shapes` packs pieces into flat tensors, with no parameter sharing across shapes.
- **Cost.** On this CPU a 256² render of 64 proposals dominates (~3 s per proposal batch of ~1k params, single thread). For M1 the plan is to (a) re-render only the pieces a proposal touches and reuse cached per-piece occupancy for the rest, and (b) have the local step touch only those pieces, with the baseline being the current state with the *same* pieces stepped. That keeps "baseline after the same local step" while making each proposal O(touched pieces). Resolution for the Voronoi Scissors pairs: shapes normalized to 10,000 px² fit comfortably at 128–160².

## Open questions for the user

1. **Voronoi Scissors / Duncan et al. inputs.** They can't be fetched from this environment (egress policy). Can you add the original shape files to the repo, or should I trace them from Fig. 13 (results then labelled "traced inputs, not directly comparable")?
2. **Prototypes.** `tri_grammar_srd.py`, `two_target_srd.py`, the dog/duck snapshot and `SRD_BASELINE_TASK.md` aren't in the repo. Push them if you want them used; the dog and duck silhouettes in particular are needed for M6.
3. **Compute.** M0 took 8–86 min per shape on one CPU thread vs 80–107 s on the paper's GPU. The Voronoi Scissors budgets (1–6 h on a 56-core Xeon) aren't reproducible here at equal wall-clock. Is a GPU or larger-CPU environment available for M4/M5?
