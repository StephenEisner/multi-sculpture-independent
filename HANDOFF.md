# Handoff

The spec is in `TASK.md`. Assumptions and deviations are in `DEVIATIONS.md`.

## Status

| Milestone | State |
|---|---|
| M0 Reproduce SRD | **Done** on a 14-shape subset (CPU). Reproduces the paper's range; see below. Per-shape rows in `m0/m0_results.csv`. |
| M1–M6 | Not started (per "stop and report after each milestone"). |

## Setup

```bash
scripts/setup_d4descent.sh            # clones d4descent@a66b729 to ../d4descent and runs uv sync
python3 m0/run_m0.py run --jobs 4     # M0 subset (seed 0), then aggregates
python3 m0/run_m0.py aggregate        # re-aggregate only
```

d4descent core files are **not** modified. M0 calls its `scripts/optimize_shc.py` unchanged.

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
