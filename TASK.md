# Multi-sculpture: 2D two-shape dissection with Stochastic Rewrite Descent

This file supersedes `SRD_BASELINE_TASK.md`. It merges that task spec with the part grammar, the rewrite scoping (shared vs per-target), and lessons from two prototypes.

Record decisions, blockers, and results in `HANDOFF.md` as you go. Log anything you had to assume in `DEVIATIONS.md`.

## Goal and roadmap

One inventory of smooth rigid parts is rearranged into several target forms. Parts are smooth (no blocky pieces) and assembled without glue.

The optimizer is SRD from *Design for Descent* (Kodnongbua, Zhang, Sharp, Schulz, SIGGRAPH Asia 2025). The baseline to beat is Voronoi Scissors (Qi et al.).

Staging, per Professor Schulz:

1. **2D, two shapes.** This spec. First the benchmark pairs, then the project pair dog ↔ duck.
2. **2D, n shapes.** Do not start it yet. The design below is written so that it is a loop over targets, not a rewrite.
3. **3D.** Later. Do not start it, but avoid choices that would block it.

## Problem definition

**Inputs:** n target silhouettes (n = 2 in this phase), scaled to equal area, and optionally a fixed piece count k.

**Output:** K pieces. Each piece is one closed Arc–Line outline in its own local frame, plus one pose per target.

A pose is (θ_t, x_t, y_t, s_t), where s_t ∈ {+1, −1} is a reflection flag. Reflection is gated by `ALLOW_FLIP`.

**Requirements** (the same three Voronoi Scissors guarantees):

1. **Isometry.** Each piece is the same shape in every arrangement. This holds by construction: geometry is shared, only poses differ.
2. **Seamless.** No overlaps between placed pieces, and no internal gaps.
3. **Approximation.** Each arrangement approximates its target as closely as possible.

**Reflections:** Voronoi Scissors excludes them. The main comparison runs with `ALLOW_FLIP=False`. Report a separate "with reflections" column. For dog ↔ duck, Flip is wanted, so run it with the flag on.

## Representation

Each piece has:

- `V` (m×2): vertices in the local frame, ordered counter-clockwise.
- `arc` (m): per-segment flag.
- `k` (m): bulge per arc segment, meaning the orthogonal deviation from the chord midpoint. Lines have k = 0 and no gradient on k.
- For each target t: `theta[t]`, `trans[t]`, `flip[t]`.

The rendered piece in target t is T_t(sample(V, k, arc)), where T_t rotates, optionally reflects, and translates.

Build on the d4descent Arc–Line grammar rather than reimplementing it. Repo: https://github.com/milmillin/d4descent (Python, `uv sync`).

- Grammar: `src/d4descent/objects/arclines.py` (`Shape.apply_rewrite`, `resolve_intersections`, `canonicalize_loops`)
- Optimizer loop: `src/d4descent/optimizer.py`
- Raster loss: `src/d4descent/losses/raster.py`

Wrap or subclass. Don't edit core files in place; if a core change is unavoidable, note it in `HANDOFF.md`.

## Loss

L = Σ_t [ w_cov · L2(softraster(∪_i T_i^t(piece_i)), target_t) + w_ov · ∫ max(0, Σ_i occ_i^t − 1) ] + g

- g is the non-differentiable simplicity term: w_part · (#pieces) + w_seg · (#segments).
- Use the d4descent soft rasterizer, which uses winding-number insideness.
- Anneal w_ov upward over the run, so overlap is driven to about zero by the end.
- Add small regularizers: penalize a piece's local signed area going negative (fold-over), and penalize segments shorter than about 1 px.

**Gradient flow is the core of the method:**

- **Piece geometry** (V, k) receives the *sum* of gradients over all targets.
- **Each pose** receives only its own target's gradient.

When targets pull a shared boundary in opposite directions, the pulls cancel, and that conflict is what CutPart exists to resolve.

## Rewrites and their scope

The rule: **a rewrite's scope is determined by what it writes to.**

- A rewrite that edits piece geometry is **shared**: it changes that piece in every target at once.
- A rewrite that edits a single pose is **per-target**: other targets are untouched.
- **Scoring is always on the loss summed over all targets**, whatever the scope. A per-target move that helps the duck is still rejected if it doesn't lower the total.

| Rewrite | Scope | Exact? | Notes |
|---|---|---|---|
| Split segment | Shared | Yes (line), near-exact (arc) | Adds a boundary vertex. Main source of new detail. |
| Merge segment | Shared | Only if deviation < ε | Inverse of Split. Keep the paper's ε guard. |
| ToArc | Shared | Yes | Starts with k = 0. |
| ToLine | Shared | Only if \|k\| < ε | Inverse of ToArc. |
| Canonicalize / within-piece intersection repair | Shared | Yes | Purely local to one outline. |
| CutPart(i, chord) | Shared | Yes, in every target | Both halves inherit all of the parent's poses. The core dissection move. The chord must lie inside the piece. |
| FusePart(i, j) | Shared | Only under a condition | Allowed only if i and j are adjacent and within ε of the same relative pose in **every** target. Rarely true, so reversibility comes mostly from RemoveSmallPart. |
| AddPart | Shared | No | The piece exists in all targets, so it needs a spawn position in **each** one. Sample from each target's uncovered residual independently. |
| RemoveSmallPart(i) | Shared | Near-exact when small | Removes the piece from every target. Also runs as a repair. |
| Continuous pose step | Per-target | — | Gradient on (θ_t, x_t, y_t). |
| Rotate(i, t, Δθ) | Per-target | No | Discrete jumps: ±90°, 180°, random. Rotate about the piece's world centroid. |
| Flip(i, t) | Per-target | No | Toggles s_t, keeping the centroid fixed. Gated by `ALLOW_FLIP`. |
| SwapPoses(i, j, t) | Per-target | No | Exchanges the two pieces' centroids and orientations in target t only. |
| Relocate(i, t) | Per-target | No | Moves the centroid to a sampled uncovered spot in target t. |
| TrimOverlap(i, j, t) | **Hybrid** | No | Triggered by an overlap in one target, but it edits shared geometry, so it changes piece i everywhere. Score on the summed loss. Implement it after the v0 set works. |

Why this matters for n targets: shared rewrites must be valid in every target simultaneously. Exact shared rewrites (Split, ToArc, CutPart) get this for free because geometry lives in the local frame. Conditional shared rewrites (Merge, FusePart) must check their condition in every target. Per-target rewrites generalize to n with no changes.

### Locks for SRD's greedy apply step

After scoring, apply the improving rewrites greedily by decreasing ΔL, skipping any that conflict with one already applied.

- A **shared** rewrite on piece i holds (i, t) for every t.
- A **per-target** rewrite on piece i in target t holds only (i, t). Two pose moves on the same piece in different targets do not conflict.
- **SwapPoses** holds (i, t) and (j, t).
- **AddPart** holds a single global "add" lock, so one spawn happens per round. Loosen this later if it's too slow.

Give pieces stable IDs. Rewrites reference IDs, not list indices, because earlier rewrites in the same round can add or remove pieces.

## SRD loop

Each outer iteration:

1. **Continuous phase.** Run a few Adam steps with separate learning-rate groups for vertices, bulges, translation, and rotation. After any rewrite, reset the optimizer state for the parameters that rewrite touched.
2. **Repair.** Remove degenerate pieces, collapse coincident vertices, fix within-piece self-intersections, and wrap angles.
3. **Propose.** Sample K candidate rewrites (the paper uses 64).
4. **Score each candidate.** Apply it to a copy, take one local optimization step, and compute ΔL.
5. **Apply.** Take the greedy compatible subset of the improving candidates, using the locks above.

**Scoring baseline (important).** Compare each candidate against the current state *after the same one local step*, not against the current loss. The prototypes showed that one Adam step alone is a large enough jolt to make every candidate look worse, so no rewrite was ever accepted. Check how the d4descent code handles this and log what you choose.

## Initialization

Try both and report which works better:

- **Spec default:** partition target A into k pieces (random chords or a few Voronoi cells). Place them inside B with random poses.
- **Growth start:** a few tiny pieces spawned at random covered spots in each target. Growth, CutPart, and AddPart build up from there. This is what the dog ↔ duck prototype used.

Run multiple restarts in parallel and keep the best, as Voronoi Scissors does. That keeps the time-budget comparison fair.

## Piece count

The comparison fixes k. Try both options and report which works:

- **Free-k:** split and merge freely during the run, then finish at exactly k.
- **Fixed-k:** hold k constant after initialization.

## Final validity pass

Convert the result to exact polygons, with arcs densely sampled. Check:

- overlap area in each arrangement
- internal gap area not present in the target
- that every piece is a simple, connected polygon

Report residuals honestly. If overlap is not exactly zero, add a polygon-level clean-up and document it.

This is the known hard part: trimming a piece to fix an overlap in A also changes it in B. A result with overlaps is not a valid dissection.

## Evaluation protocol (must match Voronoi Scissors Sec. 6.1)

1. Normalize each input shape to an area of 10,000 px².
2. Uniformly sample 100 points on each shape. The paper says "on each shape"; treat this as the boundary and log the assumption.
3. Align each arrangement to its input by ICP with translation only (rotations fixed).
4. Report mean Chamfer and Hausdorff distance per shape, plus their averages over A and B.
5. Time budgets: 1 h for 4 pieces, 4 h for 5, 6 h for 6. Pick the result with the smallest average L2. Record hardware; theirs was a 56-core Xeon Max 9480.
6. Use the same hyperparameters for every pair, as they did. No per-pair tuning.

### Numbers to beat (Voronoi Scissors, Table 1)

| Pair | k | Avg Chamfer | Avg Hausdorff |
|---|---|---|---|
| Dog–Bone | 4 | 2.54 | 8.89 |
| Dog–Bone | 5 | 3.85 | 13.69 |
| Trump–Map | 5 | 3.17 | 11.18 |
| Bunny–Egg | 6 | 3.26 | 9.74 |
| Caterpillar–Butterfly | 6 | 4.61 | 13.71 |
| Cat–Bear | 6 | 2.81 | 8.92 |
| Serpent–Apple | 5 | 6.60 | 29.36 |
| **Average** | | **3.83** | **13.64** |

The Hat–Ghost pair (k = 4, 5) appears in their ablations and is a good development case.

### Input shapes

These pairs come from Duncan et al. 2017. Try to get the original inputs from that paper's supplementary material or project page. Voronoi Scissors code: no release was found; check once more before assuming none exists.

If you can't get the originals, trace them from the figures (Voronoi Scissors Fig. 13 shows them in black). Label any such results "traced inputs, not directly comparable." Don't claim a win on traced inputs without flagging it.

## Milestones (stop and report after each)

- **M0: Reproduce SRD.** Run d4descent Arc–Line single-image fitting on its own benchmark (`runs/_rungen_arclines.py`). Confirm results are in the paper's range.
- **M1: Multi-arrangement wrapper.** Build the piece + per-target-pose state on top of the Arc–Line grammar. Implement the v0 rewrites and locks. Unit-test that every exact shared rewrite leaves both renders unchanged, and that every per-target rewrite leaves the other target's render unchanged.
- **M2: Harness.** Build the evaluation protocol and the validity checker. Test them on synthetic cases with known answers.
- **M3: Sanity dissection.** Square ↔ equilateral triangle, k = 4. Dudeney's exact solution exists, so the error should approach 0. Report how close SRD gets and how often across restarts.
- **M4: First baseline pair.** Dog–Bone, k = 4, under the 1 h budget. Compare to 2.54 / 8.89.
- **M5: Full table.** All seven rows, plus the with-reflections column.
- **M6: Dog ↔ duck.** The project pair, with Flip on. No published numbers exist, so report IoU per target, overlap and gap area, piece count, and Chamfer/Hausdorff. Log IoU separately for the regions where the forms disagree (the dog's legs and ears; the duck's bill and tail).
- **Ablations**, after M5 if time allows:
  - remove each rewrite family (CutPart, SwapPoses, Flip, discrete Rotate, TrimOverlap), mirroring the paper's Table 2
  - fixed-k vs free-k
  - restart count vs quality

**v0 rewrite set:** Split/Merge, ToArc/ToLine, CutPart, AddPart/RemoveSmallPart, Rotate, Flip, SwapPoses, soft overlap penalty. Then add Relocate, TrimOverlap, and FusePart one at a time, ablating each.

## Diagnostics to log every run

- **Accept rate per rewrite type, split by scope.** This tells you what to tune before scaling to n targets.
- **Cross-target gradient conflict.** Per piece, the cosine between target A's and target B's gradient on the shared geometry parameters. If it is often strongly negative, CutPart is doing the real work.
- **Piece-count trajectory.** Watch for split/merge ping-pong (see findings below).
- **Per-target IoU**, overlap area, and gap area over time.

## Findings from the prototypes

Two small prototypes are included as reference only. They use pure-numpy `autograd` because PyTorch wasn't available in their environment, so they are slow (about 4–10 s per SRD iteration at 56–64 px). Use PyTorch or JAX in the real implementation.

**`tri_grammar_srd.py`: triangle-mesh pieces, one target (dog).**

- Filled most of the silhouette within about 10 iterations, then plateaued.
- The piece count swung back and forth (7 → 14 → 10) as Split and Merge undid each other without improving the loss. Needs a cooling schedule, or a cost on structural change late in the run.
- The overlap penalty pushed neighbouring pieces apart, leaving visible gaps. For a seamless dissection, adjacent pieces may need to share boundary vertices rather than just being penalized for overlapping.
- Sliver and per-triangle penalties initially blocked Grow entirely. Penalty weights need scaling relative to the gain one new primitive can deliver.

**`two_target_srd.py`: Arc–Line pieces, two targets (dog ↔ duck), scoped rewrites and locks as specified above.**

- Uses parabolic rather than circular arcs, which is a deviation. The real implementation should use d4descent's circular arcs.
- Snapshot at iteration 30: 8 pieces, IoU 0.87 (dog) and 0.92 (duck). See `dog_duck_snapshot_iter30.png`. Recognizable shared pieces formed, e.g. the dog's head reused as the duck's head.
- Accept rates up to that point:

| Rewrite | Scope | Valid proposals | Accepted | Rate |
|---|---|---|---|---|
| Split | Shared | 145 | 39 | 0.27 |
| ToLine | Shared | 42 | 16 | 0.38 |
| Merge | Shared | 52 | 14 | 0.27 |
| Rotate | Per-target | 109 | 11 | 0.10 |
| CutPart | Shared | 49 | 9 | 0.18 |
| Relocate | Per-target | 51 | 4 | 0.08 |
| Flip | Per-target | 44 | 4 | 0.09 |
| AddPart | Shared | 59 | 4 | 0.07 |
| SwapPoses | Per-target | 117 | 1 | 0.01 |
| ToArc | Shared | 15 | 1 | 0.07 |
| RemoveSmallPart | Shared | 57 | 0 | 0.00 |

- SwapPoses almost never fired. Swapping two differently-shaped pieces rarely helps after only one local step; try scoring swaps with a few more local steps, or proposing swaps only between pieces of similar area.
- Some pieces still overhang the silhouettes, and there are gaps between pieces. Expect both to need the w_ov annealing and the final validity pass.
- ToLine was accepted often, so outlines drift toward polygons. Watch the fraction of boundary that is arc if smoothness matters (it does for the no-blocky-parts constraint).

## Deliverables

- `srd_dissect/`: the multi-arrangement Arc–Line wrapper, rewrites with scopes and locks, loss, and runner.
- `eval/`: the evaluation protocol and validity checker.
- `results/table1.csv`: per-shape and averaged Chamfer/Hausdorff, residual overlap and gap, runtime, restarts, and seed.
- For each pair, a figure showing both arrangements with pieces colored consistently across targets, next to the targets.
- An updated `HANDOFF.md` covering what works, what doesn't, and open questions.

## Ground rules

- Log seeds and configs for every reported number.
- If a result looks better than the baseline, check the validity residuals before celebrating.
- Ask before anything that changes the problem definition, e.g. allowing non-rigid pieces or dropping the seamless requirement.
