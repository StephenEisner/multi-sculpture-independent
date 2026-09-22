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
