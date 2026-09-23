#!/bin/bash
# Overnight benchmark pipeline (stand-in inputs; see DEVIATIONS.md). Same hyperparameters for every pair.
# Usage: scripts/run_benchmarks.sh "<config flags>" <budget_s>
cd "$(dirname "$0")/.."
CFG=${1:-"--init partition --mode fixed"}
B=${2:-2400}
R=4
run() {  # name pairA pairB k extra...
  local name=$1 a=$2 b=$3 k=$4; shift 4
  if grep -q "^$name," results/table1_standin.csv 2>/dev/null; then echo "skip $name"; return; fi
  PYTHONPATH=. uv run python -m srd_dissect.bench --name $name --pair $a $b --k $k --restarts $R --time-budget $B \
    --out runs/bench --csv results/table1_standin.csv --label "stand-in inputs, not comparable" -- $CFG "$@"
}
# M4
run dog-bone_k4 mdi:dog-side mdi:bone 4
# M6 (project pair, flips on)
run dog-duck_k6_flip mdi:dog-side mdi:duck 6 --allow-flip
# M5 (no reflections)
run dog-bone_k5 mdi:dog-side mdi:bone 5
run serpent-apple_k5 mdi:snake mdi:apple 5
run bunny-egg_k6 mdi:rabbit mdi:egg 6
run cat-bear_k6 mdi:cat mdi:teddy-bear 6
run caterpillar-butterfly_k6 mdi:bug mdi:butterfly 6
# M5 with reflections
run dog-bone_k4_flip mdi:dog-side mdi:bone 4 --allow-flip
run dog-bone_k5_flip mdi:dog-side mdi:bone 5 --allow-flip
run serpent-apple_k5_flip mdi:snake mdi:apple 5 --allow-flip
run bunny-egg_k6_flip mdi:rabbit mdi:egg 6 --allow-flip
run cat-bear_k6_flip mdi:cat mdi:teddy-bear 6 --allow-flip
run caterpillar-butterfly_k6_flip mdi:bug mdi:butterfly 6 --allow-flip
echo PIPELINE_DONE
