#!/bin/bash
# Voronoi Scissors Table 1 rows on the Duncan et al. inputs traced from the paper figures (same config as the
# stand-in table: partition init, fixed k, 4 restarts x 40 min, no reflections).
cd "$(dirname "$0")/.."
CFG="--init partition --mode fixed"; B=${1:-2400}
run() {
  local name=$1 a=$2 b=$3 k=$4
  if grep -q "^$name," results/table1_duncan_raw.csv 2>/dev/null; then echo "skip $name"; return; fi
  PYTHONPATH=. uv run python -m srd_dissect.bench --name $name --pair duncan:$a duncan:$b --k $k --restarts 4 \
    --time-budget $B --out runs/duncan --csv results/table1_duncan_raw.csv --label "traced Duncan inputs" -- $CFG
}
run bunny-egg_k6 bunny egg 6
run cat-bear_k6 cat bear 6
run serpent-apple_k5 serpent apple 5
run caterpillar-butterfly_k6 caterpillar butterfly 6
run trump-map_k5 trump us_map 5
echo DUNCAN_DONE
