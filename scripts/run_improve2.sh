#!/bin/bash
cd "$(dirname "$0")/.."
until grep -q IMPROVE1_DONE /tmp/claude-0/improve1.log; do sleep 30; done
B=1200
OV="--init overlay --overlay-min-core 3 --overlay-min-frac 0.01 --mode fixed --lr-schedule cosine"
b() { PYTHONPATH=. uv run python -m srd_dissect.bench --name $1 --pair duncan:$2 duncan:$3 --k $4 --restarts 4 --time-budget $B --out runs/improve2 --csv results/improve2.csv -- "${@:5}"; }
b bunny-egg_ovbal bunny egg 6 $OV
b cat-bear_ovbal cat bear 6 $OV
b bunny-egg_ovbal_surface bunny egg 6 $OV --surface-loss
b cat-bear_ovbal_surface cat bear 6 $OV --surface-loss
echo IMPROVE2_DONE
