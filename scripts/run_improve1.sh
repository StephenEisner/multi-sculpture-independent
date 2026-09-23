#!/bin/bash
cd "$(dirname "$0")/.."
B=1200
b() { PYTHONPATH=. uv run python -m srd_dissect.bench --name $1 --pair duncan:$2 duncan:$3 --k $4 --restarts 4 --time-budget $B --out runs/improve1 --csv results/improve1.csv -- "${@:5}"; }
b bunny-egg_overlay_cosine bunny egg 6 --init overlay --mode fixed --lr-schedule cosine
b cat-bear_overlay_cosine cat bear 6 --init overlay --mode fixed --lr-schedule cosine
b cat-bear_baseline cat bear 6 --init partition --mode fixed
b bunny-egg_cosine bunny egg 6 --init partition --mode fixed --lr-schedule cosine
echo IMPROVE1_DONE
