#!/bin/bash
# Directions, then one arm at a time, see PLAN_BOTH.md.
set -e
export HF_HUB_OFFLINE=1
.venv/bin/python research/build_both.py > runs/research/both_build.log 2>&1
go() { n=$1; shift; .venv/bin/python whyfix/generate.py "$@" --gsm-test 64 --gsm-start 448 --seed 11 --max 3000 --out runs/research/both_${n}_traces.jsonl > runs/research/both_$n.log 2>&1; }
go none --arm none
go doubt --arm dir_self_doubt
go rep --arm dir_neurospeak
go both --arm dir_both_rep --also dir_self_doubt
go random2 --arm dir_both_m1 --also dir_both_m2
