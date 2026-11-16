#!/bin/bash
# One arm at a time, see PLAN_FINISH.md.
set -e
export HF_HUB_OFFLINE=1
go() { n=$1; shift; .venv/bin/python whyfix/generate.py "$@" --gsm-test 64 --gsm-start 640 --seed 17 --max 9000 --out runs/research/finish_${n}_traces.jsonl > runs/research/finish_$n.log 2>&1; }
go rep --arm dir_neurospeak
go doubt --arm dir_self_doubt
go both --arm dir_both_rep --also dir_self_doubt
