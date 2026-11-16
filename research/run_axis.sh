#!/bin/bash
# One arm at a time, see PLAN_AXIS.md.
set -e
export HF_HUB_OFFLINE=1
go() { n=$1; shift; .venv/bin/python research/rescue.py "$@" --out runs/research/axis_${n}_traces.jsonl > runs/research/axis_$n.log 2>&1; }
go rep_25 --arm dir_neurospeak --gain 0.25
go rep_50 --arm dir_neurospeak --gain 0.5
go rep_75 --arm dir_neurospeak --gain 0.75
go only_109 --arm dir_rep_only --gain 1.09
go both --arm dir_rep_only --gain 1 --also dir_self_doubt
