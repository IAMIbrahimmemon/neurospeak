#!/bin/bash
# Calibration, then one arm at a time, on the problems in PLAN_ADD.md.
set -e
export HF_HUB_OFFLINE=1
.venv/bin/python research/add_calibrate.py > runs/research/add_calibrate.log 2>&1
go() { .venv/bin/python whyfix/generate.py --arm "$2" --add "$3" --gsm-test 48 --gsm-start 304 --seed 8 --max 3000 --out runs/research/add_$1_traces.jsonl > runs/research/add_$1.log 2>&1; }
go none none 0
for arm in doubt_low doubt_high random_high rep_high; do
  read -r dir amount < <(.venv/bin/python -c "import json;a=json.load(open('runs/research/add_gains.json'))['arms']['$arm'];print(a['direction'],a['amount'])")
  go $arm "$dir" "$amount"
done
