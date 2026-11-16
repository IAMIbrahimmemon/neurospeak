#!/bin/bash
# The amended arm, after run_add.sh, see PLAN_ADD.md.
set -e
export HF_HUB_OFFLINE=1
read -r dir amount < <(.venv/bin/python -c "import json;a=json.load(open('runs/research/add_gains.json'))['arms']['random_low'];print(a['direction'],a['amount'])")
.venv/bin/python whyfix/generate.py --arm "$dir" --add "$amount" --gsm-test 48 --gsm-start 304 --seed 8 --max 3000 --out runs/research/add_random_low_traces.jsonl > runs/research/add_random_low.log 2>&1
