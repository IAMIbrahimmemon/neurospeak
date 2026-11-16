#!/bin/bash
# One arm at a time, on the problems in PLAN_LONG_CAP.md.
set -e
mkdir -p runs/research
for arm in none dir_self_doubt dir_self_doubt_matched; do
  .venv/bin/python whyfix/generate.py --arm $arm --gsm-test 32 --gsm-start 208 --seed 5 --max 10000 --out runs/research/long_${arm}_traces.jsonl > runs/research/long_$arm.log 2>&1
done
