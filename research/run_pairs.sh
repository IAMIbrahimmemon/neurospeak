#!/bin/bash
# One arm at a time, on the problems in PLAN_PAIRS.md.
set -e
mkdir -p runs/research
for arm in none dir_writer_2 dir_writer_3 dir_self_doubt_matched; do
  HF_HUB_OFFLINE=1 .venv/bin/python whyfix/generate.py --arm $arm --gsm-test 64 --gsm-start 240 --seed 6 --max 3000 --out runs/research/pairs_${arm}_traces.jsonl > runs/research/pairs_$arm.log 2>&1
done
