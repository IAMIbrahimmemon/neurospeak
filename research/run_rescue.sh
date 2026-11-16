#!/bin/bash
# One arm at a time, on the runs in PLAN_RESCUE.md.
set -e
for arm in none dir_self_doubt dir_neurospeak dir_self_doubt_matched; do
  HF_HUB_OFFLINE=1 .venv/bin/python research/rescue.py --arm $arm --out runs/research/rescue_${arm}_traces.jsonl > runs/research/rescue_$arm.log 2>&1
done
