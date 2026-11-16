#!/bin/bash
# See PLAN_REP_PORT2.md.
set -e
export HF_HUB_OFFLINE=1
for arm in none rep rep_matched; do
  .venv/bin/python research/port_gen.py --arm $arm --limit 128 --start 512 --seed 12 --out runs/research/rport2_${arm}_traces.jsonl > runs/research/rport2_$arm.log 2>&1
done
