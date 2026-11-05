#!/bin/bash
# Generates the four arms one at a time on the unused problems in PLAN_REPLICATE.md.
set -e
for arm in none dir_self_doubt dir_neurospeak dir_self_doubt_matched; do
  .venv/bin/python whyfix/generate.py --arm $arm --gsm-test 64 --gsm-start 144 --seed 4 --out runs/whyfix/gsm64b_${arm}_traces.jsonl > runs/whyfix/generate_gsm64b_$arm.log 2>&1
done
