#!/bin/bash
# Generates the three new arms one at a time, on the same problems and seed as the repetition direction's test.
set -e
for arm in dir_self_doubt dir_self_doubt_matched dir_self_doubt_orth; do
  .venv/bin/python whyfix/generate.py --arm $arm --gsm-test 64 --seed 3 --out runs/whyfix/gsm64_${arm}_traces.jsonl > runs/whyfix/generate_gsm64_$arm.log 2>&1
done
