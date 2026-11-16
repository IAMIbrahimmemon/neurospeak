#!/bin/bash
# Fixes the gains first, then one arm at a time. See PLAN_DOSE.md.
set -e
export HF_HUB_OFFLINE=1
test -f runs/research/dose_gains.json || .venv/bin/python research/dose_calibrate.py > runs/research/dose_calibrate.log 2>&1
for arm in matched_mean doubt_dose random_dose rep_only doubt_only; do
  read dir gain < <(.venv/bin/python -c "import json;a=json.load(open('runs/research/dose_gains.json'))['arms']['$arm'];print(a['direction'],a['gain'])")
  .venv/bin/python research/rescue.py --arm $dir --gain $gain --out runs/research/dose_${arm}_traces.jsonl > runs/research/dose_$arm.log 2>&1
done
