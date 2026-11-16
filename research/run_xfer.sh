#!/bin/bash
# Study 21, see PLAN_TRANSFER.md; run with OUT=xfer. Arms given as "name layer" pairs; resumes partial runs and retries killed ones.
export HF_HUB_OFFLINE=1
for spec in "$@"; do
  set -- $spec; arm=$1; layer=$2
  f=runs/research/${OUT:-layers}_${arm}${layer:+_at$layer}_traces.jsonl
  for try in 1 2 3 4 5; do
    [ -f "$f" ] && [ "$(wc -l < "$f")" -eq 64 ] && break
    echo "$(date) $arm try $try from $( [ -f "$f" ] && wc -l < "$f" || echo 0)" >> runs/research/${OUT:-layers}_restarts.log
    extra=""; [ -n "$layer" ] && extra="--layer $layer"
    .venv/bin/python whyfix/generate.py --arm "$arm" $extra --gsm-test 64 --gsm-start 800 --seed 20 --max 3000 --resume --out "$f" >> runs/research/${OUT:-layers}_${arm}${layer:+_at$layer}.log 2>&1
  done
  [ "$(wc -l < "$f")" -eq 64 ] || { echo "$(date) $arm gave up" >> runs/research/${OUT:-layers}_restarts.log; exit 1; }
  echo "$(date) $arm done" >> runs/research/${OUT:-layers}_restarts.log
done
