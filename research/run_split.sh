#!/bin/bash
# Study 23, see PLAN_SPLIT.md. Arms given as "label flags..."; every arm removes dir_self_doubt at L19. Resumes partial runs and retries killed ones.
export HF_HUB_OFFLINE=1
for spec in "$@"; do
  set -- $spec; label=$1; shift
  f=runs/research/split_${label}_traces.jsonl
  for try in 1 2 3 4 5; do
    [ -f "$f" ] && [ "$(wc -l < "$f")" -eq 64 ] && break
    echo "$(date) $label try $try from $( [ -f "$f" ] && wc -l < "$f" || echo 0)" >> runs/research/split_restarts.log
    .venv/bin/python whyfix/generate.py --arm dir_self_doubt --layer 19 "$@" --gsm-test 64 --gsm-start 800 --seed 20 --max 3000 --resume --out "$f" >> runs/research/split_${label}.log 2>&1
  done
  [ "$(wc -l < "$f")" -eq 64 ] || { echo "$(date) $label gave up" >> runs/research/split_restarts.log; exit 1; }
  echo "$(date) $label done" >> runs/research/split_restarts.log
done
