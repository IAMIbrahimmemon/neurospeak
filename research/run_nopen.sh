#!/bin/bash
# Study 19, one arm, see PLAN_NOPEN.md. Resumes a partial run and retries a killed one.
export HF_HUB_OFFLINE=1
f=runs/research/head_doubt_traces.jsonl
for try in 1 2 3 4 5; do
  [ -f "$f" ] && [ "$(wc -l < "$f")" -eq 96 ] && { echo "$(date) doubt done" >> runs/research/nopen_restarts.log; exit 0; }
  echo "$(date) doubt try $try from $( [ -f "$f" ] && wc -l < "$f" || echo 0)" >> runs/research/nopen_restarts.log
  .venv/bin/python whyfix/generate.py --arm dir_self_doubt --gsm-test 96 --gsm-start 704 --seed 18 --max 9000 --resume --out "$f" >> runs/research/head_doubt.log 2>&1
done
echo "$(date) doubt gave up" >> runs/research/nopen_restarts.log; exit 1
