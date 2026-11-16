#!/bin/bash
# Study 28, see PLAN_BAN.md. Unedited model with doubt-word tokens banned. Resumes partial runs and retries killed ones.
export HF_HUB_OFFLINE=1
f=runs/research/ban_traces.jsonl
for try in 1 2 3 4 5; do
  [ -f "$f" ] && [ "$(wc -l < "$f")" -eq 64 ] && break
  echo "$(date) ban try $try from $( [ -f "$f" ] && wc -l < "$f" || echo 0)" >> runs/research/ban_restarts.log
  .venv/bin/python whyfix/generate.py --arm none --ban 13784,13428,11158,3655,77264,85152,84485,50821,32645,70546,3404 --gsm-test 64 --gsm-start 800 --seed 20 --max 3000 --resume --out "$f" >> runs/research/ban.log 2>&1
done
[ "$(wc -l < "$f")" -eq 64 ] || { echo "$(date) ban gave up" >> runs/research/ban_restarts.log; exit 1; }
echo "$(date) ban done" >> runs/research/ban_restarts.log
