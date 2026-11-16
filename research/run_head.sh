#!/bin/bash
# One arm at a time, see PLAN_HEAD.md. Skips arms already complete, resumes a partial arm, and retries an arm that was killed.
export HF_HUB_OFFLINE=1
E=data/astra_p50c_edit.npz
go() {
  n=$1; shift; f=runs/research/head_${n}_traces.jsonl
  for try in 1 2 3 4 5; do
    [ -f "$f" ] && [ "$(wc -l < "$f")" -eq 96 ] && return 0
    echo "$(date) $n try $try from $( [ -f "$f" ] && wc -l < "$f" || echo 0)" >> runs/research/head_restarts.log
    .venv/bin/python whyfix/generate.py "$@" --gsm-test 96 --gsm-start 704 --seed 18 --max 9000 --resume --out "$f" >> runs/research/head_$n.log 2>&1
  done
  echo "$(date) $n gave up" >> runs/research/head_restarts.log; exit 1
}
go astra_pp --arm none --edit $E --presence 1.5
go none_pp --arm none --presence 1.5
go doubt_pp --arm dir_self_doubt --presence 1.5
go astra_doubt_pp --arm dir_self_doubt --edit $E --presence 1.5
go matched_pp --arm dir_self_doubt_matched --presence 1.5
go none --arm none
echo "$(date) all done" >> runs/research/head_restarts.log
