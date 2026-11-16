#!/bin/bash
# One arm at a time, see PLAN_TILT.md. Gains are frozen in runs/research/tilt_gains.json.
set -e
export HF_HUB_OFFLINE=1
P=$(.venv/bin/python -c "import json;print(json.load(open('runs/research/tilt_gains.json'))['gains']['perp_eq'])")
R=$(.venv/bin/python -c "import json;print(json.load(open('runs/research/tilt_gains.json'))['activity']['dir_tilt']['mean_abs'])")
E=$(.venv/bin/python -c "import json,math;a=json.load(open('runs/research/tilt_gains.json'))['activity'];print(a['dir_tilt']['mean_abs']/(a['only']['mean_abs']*math.cos(math.radians(42))))")
go() { n=$1; shift; [ "$(wc -l < runs/research/tilt_${n}_traces.jsonl 2>/dev/null || echo 0)" -eq "$(wc -l < runs/research/tilt_prefixes.jsonl)" ] && return 0
  .venv/bin/python research/rescue.py --runs runs/research/tilt_prefixes.jsonl --seed 16 "$@" --out runs/research/tilt_${n}_traces.jsonl > runs/research/tilt_$n.log 2>&1; }
go none --arm none
go only --arm dir_rep_only
go tilt --arm dir_tilt
go only_eq --arm dir_rep_only --gain "$E"
go both --arm dir_rep_only --also dir_self_doubt
go perp_eq --arm dir_perp --gain "$P"
go rand_eq --arm dir_rand --shift "$R"
