#!/bin/bash
# Gate arm, then directions, then arms, see PLAN_REP_PORT.md.
set -e
export HF_HUB_OFFLINE=1
go() { .venv/bin/python research/port_gen.py --arm "$1" --limit 64 --start 384 --seed 10 --out runs/research/rport_$1_traces.jsonl > runs/research/rport_$1.log 2>&1; }
go none
.venv/bin/python research/build_rep_port.py > runs/research/rport_build.log 2>&1
go rep
go rep_matched
