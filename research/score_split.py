"""Judges bars S1-S6 in PLAN_SPLIT.md (study 23) and L1-L5 in PLAN_LEVEL.md (study 24; L5 is S6 for the 049 arms). Unedited and full removal are study 20's traces, controls are study 22's, all on the same problems."""
import json, os, re, sys
from neurospeak.investigate import collapsed

sys.path[:0] = ["investigate", "whyfix"]
from behaviors import DOUBT
from gsm_score import ci
from sensitivity_replicate_rules import right

P = "runs/research/"
SPLIT = {"gain065": dict(gain=0.65, mode="full"), "gain035": dict(gain=0.35, mode="full"),
         "centered": dict(gain=1.0, mode="centered"), "offset": dict(gain=1.0, mode="offset"),
         "offset049": dict(gain=1.0, mode="offset", level=0.49), "centered049": dict(gain=1.0, mode="centered", level=0.49)}
CTLS = [f"ctl{s}" for s in (101, 102, 103, 104)]
files = {"none": P + "layers_none_traces.jsonl", "full": P + "layers_dir_layer_19_traces.jsonl",
         **{c: P + f"dosectl_dir_dosectl_{c[3:]}_at19_traces.jsonl" for c in CTLS},
         **{s: P + f"split_{s}_traces.jsonl" for s in SPLIT}}
R = {a: [json.loads(x) for x in open(f)] for a, f in files.items() if os.path.exists(f)}
R = {a: v for a, v in R.items() if len(v) == 64}
assert all(c in R for c in ["none", "full", *CTLS])
assert all([r["id"] for r in v] == [r["id"] for r in R["none"]] for v in R.values())
for s, want in SPLIT.items():
    if s in R:
        assert all(r["run"]["arm"] == "dir_self_doubt" and r["run"]["layer_used"] == 19 and r["run"]["max"] == 3000
                   and r["run"]["seed"] == 20 and not r["run"]["add"] and r["run"]["gain"] == want["gain"]
                   and r["run"]["mode"] == want["mode"] and r["run"].get("level") == want.get("level") for r in R[s]), s
mean = lambda v: sum(v) / len(v)
ok = {a: [100 * right(r) for r in v] for a, v in R.items()}
last_box = lambda t: (re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", t) or ["x"])[-1].strip()
nonempty = {a: [100 * (right(r) and last_box(r["completion"]) != "") for r in v] for a, v in R.items()}
fin = {a: [100 * (right(r) and not r["cut"]) for r in v] for a, v in R.items()}
doubt = {a: mean([100 * len(DOUBT.findall(r["completion"])) / max(len(r["completion"].split()), 1) for r in v]) for a, v in R.items()}
wait_k = {a: mean([1000 * len(re.findall(r"\bwait\b", r["completion"], re.I)) / max(r["n_tokens"], 1) for r in v]) for a, v in R.items()}
bad = lambda r: bool(re.search(r" {200,}", r["completion"])) or collapsed(r["completion"])
arms = {a: dict(correct=sum(x > 0 for x in ok[a]), right_last_box_nonempty=sum(x > 0 for x in nonempty[a]),
                finished=sum(not r["cut"] for r in R[a]), finished_right=sum(x > 0 for x in fin[a]),
                doubt_per_100=round(doubt[a], 3), wait_per_1k=round(wait_k[a], 2), bad=sum(bad(r) for r in R[a]),
                tokens=sum(r["n_tokens"] for r in R[a]) // 64) for a in R}
pool = [mean([ok[c][i] for c in CTLS]) for i in range(64)]
have = lambda *a: all(x in R for x in a)
out = dict(arms=arms, done=[s for s in SPLIT if s in R])
if have("gain065"):
    out["S1"] = ci(ok["gain065"], pool)
out["S2"] = {s: dict(minus_unedited=ci(ok[s], ok["none"]), minus_full=ci(ok[s], ok["full"])) for s in ("gain035", "gain065") if s in R}
if have("centered"):
    out["S3"] = ci(ok["centered"], ok["none"])
    out["S3_minus_full"] = ci(ok["centered"], ok["full"])
if have("offset"):
    out["S4"] = ci(ok["offset"], ok["none"])
    out["S4_minus_full"] = ci(ok["offset"], ok["full"])
if have("centered", "offset"):
    out["S5"] = ci(ok["centered"], ok["offset"])
if have("offset049"):
    out["L1"] = ci(ok["offset049"], ok["none"])
    out["L2"] = ci(ok["offset049"], ok["gain065"])
if have("centered049"):
    out["L3"] = ci(ok["centered049"], ok["full"])
    out["L4"] = ci(ok["centered049"], ok["centered"])
out["S6"] = {s: dict(bad=arms[s]["bad"], unedited_bad=arms["none"]["bad"], ok=arms[s]["bad"] <= arms["none"]["bad"] + 2) for s in SPLIT if s in R}
out["pass"] = {k: v for k, v in dict(
    S1=out["S1"]["low"] > 0 if "S1" in out else None,
    S3=out["S3"]["low"] > 0 if "S3" in out else None,
    S5=("centered" if out["S5"]["low"] > 0 else "offset" if out["S5"]["high"] < 0 else "neither") if "S5" in out else None,
    S6=all(v["ok"] for v in out["S6"].values())).items() if v is not None}
json.dump(out, open(P + "split_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
print(json.dumps(arms, indent=1))
