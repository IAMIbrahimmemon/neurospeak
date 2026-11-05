"""Adds each tested word's scores to its record in runs/words/. Rerun whenever names are added; it only rewrites `tests`.  python3 wordtest/build_records.py"""
import json

D = "runs/wordtest"
units = {r["id"]: r for r in json.load(open(f"{D}/units.json"))["records"]}
read = {r["unit"]: r for r in json.load(open(f"{D}/results.json"))["rows"] if r["cond"] == "real" and r["kind"] == "feature"}
steer = {}
for r in json.load(open(f"{D}/replication_results.json"))["rows"]:
    if r["cond"] == "steer_real" and r["kind"] == "feature":
        steer.setdefault(r["unit"], []).append(bool(r["ok"]))
for L in (15, 19):
    path = f"runs/words/Qwen__Qwen3.5-0.8B/L{L}.json"
    names = json.load(open(path))
    tested = 0
    for uid, u in units.items():
        if u["kind"] != "feature" or u["layer"] != L or str(u["index"]) not in names:
            continue
        rec = names[str(u["index"])]
        r = read.get(uid)
        rec["tests"] = dict(read_balanced=r["balanced"] if r else None, strong_found=r["positive_top"] if r else None, weak_found=r["positive_weak"] if r else None,
                            same_token_rejected=r.get("same_token") if r else None, effect_right=sum(steer.get(uid, [])), effect_trials=len(steer.get(uid, [])))
        tested += 1
    json.dump(names, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"L{L}: {len(names)} names, {tested} with test scores")
