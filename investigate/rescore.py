"""Rechecks the intact check on saved runs with the fixed collapse detector, keeping the status each run printed as run."""
import glob, json
from neurospeak.investigate import collapsed

rows = {}
for f in sorted(glob.glob("runs/investigate/*.json")):
    r = json.load(open(f))
    if "behavior" not in r:
        continue
    dead = {k: sum(collapsed(t) for t in v) for k, v in r["texts"].items()}
    checks = {c["name"]: c["passed"] for c in r["checks"]}
    checks["The output stays intact"] = dead["removed"] <= dead["unedited"]
    status = "could not test" if not checks["The behavior is present"] else "works" if all(checks.values()) else "did not pass"
    name = f.split("/")[-1][:-5]
    rows[name] = dict(status_as_run=r["status"], status=status, unedited=r["unedited"], removed=r["removed"], control=r["control"], collapsed=dead,
                      failed=[k for k, v in checks.items() if not v], cost=next((c["got"] for c in r["checks"] if "cheap" in c["name"]), None))
    print(f"{name:26s} as run: {r['status']:14s} now: {status:14s} {r['unedited']:.3g} -> {r['removed']:.3g} (control {r['control']:.3g}) collapsed {dead} cost {rows[name]['cost']}")
json.dump(rows, open("runs/investigate/summary.json", "w"), indent=1)
