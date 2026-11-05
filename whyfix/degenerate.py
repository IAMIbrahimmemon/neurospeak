"""Share of each arm's traces that collapse into gibberish, which the word-based repeat metric can't see.
A trace counts as collapsed if one character repeats 50 or more times in a row, or it is under 100 words at the cap."""
import glob, json, re, sys
from neurospeak import loopiness

out = {}
for f in sorted(glob.glob("runs/whyfix/*_traces.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    if not rows:
        continue
    bad = [bool(re.search(r"(\S)\1{49,}", r["completion"])) or (r["cut"] and len(r["completion"].split()) < 100) for r in rows]
    words = sum(len(r["completion"].split()) for r in rows) / len(rows)
    name = f.split("/")[-1].replace("_traces.jsonl", "")
    out[name] = dict(n=len(rows), collapsed=sum(bad), mean_words=round(words), repeats=round(100 * sum(loopiness(r["completion"]) for r in rows) / len(rows), 1))
    print(f"{name:28s} n={len(rows):3d} collapsed {sum(bad):3d}  mean words {words:6.0f}  repeats {out[name]['repeats']:5.1f}")
json.dump(out, open("runs/whyfix/degenerate.json", "w"), indent=1)
