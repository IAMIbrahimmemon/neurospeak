"""Rescores both 64-problem tests comparing numbers by value, so 12.00 equals 12. A check on the frozen scorer, which decides the bars."""
import json, sys
sys.path.insert(0, "whyfix")
from gsm_score import ci


from sensitivity_replicate_rules import right


ARMS = ("none", "dir_self_doubt", "dir_neurospeak", "dir_self_doubt_matched")
total = {a: 0 for a in ARMS}
out = {}
for prefix in ("gsm64", "gsm64b"):
    rows = {a: [json.loads(l) for l in open(f"runs/whyfix/{prefix}_{a}_traces.jsonl")] for a in ARMS}
    s = {a: [100 * right(r) for r in v] for a, v in rows.items()}
    out[prefix] = {a: sum(x > 0 for x in v) for a, v in s.items()}
    out[prefix + "_vs_unedited"] = ci(s["dir_self_doubt"], s["none"])
    out[prefix + "_vs_matched"] = ci(s["dir_self_doubt"], s["dir_self_doubt_matched"])
    out[prefix + "_vs_repetition"] = ci(s["dir_self_doubt"], s["dir_neurospeak"])
    for a in ARMS:
        total[a] += out[prefix][a]
out["both_of_128"] = total
json.dump(out, open("runs/whyfix/replicate_sensitivity.json", "w"), indent=1)
print(json.dumps(out))
