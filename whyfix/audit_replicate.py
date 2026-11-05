"""Backs the replication write-up's prose counts, and tests self-doubt against repetition on the stricter count."""
import json, re, sys
sys.path.insert(0, "whyfix")
from gsm_score import ci
from sensitivity_replicate_rules import right

BOX = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
ARMS = ("none", "dir_self_doubt", "dir_neurospeak", "dir_self_doubt_matched")
out = {}
for prefix in ("gsm64", "gsm64b"):
    R = {a: [json.loads(l) for l in open(f"runs/whyfix/{prefix}_{a}_traces.jsonl")] for a in ARMS}
    fc = {a: [100 * (right(r) and not r["cut"]) for r in v] for a, v in R.items()}
    out[prefix] = dict(finished_correct_vs_repetition=ci(fc["dir_self_doubt"], fc["dir_neurospeak"]))
    for a, v in R.items():
        c = dict(correct_cut=0, correct_cut_box_inside_thinking=0, answer_after_equals=0)
        for r in v:
            t = r["completion"]
            c["answer_after_equals"] += bool(re.search(r"=\s*\$?" + re.escape(r["answer"]) + r"(?![\d.])", t.replace(",", "")))
            if right(r) and r["cut"]:
                c["correct_cut"] += 1
                last = [m for m in re.finditer(BOX, t) if m.group(1).strip()][-1]
                close = t.find("</think>")
                c["correct_cut_box_inside_thinking"] += close == -1 or last.start() < close
        out[prefix][a] = c
json.dump(out, open("runs/whyfix/replicate_audit.json", "w"), indent=1)
print(json.dumps(out, indent=0))
