"""Checks whether the self-doubt arm's correct count survives stricter scoring, and what its runs do after boxing an answer."""
import json, re, sys, random
from neurospeak import loopiness
src = open("whyfix/score_self_doubt.py").read()
exec(src[src.index("def boxed"):src.index("def ci")])
arms = ["none", "dir_neurospeak", "dir_self_doubt", "dir_self_doubt_matched", "dir_self_doubt_orth"]
R = {a: [json.loads(l) for l in open(f"runs/whyfix/gsm64_{a}_traces.jsonl")] for a in arms}
BOX = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
out = {}
for a, rows in R.items():
    c = dict(last_box=0, first_box=0, finished_and_right=0, cut_and_right=0, right_all_boxes_agree=0, right_after_think=0, right_inside_think=0, leaves_think=0, chars_after_box=[], rep_after_box=[])
    for r in rows:
        t = r["completion"]; ans = norm(r["answer"])
        boxes = [x.strip() for x in re.findall(BOX, t) if x.strip()]
        c["leaves_think"] += "</think>" in t
        if not boxes: continue
        c["first_box"] += norm(boxes[0]) == ans
        if norm(boxes[-1]) != ans: continue
        c["last_box"] += 1
        c["cut_and_right" if r["cut"] else "finished_and_right"] += 1
        c["right_all_boxes_agree"] += all(norm(b) == ans for b in boxes)
        end = [m for m in re.finditer(BOX, t) if m.group(1).strip()][-1]
        think = t.find("</think>")
        c["right_after_think" if think != -1 and end.start() > think else "right_inside_think"] += 1
        c["chars_after_box"].append(len(t) - end.end())
    c["chars_after_box"] = sorted(c["chars_after_box"])
    out[a] = c
    print(a, json.dumps(c))
s, n = R["dir_self_doubt"], R["none"]
wins = [i for i, (x, y) in enumerate(zip(s, n)) if norm(boxed(x["completion"])) == norm(x["answer"]) and norm(boxed(y["completion"])) != norm(y["answer"])]
print("self-doubt wins", len(wins), [s[i]["id"] for i in wins])
for i in wins[:3] + wins[-2:]:
    x = s[i]; t = x["completion"]; m = [q for q in re.finditer(BOX, t) if q.group(1).strip()][-1]
    print("\n=====", x["id"], "answer", x["answer"], "cut", x["cut"], "tokens", x["n_tokens"], "box at", m.start(), "of", len(t))
    print("Q:", x["problem"][:220]); print("BEFORE BOX:", t[max(0, m.start() - 500):m.end()]); print("AFTER BOX:", repr(t[m.end():m.end() + 350]))
    print("UNEDITED tail:", repr(n[i]["completion"][-200:]))
