"""Mechanics test for brain.trace. Appends one line per decision, so it can stop and resume.  .venv/bin/python tracetest/run.py"""
import json, os, random
import torch

import neurospeak as ns
from neurospeak.core import DEV
from neurospeak.trace import child_after, lead_with_held, parents, sources

PARENT, CHILD, N, CONTEXT, OUT = 15, 19, 150, 128, "runs/tracetest/mech.jsonl"


def main():
    S = torch.load("runs/wordtest/scan.pt", map_location="cpu")
    brain = ns.load(dtype=torch.float32)
    R = [json.loads(l) for l in open("runs/whytest/L19.jsonl")]
    random.Random(31).shuffle(R)
    R = [r for r in R if r["single"]][:N]
    done = {(json.loads(l)["chunk"], json.loads(l)["pos"]) for l in open(OUT)} if os.path.exists(OUT) else set()
    rng = random.Random(32)
    f_out = open(OUT, "a")
    for r in R:
        if (r["chunk"], r["pos"]) in done:
            continue
        ids = S["tokens"][r["chunk"], r["pos"] - CONTEXT + 1:r["pos"] + 1].long().to(DEV)[None]
        child = min(r["single"], key=lambda s: s["measured"])
        pos, f = child["pos"], child["f"]
        vals, idx, scores = parents(brain, ids, PARENT, CHILD, pos, f)
        k, flat = scores.shape[1], scores.view(-1)
        word = lambda i: (i // k, int(idx[i // k, i % k]), float(vals[i // k, i % k]))
        tops = [word(i) for i in flat.topk(5).indices.tolist() if flat[i] > 0]
        if len(tops) < 3:
            continue
        here = torch.arange(pos * k, (pos + 1) * k, device=DEV)
        v = vals.view(-1)
        same = [word(i) for i in here[v[here].topk(3).indices].tolist() if v[i] > 0]
        T = ids.shape[1]
        pool = [i for i in range((T - 8) * k, T * k) if v[i] > 0]
        rand = [word(i) for i in rng.sample(pool, 3)]
        lists = {"none": [], "parents": tops[:3], "active_same": same, "random_word": rand}
        got = child_after(brain, ids, PARENT, CHILD, pos, f, list(lists.values()) + [[w] for w in tops])
        base = got[0]
        drops = {n: base - got[j] for j, n in enumerate(lists)}
        singles = [base - x for x in got[len(lists):]]

        lead_cut = lead_with_held(brain, ids, PARENT, CHILD, tops, None, r["answer"], r["against"])
        lead_held = lead_with_held(brain, ids, PARENT, CHILD, tops, (pos, f), r["answer"], r["against"])
        others = [g for g in idx[pos].tolist() if g != f]
        other = rng.choice(others)
        lead_other = lead_with_held(brain, ids, PARENT, CHILD, tops, (pos, other), r["answer"], r["against"])

        src = sources(brain, ids, CHILD, pos, f)
        f_out.write(json.dumps(dict(chunk=r["chunk"], pos=r["pos"], source=r["source"], answer=r["answer"], against=r["against"], child=f, child_pos=pos,
                                    child_base=base, drops=drops, singles=singles, parents=tops, gap=r["gap"], lead_cut=lead_cut, lead_held=lead_held, lead_other=lead_other,
                                    last_token=T - 1, top_source=src[0], source_fall_at_last=next(x for p, x in src if p == pos))) + "\n")
        f_out.flush()
    print("done")


if __name__ == "__main__":
    main()
