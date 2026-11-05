"""Builds C4: which L15 word feeds this L19 word?  .venv/bin/python tracetest/make_reading.py"""
import json, os, random, re
import torch

import neurospeak as ns
from neurospeak.core import DEV
from neurospeak.trace import child_after, parents

PARENT, CHILD, N, CONTEXT, SHOW, MIN_LABEL, BATCH = 15, 19, 100, 128, 64, 12, 40
OUT = "runs/tracetest"
WORDS = "runs/words/Qwen__Qwen3.5-0.8B/L{}.json"


def views(S, tok, L, need):
    T, SEQ = S["tokens"].view(-1), S["seq"]
    half = S["half"].repeat_interleave(SEQ)
    U = S["units"][L]
    rng = random.Random(22)

    def snippet(p):
        s, e = max(p - p % SEQ, p - 16), min(p - p % SEQ + SEQ, p + 4)
        f = lambda x, y: tok.decode(T[x:y].tolist()).replace("\n", "⏎")
        return f"{f(s, p)}[[{f(p, p + 1)}]]{f(p + 1, e)}"

    out, skipped = [], []
    for f in need:
        m = U["idx"].long() == f
        rows = m.any(1)
        v = torch.zeros(len(T))
        v[rows] = (U["vals"][rows].float() * m[rows].float()).sum(1)
        v[(torch.arange(len(T)) % SEQ) < S["skip"]] = 0
        fired = ((v > 0) & (half == 0)).nonzero().squeeze(1)
        if len(fired) < MIN_LABEL:
            skipped.append(f)
            continue
        order = fired[v[fired].argsort(descending=True)].tolist()
        weak = order[len(order) // 2:]
        rng.shuffle(weak)
        quiet = []
        for q in order[:10]:
            same = ((T == T[q]) & (v == 0) & (half == 0)).nonzero().squeeze(1).tolist()
            rng.shuffle(same)
            quiet += same[:1]
        out.append(dict(id=f"L{L}F{f}", strongest=[dict(act=round(float(v[q]), 2), text=snippet(q)) for q in order[:10]],
                        weaker=[dict(act=round(float(v[q]), 2), text=snippet(q)) for q in weak[:6]], does_not_fire=[snippet(q) for q in quiet[:4]]))
    return out, skipped


def main():
    S = torch.load("runs/wordtest/scan.pt", map_location="cpu")
    brain = ns.load(dtype=torch.float32)
    tok = brain.tok
    R = [json.loads(l) for l in open(f"{OUT}/mech.jsonl")]
    trials = []
    for r in R:
        if len(trials) >= N:
            break
        ids = S["tokens"][r["chunk"], r["pos"] - CONTEXT + 1:r["pos"] + 1].long().to(DEV)[None]
        pos, child = r["child_pos"], r["child"]
        top = r["parents"][0]
        vals, idx, _ = parents(brain, ids, PARENT, CHILD, pos, child)
        near = [(int(f), float(v)) for f, v in zip(idx[top[0]].tolist(), vals[top[0]].tolist()) if v >= .5 * top[2] and f != top[1]]
        if not near:
            continue
        got = child_after(brain, ids, PARENT, CHILD, pos, child, [[]] + [[(top[0], f, v)] for f, v in near])
        limit = .1 * abs(r["singles"][0])
        decoys = [f for (f, _), g in zip(near, got[1:]) if abs(got[0] - g) < limit]
        if decoys:
            trials.append(dict(chunk=r["chunk"], pos=r["pos"], context=tok.decode(ids[0, -SHOW:].tolist()).replace("\n", "⏎"), child=child, parent=top[1], decoy=decoys[0], parent_effect=r["singles"][0]))
    json.dump(trials, open(f"{OUT}/reading.json", "w"), indent=1)
    print("trials", len(trials), "of", len(R))

    units = json.load(open("runs/wordtest/units.json"))["records"]
    for L in (PARENT, CHILD):
        path = WORDS.format(L)
        names = json.load(open(path)) if os.path.exists(path) else {}
        for p in sorted(os.listdir("runs/wordtest")):
            if re.fullmatch(r"labels_\d+\.json", p):
                for row in json.load(open(f"runs/wordtest/{p}")):
                    u = units[int(row["id"][1:])]
                    if u["kind"] == "feature" and u["layer"] == L:
                        names.setdefault(str(u["index"]), dict(fires_on=row["fires_on"], effect=row["effect"], confidence=row["confidence"]))
        json.dump(names, open(path, "w"), indent=1, ensure_ascii=False)
    have = {L: json.load(open(WORDS.format(L))) for L in (PARENT, CHILD)}
    need15 = sorted({f for t in trials for f in (t["parent"], t["decoy"])} - {int(k) for k in have[15]})
    need19 = sorted({t["child"] for t in trials} - {int(k) for k in have[19]})
    v15, s15 = views(S, tok, 15, need15)
    v19, s19 = views(S, tok, 19, need19)
    all_views = v15 + v19
    random.Random(23).shuffle(all_views)
    n = 0
    for s in range(0, len(all_views), BATCH):
        json.dump(all_views[s:s + BATCH], open(f"{OUT}/label_batch_{n}.json", "w"), indent=1, ensure_ascii=False)
        n += 1
    print("need L15", len(need15), "L19", len(need19), "batches", n, "too few firings", s15, s19)


if __name__ == "__main__":
    main()
