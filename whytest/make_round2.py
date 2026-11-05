"""Builds the two reading tests and the batch of words that need names.  .venv/bin/python whytest/make_reading.py"""
import json, os, random, re
import torch

import neurospeak as ns
from neurospeak.core import DEV
from neurospeak.why import attribute, removed_logits

L, N_R1, N_R2, CONTEXT, SHOW, MIN_LABEL, BATCH = 19, 120, 100, 128, 64, 12, 40
OUT = "runs/whytest/round2"
WORDS = f"runs/words/{ns.DEFAULT_MODEL.replace('/', '__')}/L{L}.json"


def main():
    S = torch.load("runs/wordtest/scan.pt", map_location="cpu")
    brain = ns.load(dtype=torch.float32)
    tok = brain.tok
    R = [json.loads(l) for l in open(f"runs/whytest/L{L}.jsonl")]
    first = json.load(open("runs/whytest/reading.json"))
    used = {(t["chunk"], t["pos"]) for t in first["r1"] + first["r2"]}
    R = [r for r in R if (r["chunk"], r["pos"]) not in used]
    print("unused decisions", len(R))
    random.Random(23).shuffle(R)
    show = lambda t: repr(tok.decode([t]))[1:-1]
    r1, r2 = [], []
    for r in R:
        if len(r1) >= N_R1 and len(r2) >= N_R2:
            break
        c, p, ans, alt = r["chunk"], r["pos"], r["answer"], r["against"]
        ids = S["tokens"][c, p - CONTEXT + 1:p + 1].long().to(DEV)[None]
        context = tok.decode(ids[0, -SHOW:].tolist()).replace("\n", "⏎")
        vals, idx, scores = attribute(brain, ids, L, ans, alt)
        common = dict(chunk=c, pos=p, source=r["source"], context=context, answer=show(ans), against=show(alt))
        if len(r2) < N_R2:
            k = scores.shape[1]
            best = scores.view(-1).abs().topk(5).indices.tolist()
            r2.append({**common, "words": [int(idx[i // k, i % k]) for i in best]})
        if len(r1) < N_R1:
            drivers = [s for s in r["single"] if s["measured"] <= -0.25]
            if not drivers:
                continue
            d = min(drivers, key=lambda s: s["measured"])
            pos = d["pos"]
            near = [(int(f), float(v)) for f, v in zip(idx[pos].tolist(), vals[pos].tolist()) if v >= .5 * d["v"] and f != d["f"]]
            if not near:
                continue
            base = brain.m(ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            lg = removed_logits(brain, ids, L, [[(pos, f, v)] for f, v in near])
            effect = ((lg[:, ans] - lg[:, alt]) - (base[ans] - base[alt])).abs().tolist()
            decoys = sorted((e, f) for e, (f, _) in zip(effect, near) if e < .1 * abs(d["measured"]))
            if decoys:
                r1.append({**common, "driver": d["f"], "decoys": [f for _, f in decoys[:2]], "driver_effect": d["measured"]})
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(r1=r1, r2=r2), open(f"{OUT}/reading.json", "w"), indent=1)
    print("trials", len(r1), len(r2))

    # names: start from the labels the word test already has
    names = {}
    if os.path.exists(WORDS):
        names = json.load(open(WORDS))
    units = json.load(open("runs/wordtest/units.json"))["records"]
    for path in sorted(os.listdir("runs/wordtest")):
        if re.fullmatch(r"labels_\d+\.json", path):
            for row in json.load(open(f"runs/wordtest/{path}")):
                u = units[int(row["id"][1:])]
                if u["kind"] == "feature" and u["layer"] == L:
                    names.setdefault(str(u["index"]), dict(fires_on=row["fires_on"], effect=row["effect"], confidence=row["confidence"]))
    os.makedirs(os.path.dirname(WORDS), exist_ok=True)
    json.dump(names, open(WORDS, "w"), indent=1)

    need = sorted({f for t in r1 for f in [t["driver"], *t["decoys"]]} | {f for t in r2 for f in t["words"]})
    need = [f for f in need if str(f) not in names]

    T = S["tokens"].view(-1)
    SEQ, half = S["seq"], S["half"].repeat_interleave(S["seq"])
    U = S["units"][L]
    rng = random.Random(22)

    def snippet(p):
        s, e = max(p - p % SEQ, p - 16), min(p - p % SEQ + SEQ, p + 4)
        f = lambda x, y: tok.decode(T[x:y].tolist()).replace("\n", "⏎")
        return f"{f(s, p)}[[{f(p, p + 1)}]]{f(p + 1, e)}"

    views, no_examples = [], []
    for f in need:
        m = U["idx"].long() == f
        rows = m.any(1)
        v = torch.zeros(len(T))
        v[rows] = (U["vals"][rows].float() * m[rows].float()).sum(1)
        v[(torch.arange(len(T)) % SEQ) < S["skip"]] = 0
        fired = ((v > 0) & (half == 0)).nonzero().squeeze(1)
        if len(fired) < MIN_LABEL:
            no_examples.append(f)
            continue
        order = fired[v[fired].argsort(descending=True)].tolist()
        top = order[:10]
        weak = [q for q in order[len(order) // 2:]]
        rng.shuffle(weak)
        weak = weak[:6]
        quiet = []
        for q in top:
            same = ((T == T[q]) & (v == 0) & (half == 0)).nonzero().squeeze(1).tolist()
            rng.shuffle(same)
            quiet += same[:1]
        views.append(dict(id=f"F{f}", strongest=[dict(act=round(float(v[q]), 2), text=snippet(q)) for q in top],
                          weaker=[dict(act=round(float(v[q]), 2), text=snippet(q)) for q in weak],
                          does_not_fire=[snippet(q) for q in quiet[:4]]))
    rng.shuffle(views)
    n = 0
    for s in range(0, len(views), BATCH):
        json.dump(views[s:s + BATCH], open(f"{OUT}/label_batch_{n}.json", "w"), indent=1, ensure_ascii=False)
        n += 1
    json.dump(no_examples, open(f"{OUT}/no_examples.json", "w"))
    print("words", len(need), "to name", len(views), "batches", n, "too few firings", len(no_examples))


if __name__ == "__main__":
    main()
