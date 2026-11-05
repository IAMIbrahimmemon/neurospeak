"""Mechanics test for brain.why. Appends one line per decision, so it can stop and resume.

    .venv/bin/python whytest/run.py --layer 19 --half 1 --n 300 --out runs/whytest/L19.jsonl
"""
import argparse, json, os, random
import torch

import neurospeak as ns
from neurospeak.core import DEV
from neurospeak.why import attribute, removed_logits

CONTEXT, MIN_START, TOP_SINGLE, K = 128, 128, 8, 5
CURVE = [1, 2, 5, 10, 20, 50]
SHARE = {"wiki": 0.5, "gsm8k": 0.25, "trace": 0.25}


def candidates(S, half, seed):
    """Random (chunk, position) points per source, in a fixed order."""
    rng = random.Random(seed)
    out = {s: [] for s in SHARE}
    for c, (src, h) in enumerate(zip(S["source"], S["half"].tolist())):
        if h == half:
            for p in range(MIN_START, S["seq"] - 1, 32):
                out[src].append((c, p))
    for v in out.values():
        rng.shuffle(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--half", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=ns.DEFAULT_MODEL)
    ap.add_argument("--scan", default="runs/wordtest/scan.pt")
    a = ap.parse_args()
    S = torch.load(a.scan, map_location="cpu")
    brain = ns.load(a.model, dtype=torch.float32)
    done = [json.loads(l) for l in open(a.out)] if os.path.exists(a.out) else []
    have = {(r["chunk"], r["pos"]) for r in done}
    gen = torch.Generator().manual_seed(7)
    rng = random.Random(8)
    pools = candidates(S, a.half, seed=9)
    want = {s: round(a.n * x) for s, x in SHARE.items()}
    counts = {s: sum(r["source"] == s for r in done) for s in SHARE}
    F = brain.features[a.layer]
    f = open(a.out, "a")
    for src in SHARE:
        for c, p in pools[src]:
            if counts[src] >= want[src]:
                break
            if (c, p) in have:
                continue
            ids = S["tokens"][c, p - CONTEXT + 1:p + 1].long().to(DEV)[None]
            with torch.no_grad():
                base = brain.m(ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            prob = base.softmax(-1)
            top = base.topk(2).indices.tolist()
            ans, alt = top
            if not 0.1 <= float(prob[ans]) <= 0.9:
                continue
            gap0 = float(base[ans] - base[alt])
            vals, idx, scores = attribute(brain, ids, a.layer, ans, alt)
            k = scores.shape[1]
            T = ids.shape[1]

            def word(i):
                return (i // k, int(idx[i // k, i % k]), float(vals[i // k, i % k]))

            order = [i for i in scores.view(-1).topk(max(CURVE)).indices.tolist() if scores.view(-1)[i] > 0]
            why_words = [word(i) for i in order]
            active = (vals.view(-1) > 0)
            flat = vals.view(-1)
            last = torch.arange((T - 1) * k, T * k, device=DEV)
            last8 = torch.arange((T - 8) * k, T * k, device=DEV)
            active_last = [word(i) for i in last[flat[last].topk(K).indices].tolist() if flat[i] > 0]
            active_any = [word(i) for i in flat.topk(K).indices.tolist()]
            pool8 = [i for i in last8.tolist() if active[i]]
            random_word = [word(i) for i in rng.sample(pool8, min(K, len(pool8)))]
            dirs = []
            for pos, _, v in why_words[:K]:
                d = torch.randn(brain.anatomy.hidden_size, generator=gen).to(DEV)
                dirs.append((pos, d / d.norm(), v))
            arms = {"why": why_words[:K], "active_last": active_last, "active_any": active_any,
                    "random_word": random_word, "random_dir": dirs}
            singles = [[w] for w in why_words[:TOP_SINGLE]]
            curve = [why_words[:n] for n in CURVE]
            names = list(arms)
            lists = [arms[n] for n in names] + singles + curve
            lg = removed_logits(brain, ids, a.layer, lists)
            gaps = (lg[:, ans] - lg[:, alt]).tolist()
            flips = (lg.argmax(-1) != ans).tolist()
            n = len(names)
            est = [float(scores.view(-1)[i]) for i in order[:TOP_SINGLE]]
            rec = dict(chunk=c, pos=p, source=src, p=float(prob[ans]), gap=gap0, answer=ans, against=alt,
                       n_positive=int((scores > 0).sum()),
                       arms={m: dict(drop=gap0 - gaps[j], flip=flips[j]) for j, m in enumerate(names)},
                       single=[dict(pos=w[0], f=w[1], v=w[2], est=e, measured=g - gap0)
                               for w, e, g in zip(why_words, est, gaps[n:n + len(singles)])],
                       curve=[dict(n=m, drop=gap0 - g, flip=fl) for m, g, fl in zip(CURVE, gaps[n + len(singles):], flips[n + len(singles):])])
            f.write(json.dumps(rec) + "\n"); f.flush()
            counts[src] += 1
            have.add((c, p))
    print("done", counts)


if __name__ == "__main__":
    main()
