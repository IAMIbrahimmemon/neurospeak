"""Build naming batches for chosen words from the stored scan, labeling half only. Labelers see the same evidence as the
word test and nothing about why a word was chosen. Merge their answers with --merge.

    .venv/bin/python wordtest/name_words.py --layer 19 --words 976 14177 --out runs/wordtest/naming --per-batch 16
    .venv/bin/python wordtest/name_words.py --layer 19 --out runs/wordtest/naming --merge
"""
import argparse, glob, json, os, random

import torch
from transformers import AutoTokenizer

SEQ, SKIP, LABEL = 512, 2, 0


def build(a):
    S = torch.load("runs/wordtest/scan.pt", map_location="cpu")
    tok = AutoTokenizer.from_pretrained(S["model"])
    T = S["tokens"].view(-1)
    half = S["half"].repeat_interleave(SEQ)
    usable = (torch.arange(len(T)) % SEQ >= SKIP) & (half == LABEL)
    vals, idx = S["units"][a.layer]["vals"], S["units"][a.layer]["idx"].long()
    rng = random.Random(11)

    def snippet(p):
        s, e = max(p - p % SEQ, p - 16), min(p - p % SEQ + SEQ, p + 4)
        f = lambda x, y: tok.decode(T[x:y].tolist()).replace("\n", "⏎")
        return f"{f(s, p)}[[{f(p, p + 1)}]]{f(p + 1, e)}"

    units, thin = [], []
    for n, w in enumerate(a.words):
        hit = (idx == w) & (vals > 0)
        pos = hit.any(1).nonzero().squeeze(1)
        pos = pos[usable[pos]]
        if len(pos) < 8:
            thin.append(w); continue
        v = (vals * hit).sum(1)[pos]
        order = v.argsort(descending=True)
        top = pos[order[:10]].tolist()
        rest = pos[order[len(order) // 2:]].tolist()
        weak = rng.sample(rest, min(6, len(rest)))
        on = set(pos.tolist())
        same = {T[p].item() for p in top}
        quiet = [p for p in usable.nonzero().squeeze(1).tolist() if p not in on and T[p].item() in same]
        quiet = rng.sample(quiet, min(6, len(quiet)))
        units.append(dict(id=f"U{n:03d}", word=w, strongest=[dict(activation=round((vals[p] * hit[p]).sum().item(), 2), text=snippet(p)) for p in top],
                          weaker=[snippet(p) for p in weak], does_not_fire=[snippet(p) for p in quiet]))
    os.makedirs(a.out, exist_ok=True)
    json.dump({u["id"]: u["word"] for u in units}, open(f"{a.out}/key.json", "w"))
    for b in range(0, len(units), a.per_batch):
        batch = [{k: v for k, v in u.items() if k != "word"} for u in units[b:b + a.per_batch]]
        json.dump(batch, open(f"{a.out}/batch_{b // a.per_batch}.json", "w"), indent=1, ensure_ascii=False)
    print(f"{len(units)} words in {(len(units) + a.per_batch - 1) // a.per_batch} batches; too few firings: {thin}")


def merge(a):
    key = json.load(open(f"{a.out}/key.json"))
    path = f"runs/words/Qwen__Qwen3.5-0.8B/L{a.layer}.json"
    names = json.load(open(path))
    added = 0
    for f in sorted(glob.glob(f"{a.out}/labels_*.json")):
        for r in json.load(open(f)):
            w = str(key[r["id"]])
            if w not in names:
                names[w] = dict(fires_on=r["fires_on"], effect=r["effect"], confidence=r["confidence"]); added += 1
    json.dump(names, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"added {added}, now {len(names)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--words", type=int, nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-batch", type=int, default=16)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    (merge if a.merge else build)(a)
