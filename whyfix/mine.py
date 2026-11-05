"""Finds loop onsets in unedited traces and ranks L19 words by attribution for repeating.  .venv/bin/python whyfix/mine.py"""
import json, random, sys
import torch

import neurospeak as ns
from neurospeak.core import DEV
from neurospeak.why import attribute

L, K, CONTEXT, GAP, MAX_ONSETS, OUT = 19, 16, 512, 200, 3, "runs/whyfix"


def onsets(ids, ahead=64, needed=48):
    """Tokens where sustained repetition starts: most of the next `ahead` 8-grams have appeared before."""
    first = {}
    for i in range(len(ids) - 7):
        first.setdefault(tuple(ids[i:i + 8]), i)
    repeated = [first[tuple(ids[i:i + 8])] < i for i in range(len(ids) - 7)]
    found, last = [], -GAP
    for p in range(len(repeated) - ahead):
        if p - last >= GAP and repeated[p] and sum(repeated[p:p + ahead]) >= needed:
            found.append(p)
            last = p
            if len(found) == MAX_ONSETS:
                break
    return found


def main():
    brain = ns.load(dtype=torch.float32)
    rows = [json.loads(l) for l in open("runs/base_traces.jsonl")][16:] + [json.loads(l) for l in open(f"{OUT}/mine_traces.jsonl")]
    F = brain.features[L]
    total, count = torch.zeros(len(F)), torch.zeros(len(F))
    used = traces = 0
    for r in rows:
        ids = brain.tok(r["prompt"] + r["completion"]).input_ids
        n_prompt = len(brain.tok(r["prompt"]).input_ids)
        points = [p for p in onsets(ids) if p > n_prompt]
        if points:
            print(repr(brain.tok.decode(ids[points[0] - 8:points[0]])), ">>", repr(brain.tok.decode(ids[points[0]:points[0] + 16])), flush=True)
        traces += bool(points)
        for p in points:
            ctx = torch.tensor(ids[max(0, p - CONTEXT):p], device=DEV)[None]
            vals, idx, scores = attribute(brain, ctx, L, ids[p])
            pos = scores.clamp(min=0).view(-1).cpu()
            total.index_add_(0, idx.view(-1).cpu(), pos)
            last_active = idx[-1][vals[-1] > 0].cpu().unique()
            count[last_active] += 1
            used += 1
    print("traces", len(rows), "with onsets", traces, "onsets", used)
    why_words = total.topk(K).indices.tolist()
    active_words = count.topk(K).indices.tolist()
    S = torch.load("runs/wordtest/scan.pt", map_location="cpu")
    U = S["units"][L]
    rate = torch.bincount(U["idx"].long().view(-1)[U["vals"].view(-1) > 0], minlength=len(F)).float()
    order = rate.argsort()
    rank = torch.zeros(len(F), dtype=torch.long)
    rank[order] = torch.arange(len(F))
    rng = random.Random(71)
    random_words = []
    for w in why_words:
        band = order[max(0, rank[w] - 50):rank[w] + 50].tolist()
        band = [b for b in band if b not in why_words and b not in random_words]
        random_words.append(rng.choice(band))
    W = json.load(open(f"{OUT}/words.json"))
    W.update(why2=why_words, active2=active_words, random2=random_words, onsets2=used, traces_with_onsets2=traces)
    json.dump(W, open(f"{OUT}/words.json", "w"), indent=1)
    print(why_words, active_words, "overlap", len(set(why_words) & set(active_words)))


if __name__ == "__main__":
    main()
