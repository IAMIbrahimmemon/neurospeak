"""Measures bars D1-D5 in PLAN_ACTIVITY.md on the CPU, so it doesn't disturb generation on the GPU."""
import json, random, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover
sys.path.insert(0, "investigate")
from behaviors import DOUBT

torch.set_num_threads(4)
mid, L, MAX = "Qwen/Qwen3.5-0.8B", 19, 1500
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
D = torch.load("runs/whyfix/directions.pt")
names = {"original": "dir_self_doubt", "writer_2": "dir_writer_2", "writer_3": "dir_writer_3", "repetition": "dir_neurospeak", "matched": "dir_self_doubt_matched"}
U = torch.stack([D[v].float() for v in names.values()])
grab = {}
discover(m).blocks[L].register_forward_hook(lambda _m, _i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    all_ = torch.cat([pos, neg])
    ranks = torch.zeros(len(all_))
    ranks[all_.argsort()] = torch.arange(1, len(all_) + 1).float()
    return ((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item()


rows, traces = [json.loads(l) for l in open("runs/whyfix/gsm64b_none_traces.jsonl")], []
for n, r in enumerate(rows):
    p = tok(r["prompt"], add_special_tokens=False).input_ids
    enc = tok(r["completion"], add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc.input_ids[:MAX], enc.offset_mapping[:MAX]
    text = r["completion"]
    starts = {x.start() for x in DOUBT.finditer(text)}
    pos = [i for i, (a, b) in enumerate(off) if any(a <= s < b for s in starts)]
    after = []
    for i, (a, b) in enumerate(off):
        word = a + len(text[a:b]) - len(text[a:b].lstrip())
        if i == 0 or word >= b or not text[word].isalpha():
            continue
        line = text[text.rfind("\n", 0, word) + 1:word]
        if re.fullmatch(r"\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*)?", line) or re.search(r"[.?!:]\s+$", line):
            after.append(i)
    with torch.no_grad():
        m(torch.tensor([p + ids]), use_cache=False)
    proj = (grab["h"][0, len(p):] @ U.T).cpu()
    row = {}
    for k, name in enumerate(names):
        row[name] = {}
        for lag in (1, 5):
            P = [i for i in pos if i >= lag]
            N = [i for i in range(lag, len(ids)) if i not in set(pos)]
            row[name][f"lag{lag}"] = auroc(proj[[i - lag for i in P], k], proj[[i - lag for i in N], k]) if len(P) >= 3 else None
        P, N = [i for i in pos if i in set(after)], [i for i in after if i not in set(pos)]
        row[name]["after_break"] = auroc(proj[[i - 1 for i in P], k], proj[[i - 1 for i in N], k]) if len(P) >= 3 else None
    traces.append(dict(id=r["id"], n_pos=len(pos), n_after=len(after), **row))
    print(n, r["id"], len(ids), len(pos), {k: round(row["original"][k], 3) if row["original"][k] is not None else None for k in row["original"]}, flush=True)
json.dump(traces, open("runs/research/activity_traces.json", "w"), indent=1)
