"""Study 29, see PLAN_MOMENT.md. Teacher-forces 192 unedited traces, reads block 19's output along d at answer and intermediate events, and judges bars M1-M3."""
import json, random, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover

sys.path[:0] = ["investigate", "whyfix"]
from behaviors import DOUBT
from gsm_score import ci

FILES = ["runs/whyfix/gsm64_none_traces.jsonl", "runs/whyfix/gsm64b_none_traces.jsonl", "runs/research/layers_none_traces.jsonl"]
NUM = re.compile(r"=\s*\$?(-?\d[\d,]*(?:\.\d+)?)\b")
CONCLUDE = re.compile(r"\\boxed\{\s*[^\s}]|</think>|final answer|the answer is", re.I)
BASE, MIN_BEFORE, AHEAD = 200, 50, 150

mid = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to("mps").eval()
D = torch.load("runs/whyfix/directions.pt")
g = torch.Generator().manual_seed(0)
rnd = torch.randn(D["dir_self_doubt"].shape[0], generator=g)
dirs = {k: (v.float() / v.float().norm()) for k, v in dict(doubt=D["dir_self_doubt"], rep=D["dir_neurospeak"], random=rnd).items()}
U = torch.stack(list(dirs.values()), 1).to("mps")
cap = {}
discover(m).blocks[19].register_forward_hook(lambda _m, _i, o: cap.__setitem__("p", ((o[0] if isinstance(o, tuple) else o)[0].float() @ U).cpu()))


def num(s):
    return float(s.replace(",", ""))


def score(p, t):
    return (p[t] - p[t - BASE:t].mean(0)).tolist() if t >= BASE else (p[t] - p[max(0, t - BASE):t].mean(0)).tolist()


rows = []
with torch.no_grad():
    for f in FILES:
        for r in map(json.loads, open(f)):
            text = r["prompt"] + r["completion"]
            enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
            offs = enc.offset_mapping[0].tolist()
            m(enc.input_ids.to("mps"))
            p = cap["p"]
            n0 = len(r["prompt"])
            ends = [e for _, e in offs]
            to_tok = lambda c: next(i for i, e in enumerate(ends) if e > c)
            ans, comp = num(r["answer"]), r["completion"]
            ev = []
            for x in NUM.finditer(comp):
                try:
                    v = num(x.group(1))
                except ValueError:
                    continue
                t = to_tok(n0 + x.end() - 1)
                if t < MIN_BEFORE:
                    continue
                after = tok.decode(enc.input_ids[0, t + 1:t + 1 + AHEAD])
                first3 = tok.decode(enc.input_ids[0, t + 1:t + 4])
                c, dbt = CONCLUDE.search(after), DOUBT.search(after)
                ev.append(dict(is_answer=abs(v - ans) < 1e-6, s=score(p, t), concl=bool(c and (not dbt or dbt.start() > c.start())), doubt_next=bool(DOUBT.search(first3)), t=t))
            a = next((e for e in ev if e["is_answer"]), None)
            mids = [e for e in ev if not e["is_answer"] and (a is None or e["t"] != a["t"])]
            rows.append(dict(id=r["id"], answer=a, mids=mids))
            print(r["id"], len(ev), flush=True)


def auc(pos, neg, rng=None):
    pairs = [(x > y) + 0.5 * (x == y) for x in pos for y in neg]
    return sum(pairs) / len(pairs)


K = list(dirs)
out = dict(traces=len(rows), with_answer=sum(r["answer"] is not None for r in rows), per_dir={})
for k, name in enumerate(K):
    both = [r for r in rows if r["answer"] and r["mids"]]
    a_s = [r["answer"]["s"][k] for r in both]
    m_s = [sum(e["s"][k] for e in r["mids"]) / len(r["mids"]) for r in both]
    M1 = ci(a_s, m_s)
    clean = [r for r in both if not r["answer"]["doubt_next"]]
    mids_clean = lambda r: [e for e in r["mids"] if not e["doubt_next"]] or r["mids"]
    M3 = ci([r["answer"]["s"][k] for r in clean], [sum(e["s"][k] for e in mids_clean(r)) / len(mids_clean(r)) for r in clean])
    ans = [r["answer"] for r in rows if r["answer"]]
    pos, neg = [-e["s"][k] for e in ans if e["concl"]], [-e["s"][k] for e in ans if not e["concl"]]
    rng = random.Random(0)
    boots = sorted(auc(rng.choices(pos, k=len(pos)), rng.choices(neg, k=len(neg))) for _ in range(1000)) if pos and neg else []
    M2 = dict(auc=auc(pos, neg) if pos and neg else None, low=boots[24] if boots else None, high=boots[974] if boots else None, concluded=len(pos), not_concluded=len(neg))
    out["per_dir"][name] = dict(M1=M1, M1_traces=len(both), M2=M2, M3=M3, M3_traces=len(clean),
                                answer_doubt_next=sum(r["answer"]["doubt_next"] for r in both))
d = out["per_dir"]["doubt"]
out["pass"] = dict(M1=d["M1"]["low"] > 0, M2=None if d["M2"]["concluded"] < 15 else (d["M2"]["auc"] >= 0.6 and d["M2"]["low"] > 0.5))
json.dump(dict(out, rows=rows), open("runs/research/moment.json", "w"))
print(json.dumps(out, indent=1))
