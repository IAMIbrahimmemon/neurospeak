"""Part A of PLAN_BAN.md (study 28): logit lens of the self-doubt direction, and the real effect of removing it on the banned tokens. Runs on CPU."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover

BAN = [13784, 13428, 11158, 3655, 77264, 85152, 84485, 50821, 32645, 70546, 3404]
mid = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
d = torch.load("runs/whyfix/directions.pt")["dir_self_doubt"].float()
d = d / d.norm()
norm, head = m.model.norm, m.get_output_embeddings()
with torch.no_grad():
    lens = head(norm(d[None]))[0]
rank = torch.argsort(lens, descending=True)
pos = torch.empty_like(rank); pos[rank] = torch.arange(len(rank))
top = [tok.decode([int(i)]) for i in rank[:30]]
bottom = [tok.decode([int(i)]) for i in rank[-15:]]
ban_rank = {tok.decode([i]): int(pos[i]) for i in BAN}

R = [json.loads(l) for l in open("runs/research/layers_none_traces.jsonl")][:4]
blk = discover(m).blocks[19]
state = dict(on=False)


def hook(_m, _i, o):
    if not state["on"]:
        return o
    h = o[0] if isinstance(o, tuple) else o
    new = h - (h @ d).unsqueeze(-1) * d
    return (new, *o[1:]) if isinstance(o, tuple) else new


blk.register_forward_hook(hook)
deltas, base_p, cut_p = [], [], []
with torch.no_grad():
    for r in R:
        ids = tok(r["prompt"] + r["completion"], return_tensors="pt").input_ids[:, :2500]
        n0 = len(tok(r["prompt"]).input_ids)
        lp = []
        for on in (False, True):
            state["on"] = on
            lp.append(torch.log_softmax(m(ids).logits[0, n0:].float(), -1)[:, BAN].logsumexp(-1))
        deltas.append((lp[1] - lp[0]).mean().item())
        base_p.append(lp[0].exp().mean().item())
        cut_p.append(lp[1].exp().mean().item())
out = dict(top30=top, bottom15=bottom, banned_rank_of_vocab=ban_rank, vocab=len(rank),
           traces=len(R), mean_logprob_change_banned=deltas, mean_p_banned_unedited=base_p, mean_p_banned_removed=cut_p)
json.dump(out, open("runs/research/ban_lens.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
