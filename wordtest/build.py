"""Sample units, write what labelers will see and the test snippets, and run the causal test. Run before any labeling.

    .venv/bin/python wordtest/build.py --features 100
"""
import argparse, json, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurospeak.anatomy import discover
from neurospeak.core import DEV
from neurospeak.sae import TopKSAE, checkpoint_dir

WINDOW = 64
LABEL, TEST = 0, 1
MIN_LABEL, MIN_TEST = 12, 8
GAP = 10
BATCH_SIZE = 52


def pick(cands, n, taken=None, text=None):
    """Up to n positions from cands in order, at least GAP tokens apart, with no repeated snippet."""
    out, seen = [], set()
    taken = list(taken or [])
    for p in cands:
        if any(abs(p - q) < GAP for q in taken + out): continue
        t = text(p)
        if t in seen: continue
        seen.add(t); out.append(p)
        if len(out) == n: break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/wordtest")
    ap.add_argument("--features", type=int, default=100, help="per layer")
    a = ap.parse_args()
    S = torch.load(f"{a.dir}/scan.pt", map_location="cpu")
    model, layers, SEQ, SKIP = S["model"], S["layers"], S["seq"], S["skip"]
    tok = AutoTokenizer.from_pretrained(model)
    T = S["tokens"].view(-1)
    N = len(T)
    half = S["half"].repeat_interleave(SEQ)
    source = [s for s in S["source"] for _ in range(SEQ)]
    usable = (torch.arange(N) % SEQ) >= SKIP
    rng = random.Random(7)

    def snippet(p):
        s = max(p - p % SEQ, p - 16)
        e = min(p - p % SEQ + SEQ, p + 4)
        f = lambda x, y: tok.decode(T[x:y].tolist()).replace("\n", "⏎")
        return f"{f(s, p)}[[{f(p, p + 1)}]]{f(p + 1, e)}"

    units, stats = [], {}
    for L in layers:
        U = S["units"][L]
        vals, idx = U["vals"], U["idx"].long()
        active = (vals > 0) & usable.unsqueeze(1)
        n_feat = S["n_features"][L]
        pos_half = half.unsqueeze(1).expand_as(idx)
        c_label = torch.bincount(idx[active & (pos_half == LABEL)], minlength=n_feat)
        c_test = torch.bincount(idx[active & (pos_half == TEST)], minlength=n_feat)
        total = torch.bincount(idx[active], minlength=n_feat)
        eligible = ((c_label >= MIN_LABEL) & (c_test >= MIN_TEST)).nonzero().squeeze(1).tolist()
        chosen = rng.sample(eligible, a.features)
        stats[L] = {"features": n_feat, "ever_active": int((total > 0).sum()), "eligible": len(eligible),
                    "median_active_share_of_eligible": float(total[eligible].float().median() / usable.sum())}
        print(f"L{L}", stats[L], flush=True)
        for f in chosen:
            v = torch.zeros(N)
            m = idx == f
            rows = m.any(1)
            v[rows] = (vals[rows].float() * m[rows].float()).sum(1)
            v[~usable] = 0
            units.append(dict(kind="feature", layer=L, index=f, val=v, hi=0.0, lo=0.0))
        for c, n in enumerate(S["neurons"][L]):
            v = U["neuron_acts"][:, c].clone()
            good = v[usable]
            units.append(dict(kind="neuron", layer=L, index=n, val=v, hi=float(good.quantile(0.99)), lo=float(good.median())))

    records = []
    for u in units:
        v = u["val"]
        act = (v > u["hi"]) & usable
        ina = (v <= u["lo"]) & usable
        def by_half(mask, h):
            return (mask & (half == h)).nonzero().squeeze(1)

        def choose(h, top_n, weak_n):
            idxs = by_half(act, h)
            order = idxs[v[idxs].argsort(descending=True)].tolist()
            top = pick(order, top_n, text=snippet)
            rest = [p for p in order if v[p] < v[idxs].median()] or order
            rng.shuffle(rest)
            weak = pick(rest, weak_n, taken=top, text=snippet)
            return top, weak

        def negatives_same_token(h, positives, n, taken):
            out = []
            for p in positives:
                same = by_half(ina & (T == T[p]), h).tolist()
                rng.shuffle(same)
                got = pick(same[:200], 1, taken=taken + out, text=snippet)
                out += got
                if len(out) == n: break
            return out

        l_top, l_weak = choose(LABEL, 10, 6)
        l_neg = negatives_same_token(LABEL, l_top, 4, l_top + l_weak)
        t_top, t_weak = choose(TEST, 2, 2)
        pos = t_top + t_weak
        same = negatives_same_token(TEST, pos, 2, pos)
        rand = []
        for p in pos:
            pool = by_half(ina, TEST)
            cands = pool[torch.randint(len(pool), (2000,))].tolist()
            cands = [q for q in cands if source[q] == source[p] and T[q] != T[p]]
            rand += pick(cands, 1, taken=pos + same + rand, text=snippet)
            if len(rand) == 4 - len(same): break
        records.append({
            "kind": u["kind"], "layer": u["layer"], "index": u["index"], "hi": u["hi"],
            "active_share": float(act.sum() / usable.sum()),
            "label_view": {"strongest": [{"act": round(float(v[p]), 2), "text": snippet(p)} for p in l_top],
                           "weaker": [{"act": round(float(v[p]), 2), "text": snippet(p)} for p in l_weak],
                           "does_not_fire": [snippet(p) for p in l_neg]},
            "test": {"positive_top": [snippet(p) for p in t_top], "positive_weak": [snippet(p) for p in t_weak],
                     "same_token": [snippet(p) for p in same], "random": [snippet(p) for p in rand]},
            "causal_positions": by_half(act, TEST)[v[by_half(act, TEST)].argsort(descending=True)].tolist()[:60],
        })

    # causal test
    m = AutoModelForCausalLM.from_pretrained(model, dtype="auto").to(DEV).eval()
    m.requires_grad_(False)
    an = discover(m)
    saes = {}
    for L in layers:
        c = torch.load(f"{checkpoint_dir(model)}/L{L}.pt", map_location=DEV)
        saes[L] = TopKSAE(c["config"]["d"], c["config"]["n"], c["config"]["k"]).to(DEV)
        saes[L].load_state_dict(c["state"]); saes[L].requires_grad_(False)
    cols = {L: an.columns(an.mlp_out[L] + ".weight", an.mlp_module(L).weight).detach().float() for L in layers}
    feature_pool = {L: [r["index"] for r in records if r["layer"] == L and r["kind"] == "feature"] for L in layers}
    neuron_pool = {L: [r["index"] for r in records if r["layer"] == L and r["kind"] == "neuron"] for L in layers}
    gen = torch.Generator(device="cpu").manual_seed(3)

    def run(r, positions):
        """Four rows per context: unchanged, unit removed, random direction removed, another word's direction removed."""
        L, ix = r["layer"], r["index"]
        ids = torch.stack([T[p - WINDOW + 1:p + 1] for p in positions]).to(DEV).long().repeat_interleave(4, 0)
        grabbed = {}
        pre = an.mlp_module(L).register_forward_pre_hook(lambda _m, inp: grabbed.__setitem__("x", inp[0]))
        used = []
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            last = h[0::4, -1].float()
            if r["kind"] == "feature":
                sae, s = saes[L], S["scale"][L]
                vals, idc = sae.encode(last * s)
                amount = (vals * (idc == ix).float()).sum(1) / s
                u = sae.W_dec[:, ix]
                others = [sae.W_dec[:, rng.choice([q for q in feature_pool[L] if q != ix])] for _ in positions]
            else:
                a_ = grabbed["x"][0::4, -1, ix].float()
                col = cols[L][:, ix]
                amount = a_.relu() * col.norm()
                u = col / col.norm()
                others = []
                for _ in positions:
                    c2 = cols[L][:, rng.choice([q for q in neuron_pool[L] if q != ix])]
                    others.append(c2 / c2.norm())
            used.append(amount.detach().cpu())
            rnd = torch.randn(len(positions), an.hidden_size, generator=gen).to(DEV)
            rnd = rnd / rnd.norm(dim=1, keepdim=True)
            new = h.clone()
            new[1::4, -1] -= (amount.unsqueeze(1) * u.unsqueeze(0)).to(h.dtype)
            new[2::4, -1] -= (amount.unsqueeze(1) * rnd).to(h.dtype)
            new[3::4, -1] -= (amount.unsqueeze(1) * torch.stack(others)).to(h.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        hk = an.blocks[L].register_forward_hook(hook)
        with torch.no_grad():
            lp = m(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
        hk.remove(); pre.remove()
        ok = used[0].to(DEV) > 0
        base, rem, rnd, oth = lp[0::4][ok], lp[1::4][ok], lp[2::4][ok], lp[3::4][ok]
        kl = lambda q: (base.exp() * (base - q)).sum(-1).cpu().tolist()
        return kl(rem), kl(rnd), kl(oth), (base.exp() - rem.exp()).cpu(), used[0][ok.cpu()].tolist()

    for k, r in enumerate(records):
        cands = [p for p in r.pop("causal_positions") if p % SEQ >= WINDOW - 1]
        chosen = pick(cands, 18, text=snippet)
        K, Kr, Ko, dp, amt = [], [], [], [], []
        for s in range(0, len(chosen), 6):
            if len(K) >= 6: break
            out = run(r, chosen[s:s + 6])
            K += out[0]; Kr += out[1]; Ko += out[2]; amt += out[4]
            if len(out[3]): dp.append(out[3])
        if not K:
            r["causal"] = None
            continue
        mean_dp = torch.cat(dp).mean(0)
        up, down = mean_dp.topk(8).indices.tolist(), (-mean_dp).topk(8).indices.tolist()
        r["causal"] = {"contexts": len(K), "kl_remove": sum(K) / len(K), "kl_random": sum(Kr) / len(K), "kl_other_word": sum(Ko) / len(K),
                       "amount": sum(amt) / len(amt),
                       "promotes": [tok.decode([t]) for t in up], "suppresses": [tok.decode([t]) for t in down]}
        if k % 40 == 0:
            print(f"causal {k}/{len(records)}", flush=True)

    rng.shuffle(records)
    for i, r in enumerate(records):
        r["id"] = f"U{i:03d}"
    json.dump({"stats": {str(k): v for k, v in stats.items()}, "records": records}, open(f"{a.dir}/units.json", "w"), ensure_ascii=False, indent=1)
    for b in range(0, len(records), BATCH_SIZE):
        json.dump([{"id": r["id"], **r["label_view"]} for r in records[b:b + BATCH_SIZE]],
                  open(f"{a.dir}/label_batch_{b // BATCH_SIZE}.json", "w"), ensure_ascii=False, indent=1)
    print(len(records), "units,", len(records) // BATCH_SIZE + (len(records) % BATCH_SIZE > 0), "label batches", flush=True)


if __name__ == "__main__":
    main()
