"""Run the model over held-out text and store every SAE activation plus a sample of neurons. Also measures held-out FVU
and how much of the model's prediction loss the SAE reconstruction preserves.

    .venv/bin/python wordtest/scan.py --model Qwen/Qwen3.5-0.8B --layers 15 19
"""
import argparse, glob, json, os, random
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurospeak.anatomy import discover
from neurospeak.core import DEV
from neurospeak.sae import TopKSAE, checkpoint_dir

SEQ = 512
SKIP = 2
BATCH = 8


def split_chunks(ids):
    return [ids[i:i + SEQ] for i in range(0, len(ids) - SEQ + 1, SEQ)]


def corpus(tok, a):
    """Chunks of exactly SEQ tokens as (tokens, source, group). A group never crosses the labeling/test split."""
    out = []
    n = 0
    for split in ("validation", "test"):
        p = hf_hub_download("Salesforce/wikitext", f"wikitext-103-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset")
        doc, docs = [], []
        for line in pq.read_table(p, columns=["text"]).column("text").to_pylist():
            if line.startswith(" = ") and not line.startswith(" = =") and doc:
                docs.append("".join(doc)); doc = []
            doc.append(line)
        docs.append("".join(doc))
        for i, d in enumerate(docs):
            if n >= a.wiki: break
            for c in split_chunks(tok(d).input_ids):
                out.append((c, "wiki", f"wiki-{split}-{i}")); n += SEQ

    p = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    ids = []
    for r in pq.read_table(p).to_pylist():
        ids += tok(r["question"] + "\n" + r["answer"]).input_ids + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    for k, c in enumerate(split_chunks(ids)[:a.math // SEQ]):
        out.append((c, "gsm8k", f"gsm8k-{k}"))

    rows = [json.loads(l) for f in sorted(glob.glob(a.traces_glob)) for l in open(f)]
    random.Random(0).shuffle(rows)
    n = 0
    for r in rows:
        if n >= a.traces: break
        for c in split_chunks(tok(r["prompt"] + r["completion"]).input_ids):
            out.append((c, "trace", "trace-" + r["prompt"][:200])); n += SEQ
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layers", type=int, nargs="+", default=[15, 19])
    ap.add_argument("--wiki", type=int, default=300_000)
    ap.add_argument("--math", type=int, default=60_000)
    ap.add_argument("--traces", type=int, default=60_000)
    ap.add_argument("--traces-glob", default="runs/*traces*.jsonl")
    ap.add_argument("--neurons", type=int, default=30)
    ap.add_argument("--out", default="runs/wordtest")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype="auto").to(DEV).eval()
    m.requires_grad_(False)
    an = discover(m)

    saes, scale, nidx = {}, {}, {}
    rng = random.Random(0)
    for L in a.layers:
        c = torch.load(f"{checkpoint_dir(a.model)}/L{L}.pt", map_location=DEV)
        assert c["config"]["model"] == a.model
        saes[L] = TopKSAE(c["config"]["d"], c["config"]["n"], c["config"]["k"]).to(DEV)
        saes[L].load_state_dict(c["state"]); saes[L].requires_grad_(False)
        scale[L] = c["scale"]
        nidx[L] = rng.sample(range(an.n_neurons(L)), a.neurons)

    chunks = corpus(tok, a)
    groups = sorted({g for _, _, g in chunks})
    random.Random(1).shuffle(groups)
    half_of = {g: i % 2 for i, g in enumerate(groups)}
    print(f"{len(chunks)} chunks, {len(chunks) * SEQ:,} tokens, {len(groups)} groups", flush=True)

    res, nin = {}, {}
    def block_hook(L):
        def f(_m, _i, out):
            res[L] = out[0] if isinstance(out, tuple) else out
        return f
    def neuron_hook(L):
        def f(_m, inp):
            nin[L] = inp[0]
        return f
    hooks = []
    for L in a.layers:
        hooks.append(an.blocks[L].register_forward_hook(block_hook(L)))
        hooks.append(an.mlp_module(L).register_forward_pre_hook(neuron_hook(L)))

    store = {L: {"vals": [], "idx": [], "neuron_acts": []} for L in a.layers}
    err = {L: {} for L in a.layers}
    for s in range(0, len(chunks), BATCH):
        part = chunks[s:s + BATCH]
        ids = torch.tensor([c for c, _, _ in part], device=DEV)
        with torch.no_grad():
            m(ids, use_cache=False)
            for L in a.layers:
                x = res[L].reshape(-1, an.hidden_size).float() * scale[L]
                recon, vals, idx, _ = saes[L](x)
                store[L]["vals"].append(vals.half().cpu())
                store[L]["idx"].append(idx.short().cpu())
                store[L]["neuron_acts"].append(nin[L][:, :, torch.tensor(nidx[L], device=DEV)].reshape(-1, a.neurons).float().cpu())
                e = (x - recon).pow(2).sum(-1).reshape(len(part), SEQ)[:, SKIP:]
                y = x.reshape(len(part), SEQ, -1)[:, SKIP:]
                v = (y - y.reshape(-1, an.hidden_size).mean(0)).pow(2).sum(-1)
                for row, (_, src, _) in enumerate(part):
                    d = err[L].setdefault(src, [0.0, 0.0])
                    d[0] += e[row].sum().item(); d[1] += v[row].sum().item()
        if s % (BATCH * 25) == 0:
            print(f"scanned {s}/{len(chunks)}", flush=True)
    for h in hooks: h.remove()

    fvu = {L: {src: e / max(v, 1e-9) for src, (e, v) in err[L].items()} for L in a.layers}
    print("held-out FVU", fvu, flush=True)

    torch.save({"model": a.model, "layers": a.layers, "seq": SEQ, "skip": SKIP,
                "tokens": torch.tensor([c for c, _, _ in chunks], dtype=torch.int32),
                "source": [s for _, s, _ in chunks], "half": torch.tensor([half_of[g] for _, _, g in chunks]),
                "neurons": nidx, "n_features": {L: saes[L].n for L in a.layers}, "fvu": fvu, "scale": scale,
                "units": {L: {k: torch.cat(v) for k, v in store[L].items()} for L in a.layers}}, f"{a.out}/scan.pt")
    del store
    torch.mps.empty_cache() if DEV == "mps" else None

    # How much of the model's prediction survives replacing the residual with the reconstruction
    held = torch.tensor([c for c, src, _ in chunks if src != "trace"][:40], device=DEV)
    def ce(replace=None):
        total = 0.0
        hook = None
        if replace:
            L, fn = replace
            def h(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                new = fn(t)
                return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
            hook = an.blocks[L].register_forward_hook(h)
        with torch.no_grad():
            for s in range(0, len(held), 2):     # full-vocabulary logits are large, so keep batches small
                b = held[s:s + 2]
                lg = m(b, use_cache=False).logits[:, SKIP:-1]
                total += torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), b[:, SKIP + 1:].reshape(-1), reduction="sum").item()
                del lg
        if hook: hook.remove()
        torch.mps.empty_cache() if DEV == "mps" else None
        return total / (held.shape[0] * (SEQ - SKIP - 1))
    base = ce()
    loss = {"original": base}
    for L in a.layers:
        def rec(t, L=L):
            r = saes[L](t.reshape(-1, an.hidden_size).float() * scale[L])[0] / scale[L]
            return r.reshape(t.shape).to(t.dtype)
        def mean(t, L=L):
            return (saes[L].b_dec / scale[L]).expand_as(t).to(t.dtype)
        cr, cm = ce((L, rec)), ce((L, mean))
        loss[L] = {"reconstructed": cr, "mean_only": cm, "loss_recovered": (cm - cr) / max(cm - base, 1e-9)}
    print("prediction loss", loss, flush=True)

    json.dump({"fvu": {str(k): v for k, v in fvu.items()}, "loss": {str(k): v for k, v in loss.items()}, "chunks": len(chunks), "groups": len(groups)},
              open(f"{a.out}/quality.json", "w"), indent=1)
    print("wrote", f"{a.out}/scan.pt", flush=True)


if __name__ == "__main__":
    main()
