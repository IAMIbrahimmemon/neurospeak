"""Train TopK sparse autoencoders on Qwen3.5-0.8B's residual stream. Activations stream from the
model straight into training and are never written to disk.

    .venv/bin/python sae_train.py --model Qwen/Qwen3.5-0.8B --layers 15 19 --tokens 12000000

Each checkpoint goes to runs/sae/<model>/L{layer}.pt. Rerunning resumes from the checkpoint, skipping the
tokens already used. Adam restarts on resume, which costs a brief wobble in the loss.
"""
import argparse, glob, json, os, time
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurospeak.anatomy import discover
from neurospeak.core import DEV
from neurospeak.sae import TopKSAE, checkpoint_dir

SEQ_LEN = 512
SKIP_START = 2        # the first tokens of a sequence have outlier residual norms
WIKI_SHARE = 0.85     # the rest is math, so reasoning features get learned too


def wiki_docs():
    for shard in range(2):
        p = hf_hub_download("Salesforce/wikitext", f"wikitext-103-raw-v1/train-0000{shard}-of-00002.parquet", repo_type="dataset")
        buf = []
        for line in pq.read_table(p, columns=["text"]).column("text").to_pylist():
            # a " = Title = " line starts a new article
            if line.startswith(" = ") and not line.startswith(" = ="):
                if buf: yield "".join(buf)
                buf = []
            buf.append(line)
        if buf: yield "".join(buf)


def math_docs(traces_glob="runs/*traces*.jsonl", gsm8k=True):
    while True:      # math is the smaller source, so it cycles
        if gsm8k:
            p = hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet", repo_type="dataset")
            for r in pq.read_table(p).to_pylist():
                yield r["question"] + "\n" + r["answer"]
        for f in sorted(glob.glob(traces_glob)):
            for l in open(f):
                r = json.loads(l)
                yield r["prompt"] + r["completion"]


def chunks(tok, seq_len, share=WIKI_SHARE, traces_glob="runs/*traces*.jsonl", gsm8k=True):
    """Endless stream of seq_len token lists, deterministic, so resuming can skip ahead."""
    wiki, math = wiki_docs(), math_docs(traces_glob, gsm8k)
    pending = {"wiki": [], "math": []}
    made = {"wiki": 0, "math": 0}
    while True:
        total = made["wiki"] + made["math"]
        src = "wiki" if made["wiki"] <= share * total else "math"
        while len(pending[src]) < seq_len:
            doc = next(wiki if src == "wiki" else math)
            pending[src] += tok(doc).input_ids
            if tok.eos_token_id is not None:
                pending[src].append(tok.eos_token_id)
        yield pending[src][:seq_len]
        pending[src] = pending[src][seq_len:]
        made[src] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layers", type=int, nargs="+", default=[15, 19])
    ap.add_argument("--tokens", type=int, default=12_000_000)
    ap.add_argument("--features", type=int, default=None, help="default: 16 times the residual width")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-seqs", type=int, default=8)
    ap.add_argument("--buffer-batches", type=int, default=64)
    ap.add_argument("--step-size", type=int, default=4096)
    ap.add_argument("--ckpt-every", type=int, default=500_000)
    ap.add_argument("--out", default=None, help="default: runs/sae/<model>")
    ap.add_argument("--wiki-share", type=float, default=WIKI_SHARE, help="share of chunks from Wikipedia; the rest are math and traces")
    ap.add_argument("--traces-glob", default="runs/*traces*.jsonl")
    ap.add_argument("--no-gsm8k", action="store_true", help="the non-Wikipedia share is traces only")
    a = ap.parse_args()
    a.out = a.out or checkpoint_dir(a.model)
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype="auto").to(DEV).eval()
    for p in m.parameters(): p.requires_grad_(False)

    anatomy = discover(m)
    print(anatomy.summary(), flush=True)
    d = anatomy.hidden_size
    a.features = a.features or 16 * d
    seq_len = min(SEQ_LEN, getattr(m.config.get_text_config(), "max_position_embeddings", SEQ_LEN))
    saes, opts, scale, seen = {}, {}, {}, 0
    for L in a.layers:
        saes[L] = TopKSAE(d, a.features, a.k).to(DEV)
        path = f"{a.out}/L{L}.pt"
        if os.path.exists(path):
            c = torch.load(path, map_location=DEV)
            if c["config"]["model"] != a.model:
                raise SystemExit(f"{path} was trained on {c['config']['model']}, not {a.model}")
            saes[L].load_state_dict(c["state"]); scale[L] = c["scale"]; seen = c["tokens"]
        opts[L] = torch.optim.Adam(saes[L].parameters(), lr=a.lr, betas=(0.9, 0.999))
    last_fired = {L: torch.zeros(a.features, device=DEV) for L in a.layers}  # tokens since last fired

    grabbed = {}
    hooks = [anatomy.blocks[L].register_forward_hook(
        (lambda L: lambda _m, _i, out: grabbed.__setitem__(L, out[0] if isinstance(out, tuple) else out))(L))
        for L in a.layers]

    stream = chunks(tok, seq_len, a.wiki_share, a.traces_glob, not a.no_gsm8k)
    for _ in range(seen // (seq_len - SKIP_START)):
        next(stream)
    print(f"start at {seen:,} tokens, target {a.tokens:,}, layers {a.layers}", flush=True)

    t0, t_start, next_ckpt = time.time(), seen, seen + a.ckpt_every
    while seen < a.tokens:
        # 1. fill a buffer from the model
        buf = {L: [] for L in a.layers}
        for _ in range(a.buffer_batches):
            batch = torch.tensor([next(stream) for _ in range(a.batch_seqs)], device=DEV)
            with torch.no_grad():
                m(batch, use_cache=False)
            for L in a.layers:
                buf[L].append(grabbed[L][:, SKIP_START:].reshape(-1, d).half())
        n_toks = sum(x.shape[0] for x in buf[a.layers[0]])
        perm = torch.randperm(n_toks, device=DEV)

        # 2. train each SAE on the shuffled buffer
        stats = {}
        for L in a.layers:
            x_all = torch.cat(buf[L])[perm]
            sae, opt = saes[L], opts[L]
            if L not in scale:
                # scale so the mean norm is sqrt(d), which makes the loss comparable across layers
                scale[L] = (d ** 0.5 / x_all.float().norm(dim=-1).mean()).item()
                sae.b_dec.data = (x_all.float() * scale[L]).mean(0)
            mse_sum, var_sum = 0.0, 0.0
            for s in range(0, n_toks - a.step_size + 1, a.step_size):
                x = x_all[s:s + a.step_size].float() * scale[L]
                dead = last_fired[L] > 500_000
                recon, vals, idx, aux = sae(x, dead)
                err = x - recon
                loss = err.pow(2).sum(-1).mean()
                if aux is not None:
                    aux_loss = (err.detach() - aux).pow(2).sum(-1).mean()
                    loss = loss + (1 / 32) * aux_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                opt.step()
                sae.norm_decoder()
                last_fired[L] += x.shape[0]
                last_fired[L][idx[vals > 0].unique()] = 0
                mse_sum += err.detach().pow(2).sum().item()
                var_sum += (x - x.mean(0)).pow(2).sum().item()
            stats[L] = (mse_sum / max(var_sum, 1e-9), (last_fired[L] > 500_000).float().mean().item(),
                        (vals > 0).float().sum(-1).mean().item())
        del buf
        seen += n_toks
        rate = (seen - t_start) / (time.time() - t0)
        left = (a.tokens - seen) / max(rate, 1) / 3600
        print(f"{seen:>11,} tok  {rate:6.0f} tok/s  ~{left:4.1f}h left  " + "  ".join(
            f"L{L} fvu {s[0]:.3f} dead {s[1]:.1%} l0 {s[2]:.0f}" for L, s in stats.items()), flush=True)

        if seen >= next_ckpt or seen >= a.tokens:
            next_ckpt += a.ckpt_every
            for L in a.layers:
                tmp = f"{a.out}/L{L}.pt.tmp"
                torch.save({"state": {k: v.cpu() for k, v in saes[L].state_dict().items()},
                            "scale": scale[L], "tokens": seen, "layer": L,
                            "config": {"d": d, "n": a.features, "k": a.k, "model": a.model,
                                       "seq_len": seq_len, "skip_start": SKIP_START}}, tmp)
                os.replace(tmp, f"{a.out}/L{L}.pt")
            print(f"  checkpoint at {seen:,}", flush=True)
    for h in hooks: h.remove()


if __name__ == "__main__":
    main()
