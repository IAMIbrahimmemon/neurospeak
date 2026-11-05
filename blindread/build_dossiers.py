"""Step 1 — weights-only dossiers. Never runs the model on text. Output: blindread/dossiers.json"""
import json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
E = m.model.embed_tokens.weight.detach()            # [V, d]; tied with lm_head
V = E.shape[0]
# skip unused / special rows: keep tokens that decode to something printable
keep = torch.tensor([i for i in range(V) if (s := tok.decode([i])).strip() and "<|" not in s and "�" not in s])
Ek = E[keep]

def show(i):
    return repr(tok.decode([int(i)]))[1:-1].replace(" ", "·")

def top(scores, k=20, largest=True):
    idx = scores.topk(k, largest=largest).indices
    return [show(keep[j]) for j in idx]

BANDS = {"early": [1, 2, 3], "middle": [11, 13, 15], "late": [19, 20, 22]}
rng = random.Random(1234)
picks = []
for band, layers in BANDS.items():
    for _ in range(20):
        picks.append((band, rng.choice(layers), rng.randrange(m.config.intermediate_size)))

norm_f = m.model.norm
out = []
with torch.no_grad():
    for k, (band, L, n) in enumerate(picks):
        layer = m.model.layers[L]
        ln_w = layer.post_attention_layernorm.weight          # MLP input norm
        g = layer.mlp.gate_proj.weight[n]                       # [d]
        u = layer.mlp.up_proj.weight[n]
        col = layer.mlp.down_proj.weight[:, n]                  # [d]
        # Input side ("which raw token embeddings excite it" — ignores all earlier processing)
        xn = layer.post_attention_layernorm(Ek)
        gs, us = xn @ g, xn @ u
        direct = torch.nn.functional.silu(gs) * us
        # Output side (logit lens of the output column)
        logits = m.lm_head(norm_f(col[None]))[0][keep]
        logits = logits - logits.mean()
        out.append({
            "id": f"N{k:02d}", "band": band, "layer": L, "neuron": n,
            "dossier": {
                "layer": L, "of_layers": 24,
                "output_promotes": top(logits), "output_suppresses": top(logits, largest=False),
                "input_gate_high": top(gs), "input_gate_low": top(gs, largest=False),
                "input_up_high": top(us), "input_up_low": top(us, largest=False),
                "direct_token_response_positive": top(direct), "direct_token_response_negative": top(direct, largest=False),
                "norms": {"gate": round(g.norm().item(), 3), "up": round(u.norm().item(), 3), "out": round(col.norm().item(), 3)},
                "cos_gate_up": round(torch.nn.functional.cosine_similarity(g, u, 0).item(), 3),
            }})
json.dump(out, open("blindread/dossiers.json", "w"), ensure_ascii=False, indent=1)
print(len(out), "dossiers;", sum(1 for o in out if o["band"] == "early"), "early")
print(json.dumps(out[0]["dossier"], ensure_ascii=False)[:1500])
