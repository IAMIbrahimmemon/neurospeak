"""Builds directions from the independent pairs, after checking the same code rebuilds the original direction. See PLAN_PAIRS.md."""
import json, sys
import torch
import neurospeak as ns
from neurospeak import core
sys.path.insert(0, "investigate")
from behaviors import PAIRS

brain = ns.load("Qwen/Qwen3.5-0.8B")
L = 19


def build(name, pairs):
    same = [core.Pair(p.is_, p.not_, None, brain.chat_prompt(p.ask, True)) for p in pairs]
    return brain.concept(name, same, at=[L]).dirs[L].cpu().float()


path = "runs/whyfix/directions.pt"
D = torch.load(path)
orig = D["dir_self_doubt"].float()
again = build("check", PAIRS["self_doubt"])
c = (again @ orig).item()
print("rebuild cosine", round(c, 5))
if c < 0.999:
    raise SystemExit("the build doesn't reproduce the original direction")
new = {}
for w in (2, 3):
    rows = json.load(open(f"whyfix/pairs/writer_{w}.json"))
    new[f"dir_writer_{w}"] = build(f"writer_{w}", [ns.pair(ask=r["ask"], is_=r["is_"], not_=r["not_"]) for r in rows])
torch.save({**D, **new}, path)
out = {k: dict(original=(v @ orig).item(), repetition=(v @ D["dir_neurospeak"].float()).item(), mean=(v @ D["dir_mean"].float()).item()) for k, v in new.items()}
out["writer_2_vs_3"] = (new["dir_writer_2"] @ new["dir_writer_3"]).item()
out["matched_vs_original"] = (D["dir_self_doubt_matched"].float() @ orig).item()
json.dump(out, open("runs/research/pair_cosines.json", "w"), indent=1)
print(json.dumps(out, indent=1))
