"""Mean-matched controls for the L19 self-doubt direction removed at other layers, for PLAN_TRANSFER.md."""
import json
import torch
import neurospeak as ns
from neurospeak import core
from neurospeak.investigate import matched_control, mean_residual

brain = ns.load("Qwen/Qwen3.5-0.8B")
u = torch.load("runs/whyfix/directions.pt")["dir_self_doubt"].float()
normal = core._default_corpus()[:24]
out, info = {}, {}
for l in (7, 11, 15, 23):
    m = mean_residual(brain, l, normal).cpu().float() if l < 23 else None
    if m is None:  # hidden_states[24] is post-final-norm; take block 23's output from a hook instead
        acts = []
        h = brain.anatomy.blocks[23].register_forward_hook(lambda _m, _i, o: acts.append((o[0] if isinstance(o, tuple) else o)[0].detach().float().cpu()))
        with torch.no_grad():
            for t in normal:
                brain.run(t[:2000], capture=False)
        h.remove()
        m = torch.cat(acts).mean(0)
    out[f"dir_xfer_{l}_matched"] = matched_control(u, m, seed=1)
    info[l] = dict(mean_overlap=(u @ torch.nn.functional.normalize(m, dim=0)).item(), control_cos=(out[f"dir_xfer_{l}_matched"] @ u).item())
torch.save(out, "runs/research/xfer_dirs.pt")
json.dump(info, open("runs/research/xfer_dirs.json", "w"), indent=1)
print(json.dumps(info, indent=1))
