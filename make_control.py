"""Control: same 64 neurons per layer, same Frobenius change, but a random direction orthogonal to `repetition`."""
import torch
from neurospeak import Interpreter
it = Interpreter()
it.run_source(open("examples/explore.ns").read().split("writers of")[0])   # model + concept only
b, c = it.brain, it.brain.get("repetition")
ours = torch.load("runs/neurospeak_repetition_edit.pt")
g = torch.Generator().manual_seed(0)
out = {}
for L in (15, 19):
    name = f"model.layers.{L}.mlp.down_proj.weight"
    W0 = b.params[name].detach().float().cpu()
    target = (ours[name].float() - W0).norm()
    u = c.dirs[L]
    r = torch.randn(u.shape, generator=g); r -= (r @ u) * u; r /= r.norm()
    idx = c.writers(f"L{L}.mlp", 64).indices
    mask = torch.zeros(W0.shape[1]); mask[idx] = 1
    D = -torch.outer(r, mask * (r @ W0))
    D *= target / D.norm()
    out[name] = (W0 + D).to(torch.bfloat16)
    print(L, "change norm", D.norm().item(), "target", target.item())
torch.save(out, "runs/control_random_edit.pt")
