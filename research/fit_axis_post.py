"""POST HOC, after study 16. Which one-axis angle fits study 16's in-plane arms, and study 7/8/11 pooled? Does a two-coefficient model do better? No bars."""
import json, math, sys
import numpy as np
src = open("research/score_rescue.py").read()
exec(src[src.index("SIZE, STEP"):src.index("R = {a:")])
from neurospeak.investigate import collapsed
ACT = sys.argv[1]
acts = [json.loads(l) for l in open(ACT)]
mean = lambda v: float(np.mean(v))
load = lambda p: [json.loads(l) for l in open(p)]
PHI, TH = math.radians(24.7456), math.radians(42)
c, s_ = math.cos(PHI), math.sin(PHI)
A7 = {r["id"]: r for r in acts if r["set"] == "s7"}
AT = {r["id"]: r for r in acts if r["set"].startswith("tilt")}
G = json.load(open("runs/research/tilt_gains.json"))
ONLY_EQ = G["activity"]["dir_tilt"]["mean_abs"] / (G["activity"]["only"]["mean_abs"] * math.cos(TH))

arms16 = ("none", "only", "only_eq", "tilt", "both", "perp_eq")
T = {a: load(f"runs/research/tilt_{a}_traces.jsonl") for a in arms16}
tid = [r["id"] for r in T["none"]]
tp = {k: np.array([mean(AT[i]["proj"][k]) for i in tid]) for k in ("only", "s", "tilt", "perp")}
z = np.zeros(len(tid))
D16 = dict(none=(z, z), only=(tp["only"], z), only_eq=(ONLY_EQ * tp["only"], z),
           tilt=(tp["tilt"] * math.cos(TH), tp["tilt"] * math.sin(TH)), both=(tp["only"], tp["s"]),
           perp_eq=(G["gains"]["perp_eq"] * tp["perp"] * -math.sin(TH), G["gains"]["perp_eq"] * tp["perp"] * math.cos(TH)))
Y16 = {a: np.array([escaped(r) for r in T[a]], float) for a in arms16}

files = dict(none="rescue_none", rep="rescue_dir_neurospeak", doubt1="rescue_dir_self_doubt", only_100="dose_rep_only", doubt2="dose_doubt_dose", doubt_only2="dose_doubt_only",
             rep_25="axis_rep_25", rep_50="axis_rep_50", rep_75="axis_rep_75", only_109="axis_only_109", both="axis_both")
S = {a: load(f"runs/research/{f}_traces.jsonl") for a, f in files.items()}
sid = [r["id"] for r in S["none"]]
ao, asd, ar = (np.array([mean(A7[i]["proj"][k]) for i in sid]) for k in ("only", "s", "rep"))
aso = -s_ * ao + c * asd
zz = np.zeros(len(sid))
D7 = dict(none=(zz, zz), rep=(ar * c, ar * s_), rep_25=(.25 * ar * c, .25 * ar * s_), rep_50=(.5 * ar * c, .5 * ar * s_), rep_75=(.75 * ar * c, .75 * ar * s_),
          doubt1=(zz, asd), doubt2=(zz, 2 * asd), doubt_only2=(2 * aso * -s_, 2 * aso * c), only_100=(ao, zz), only_109=(1.09 * ao, zz), both=(ao, asd))
Y7 = {a: np.array([escaped(r) for r in S[a]], float) for a in files}


def logit(X, Y, iters=300):
    X = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        step = np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ (p - Y))
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    p = np.clip(1 / (1 + np.exp(-np.clip(X @ w, -30, 30))), 1e-12, 1 - 1e-12)
    return w, float((Y * np.log(p) + (1 - Y) * np.log(1 - p)).sum())


def stack(D, Y):
    x = np.concatenate([D[a][0] for a in D]); y = np.concatenate([D[a][1] for a in D]); e = np.concatenate([Y[a] for a in D])
    return x, y, e


def scan(x, y, e, label):
    L = [(d, logit(x * math.cos(math.radians(d)) + y * math.sin(math.radians(d)), e)[1]) for d in range(-20, 91)]
    b = max(L, key=lambda t: t[1])
    band = [d for d, l in L if l >= b[1] - 1.92]
    print(f"{label}: best {b[0]} deg, band {min(band)} to {max(band)}, loglik {b[1]:.1f}; at 42: {dict(L)[42]:.1f}, at 25: {dict(L)[25]:.1f}, at 0: {dict(L)[0]:.1f}")
    return b[0]


def show(D, Y, w, f):
    print("   ", {a: (round(float((1 / (1 + np.exp(-np.clip(f(*D[a]) @ w, -30, 30)))).sum()), 1), int(Y[a].sum())) for a in D})


x6, y6, e6 = stack(D16, Y16); x7, y7, e7 = stack(D7, Y7)
print("arm means (only, s):", {a: (round(v[0].mean(), 2), round(v[1].mean(), 2)) for a, v in D16.items()})
d16 = scan(x6, y6, e6, "study 16 alone")
dp = scan(np.r_[x6, x7], np.r_[y6, y7], np.r_[e6, e7], "study 7/8/11 + 16")
scan(x7, y7, e7, "study 7/8/11 alone")
for name, D, Y in (("16", D16, Y16), ("7/8/11", D7, Y7)):
    for lab, fn, xs in (("one axis, pooled best angle", lambda a, b: np.column_stack([a * math.cos(math.radians(dp)) + b * math.sin(math.radians(dp)), np.ones(len(a))]), (x6, y6, e6, x7, y7, e7)),):
        X = fn(np.r_[x6, x7], np.r_[y6, y7]); E = np.r_[e6, e7]
        w, ll = logit(X[:, 0], E)
        print(f"  {lab} on {name} (pred, obs):"); show(D, Y, w, fn)
free2 = lambda a, b: np.column_stack([a, b, np.ones(len(a))])
X2 = np.column_stack([np.r_[x6, x7], np.r_[y6, y7]]); w2, ll2 = logit(X2, np.r_[e6, e7])
print("two free slopes:", np.round(w2, 2), "loglik", round(ll2, 1), "implied angle", round(math.degrees(math.atan2(w2[1], w2[0])), 1))
print("   16:"); show(D16, Y16, w2, free2)
print("   7/8/11:"); show(D7, Y7, w2, free2)
# self-doubt lowered and raised get their own slopes
pos = lambda b: np.maximum(b, 0); neg = lambda b: np.minimum(b, 0)
fa = lambda a, b: np.column_stack([a, pos(b), neg(b), np.ones(len(a))])
X3 = np.column_stack([np.r_[x6, x7], pos(np.r_[y6, y7]), neg(np.r_[y6, y7])]); w3, ll3 = logit(X3, np.r_[e6, e7])
print("self-doubt lowered vs raised:", np.round(w3, 2), "loglik", round(ll3, 1), "gain over two slopes", round(ll3 - ll2, 1))
print("   16:"); show(D16, Y16, w3, fa)
print("   7/8/11:"); show(D7, Y7, w3, fa)
# per-run predictor: does a run's own activity predict its exit inside each repetition-only arm of study 16?
for a in ("only", "only_eq", "both", "tilt"):
    Xa = D16[a][0] * math.cos(math.radians(dp)) + D16[a][1] * math.sin(math.radians(dp))
    w, ll = logit(Xa, Y16[a])
    print(a, "within-arm slope on own push", round(w[0], 2), "mean push", round(Xa.mean(), 2))

# prefix-clustered bootstrap: the same prefix appears in every arm, so the likelihood bands above are too narrow
rng = np.random.default_rng(0)
A16 = list(D16); A7L = list(D7)
M6 = {a: (D16[a][0], D16[a][1], Y16[a]) for a in A16}; M7 = {a: (D7[a][0], D7[a][1], Y7[a]) for a in A7L}
def boot_arrays():
    i6, i7 = rng.integers(0, len(tid), len(tid)), rng.integers(0, len(sid), len(sid))
    x = np.concatenate([M6[a][0][i6] for a in A16] + [M7[a][0][i7] for a in A7L])
    y = np.concatenate([M6[a][1][i6] for a in A16] + [M7[a][1][i7] for a in A7L])
    e = np.concatenate([M6[a][2][i6] for a in A16] + [M7[a][2][i7] for a in A7L])
    return x, y, e
ang, inter = [], []
for b in range(300):
    x, y, e = boot_arrays()
    w, _ = logit(np.column_stack([x, y]), e)
    ang.append(math.degrees(math.atan2(w[1], w[0])))
    w, _ = logit(np.column_stack([x, y, x * y]), e)
    inter.append(w[2])
ang, inter = np.sort(ang), np.sort(inter)
print("clustered bootstrap, pooled angle from two slopes: median %.1f, 95%% %.1f to %.1f" % (ang[150], ang[7], ang[292]))
Xi = np.column_stack([np.r_[x6, x7], np.r_[y6, y7], np.r_[x6, x7] * np.r_[y6, y7]])
wi, lli = logit(Xi, np.r_[e6, e7])
print("interaction term x*y:", np.round(wi, 3), "loglik", round(lli, 1), "gain", round(lli - ll2, 1), "; bootstrap 95%% %.3f to %.3f" % (inter[7], inter[292]))
fi = lambda a, b: np.column_stack([a, b, a * b, np.ones(len(a))])
print("   16:"); show(D16, Y16, wi, fi)
print("   7/8/11:"); show(D7, Y7, wi, fi)
