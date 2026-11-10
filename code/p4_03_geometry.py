"""p4_03 (round 4, CONFIRMATORY under PROTOCOL-4.md): the geometry of the count on the named split, with the
analysis of exp23_geometry.py (split-half part and reader-attention part) applied to freshly captured
activations, since exp03's saved activations cover WikiText-2 train only.

Capture (Pythia-410M, float32, eager attention): widths 40 and 60, the first 40 paragraphs per width (seed-100
order) that wrap into >= 5 lines and tokenise to <= 700 tokens. At every word-end decision point: the residual
after layers 4, 5, 6, 8, 10 and 12, characters used and remaining, the paragraph index, and the attention of
L5H6, L5H7, L6H2, L8H15, L17H1 and L8H11 to the line-start newline.
Split-half (exp23 part c): classes of characters used with >= 20 points; halves by paragraph parity; classes
with >= 8 points in each half; top-6 directions of half A's centred class means; half B's means projected on
them and fitted by a straight line and by a natural cubic spline (knots at the 10/30/50/70/90% quantiles of the
classes); variance-weighted R^2; A-B reliability per direction; the same with labels shuffled (seed 0).
Reader attention (exp23 part b): the mean share by characters remaining (classes with >= 15 points), the
location of its peak, and the share of between-class variance explained by the best monotone fit.
Run: python3 p4_03_geometry.py <train|conf4-pythia410m> [--smoke]
"""
import sys, json, numpy as np, torch
from common import *
from newtext2 import paras

SPLIT = sys.argv[1]; SMOKE = "--smoke" in sys.argv
assert SPLIT in ("train", "conf4-pythia410m")
NP = 6 if SMOKE else 40
LAYERS = (4, 5, 6, 8, 10, 12)
HEADS = dict(L5H6=(5, 6), L5H7=(5, 7), L6H2=(6, 2), L8H15=(8, 15), L17H1=(17, 1), L8H11=(8, 11))
tok, model = load("pythia-410m", attn="eager")
nl = newline_id(tok)
layers = model.gpt_neox.layers
RES = {}
for l in LAYERS: layers[l].register_forward_hook(lambda m, a, o, l=l: RES.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].detach()))

data = {}
for W in (40, 60):
    X = {l: [] for l in LAYERS}; used, rem, para, A = [], [], [], {h: [] for h in HEADS}; n = 0
    for p in paras(SPLIT, 100):
        if n >= NP: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        with torch.no_grad(): att = model(torch.tensor([ids], device="cuda"), output_attentions=True).attentions
        nlp = [i for i, t in enumerate(ids) if t == nl]
        for r in recs:
            q0 = [q for q in nlp if q < r["i"]][-1]
            for l in LAYERS: X[l].append(RES[l][r["i"]].float().cpu().numpy())
            used.append(r["used"]); rem.append(r["remaining"]); para.append(n)
            for h, (l, hh) in HEADS.items(): A[h].append(float(att[l][0, hh, r["i"], q0]))
        n += 1
    data[W] = dict(X={l: np.array(v) for l, v in X.items()}, used=np.array(used), rem=np.array(rem), para=np.array(para), A={h: np.array(v) for h, v in A.items()})

def spline_basis(x, knots):
    x = np.asarray(x, float); K = len(knots)
    d = lambda k: (np.clip(x - knots[k], 0, None) ** 3 - np.clip(x - knots[-1], 0, None) ** 3) / (knots[-1] - knots[k])
    return np.stack([np.ones_like(x), x] + [d(k) - d(K - 2) for k in range(K - 2)], 1)
def fitr2(Y, X):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None); E = Y - X @ beta
    return 1 - (E ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
def split_half(X, used, para, classes, shuffle=False):
    u = used.copy()
    if shuffle: u = np.random.default_rng(0).permutation(u)
    Ah = para % 2 == 0; Bh = ~Ah
    ok = [c for c in classes if (u[Ah] == c).sum() >= 8 and (u[Bh] == c).sum() >= 8]
    MA = np.stack([X[Ah & (u == c)].mean(0) for c in ok]); MB = np.stack([X[Bh & (u == c)].mean(0) for c in ok])
    mu = MA.mean(0); _, _, Vt = np.linalg.svd(MA - mu, full_matrices=False); V = Vt[:6]
    YA = (MA - mu) @ V.T; YB = (MB - mu) @ V.T
    xs = np.array(ok, float); knots = np.quantile(xs, [0.1, 0.3, 0.5, 0.7, 0.9])
    lin = fitr2(YB, np.stack([np.ones_like(xs), xs], 1)); spl = fitr2(YB, spline_basis(xs, knots))
    w = YB.var(0); w = w / w.sum()
    rel = [float(np.corrcoef(YA[:, k], YB[:, k])[0, 1]) for k in range(V.shape[0])]
    return dict(n_classes=len(ok), r2_linear=float((w * lin).sum()), r2_spline=float((w * spl).sum()),
                r2_linear_by_pc=[float(x) for x in lin], r2_spline_by_pc=[float(x) for x in spl],
                reliability_by_pc=rel, var_share_B=[float(x) for x in w])
def pava(y):
    out = []
    for v in y:
        out.append([v, 1])
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            b = out.pop(); a = out.pop(); out.append([(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1]])
    return np.concatenate([[v] * n for v, n in out])

res = dict(split=SPLIT, split_half={}, attention={})
for W, d in data.items():
    classes = [u for u in range(W + 1) if (d["used"] == u).sum() >= 20]
    res["n_points_" + str(W)] = int(len(d["used"]))
    for l in LAYERS:
        res["split_half"].setdefault(W, {})[l] = dict(real=split_half(d["X"][l], d["used"], d["para"], classes),
                                                      shuffled=split_half(d["X"][l], d["used"], d["para"], classes, True))
    for h in HEADS:
        a = d["A"][h]; rr = [r for r in range(0, W) if (d["rem"] == r).sum() >= 15]
        prof = np.array([a[d["rem"] == r].mean() for r in rr]); ss = ((prof - prof.mean()) ** 2).sum()
        mono = max(1 - ((prof - f) ** 2).sum() / ss for f in (pava(prof), -pava(-prof))) if ss > 0 else 1.0
        res["attention"].setdefault(W, {})[h] = dict(peak_at_remaining=int(rr[int(np.argmax(prof))]), min_at_remaining=int(rr[int(np.argmin(prof))]),
            monotone_r2=float(mono), corr_with_remaining=float(np.corrcoef(d["rem"], a)[0, 1]), profile=[float(x) for x in prof], remaining=rr)
save_json(res, f"p4_03_geometry_{SPLIT}" + ("_smoke" if SMOKE else "") + ".json")
for W in (40, 60):
    for l in LAYERS:
        r = res["split_half"][W][l]["real"]
        print(W, l, round(r["var_share_B"][0], 2), round(r["r2_linear_by_pc"][0], 2), round(r["r2_linear_by_pc"][1], 2), round(r["r2_spline_by_pc"][1], 2), round(r["reliability_by_pc"][1], 2))
    print({h: (v["peak_at_remaining"], round(v["monotone_r2"], 2)) for h, v in res["attention"][W].items()})
