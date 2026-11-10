"""exp21: pilot for the second confirmation round. Exploratory, train split, new seeds.

exp20 suggested that the counting heads the decision depends on (exp04: L2H7, L4H15, L5H6, L5H7)
are re-used at the line-start newline token to store the length of the line that just ended.
Before writing PROTOCOL-2 this pilot measures, on train material only:
 A  Newline-only mean-ablation of those four heads (COUNT4): the width effect (previous line
    shortened by >= 10 characters; total behaviour, both runs ablated alike) with COUNT4 replaced
    by its newline-position mean at (i) every newline, (ii) the last newline only; controls:
    5 random sets of 4 heads from layers 0-5 outside the exp04 list, at every newline.
 B  Dissociation on natural text (width 60): separation and AUROC of the newline log-odds with
    COUNT4 ablated at newlines only, against at every position.
 C  Probe transfer: ridge from COUNT4's attention-output inputs (256 dims) to characters used,
    fitted at word ends of half the paragraphs, applied at newline tokens of the other half to
    predict the length of the line just ended (widths 40/60/80 pooled). Controls: 5 random head
    sets of 4 from layers 0-5.
 D  Previous-line tokens: in natural text (widths 40, 60), the coefficient of the previous line's
    token count in the newline log-odds, given a spline in characters remaining, tokens on the
    current line and the previous line's character length; intact, L5H7 ablated at newlines
    only, L5H11 ablated at newlines only.
"""
import json, random, numpy as np, torch
from common import *

tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
LM = 6
COUNT4 = [(2, 7), (4, 15), (5, 6), (5, 7)]
EXP04 = {(4, 15), (5, 6), (4, 10), (2, 7), (5, 7), (2, 13), (6, 11), (2, 11)}
pool = [(l, h) for l in range(LM) for h in range(Hn) if (l, h) not in EXP04 and l >= 1]
rs = np.random.default_rng(21)
RAND = [[pool[i] for i in rs.choice(len(pool), 4, replace=False)] for _ in range(5)]

CAP = {}; ST = dict(heads=[], mask=None, capture=False)
def hook(l):
    def f(m, a):
        z = a[0]
        if ST["capture"]: CAP[l] = z[0].detach().clone()
        hs = [h for (ll, h) in ST["heads"] if ll == l]
        if not hs or ST["mask"] is None: return None
        z = z.clone()
        for h in hs: z[:, ST["mask"], h * dh:(h + 1) * dh] = MEAN[l][h * dh:(h + 1) * dh]
        return (z,)
    return f
for l in range(LM): layers[l].attention.dense.register_forward_pre_hook(hook(l))
MEAN = {}

def run(ids, heads=(), where=None):
    """where: None, 'nl' (every newline), 'last' (last newline), 'all' (every position)."""
    t = torch.tensor([ids], device="cuda")
    if where is None or not heads: ST.update(heads=[], mask=None)
    else:
        m = torch.zeros(len(ids), dtype=torch.bool, device="cuda")
        if where == "all": m[:] = True
        elif where == "nl": m[t[0] == nl] = True
        elif where == "last": m[[i for i, x in enumerate(ids) if x == nl][-1]] = True
        ST.update(heads=list(heads), mask=m)
    with torch.no_grad():
        lp = torch.log_softmax(model(t).logits[0].float(), -1)[:, nl].cpu().numpy()
    ST.update(heads=[], mask=None)
    return lp

# newline-position means (wrapped train paragraphs, seed 6, widths 40/60/80)
acc = {l: [] for l in range(LM)}
ST["capture"] = True
for q, p in enumerate(train_paragraphs(seed=6)[:30]):
    lines = wrap(p, (40, 60, 80)[q % 3])
    if lines is None: continue
    ids = tok("\n".join(lines), add_special_tokens=False)["input_ids"][:700]
    run(ids)
    pos = [i for i, x in enumerate(ids) if x == nl]
    for l in range(LM): acc[l].append(CAP[l][pos])
ST["capture"] = False
MEAN = {l: torch.cat(acc[l]).mean(0) for l in range(LM)}

out = {}
# ---------------- A
rng = random.Random(21); items = []
for p in train_paragraphs(seed=23):
    if len(items) >= 100: break
    W = 60; lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1)); words = lines[k].split(" ")
    js = [j for j in range(2, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    prefix = " ".join(words[: js[0] + 1]); before = lines[:k]
    pw = before[-1].split(" "); short = before[-1]
    while len(pw) > 2 and len(before[-1]) - len(short) < 10: pw = pw[:-1]; short = " ".join(pw)
    if len(before[-1]) - len(short) < 10: continue
    ib = tok("\n".join(before) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    is_ = tok("\n".join(before[:-1] + [short]) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    nb = [i for i, t in enumerate(ib) if t == nl][-1]; ns = [i for i, t in enumerate(is_) if t == nl][-1]
    if len(ib) - nb != len(is_) - ns: continue
    items.append((ib, is_))
CONDA = {"intact": ((), None), "COUNT4_nl": (COUNT4, "nl"), "COUNT4_last": (COUNT4, "last")}
for q, r in enumerate(RAND): CONDA[f"rand{q}_nl"] = (r, "nl")
A = {}
for c, (hs, wh) in CONDA.items():
    A[c] = [float(run(s, hs, wh)[-1] - run(b, hs, wh)[-1]) for b, s in items]
E0 = np.mean(A["intact"])
out["A"] = dict(n=len(items), random_sets=[[f"L{l}H{h}" for l, h in r] for r in RAND],
                eff={c: float(np.mean(v)) for c, v in A.items()}, share_removed={c: float(1 - np.mean(v) / E0) for c, v in A.items()})
print("A", json.dumps(out["A"]), flush=True)

# ---------------- B
nat = []
for p in train_paragraphs(seed=24):
    if len(nat) >= 25: break
    lines = wrap(p, 60)
    if lines is None or len(lines) < 5: continue
    ids, recs = word_end_positions(tok, lines, 60)
    if len(ids) <= 700: nat.append((ids, recs))
B = {}
for c, (hs, wh) in {"intact": ((), None), "COUNT4_nl": (COUNT4, "nl"), "COUNT4_all": (COUNT4, "all"), "rand0_nl": (RAND[0], "nl")}.items():
    sc, lab = [], []
    for ids, recs in nat:
        lp = run(ids, hs, wh)
        for r in recs:
            if r["remaining"] <= 12 or r["label"] == 1:
                v = lp[r["i"]]; sc.append(v - np.log1p(-np.exp(min(v, -1e-6)))); lab.append(r["label"])
    sc, lab = np.array(sc), np.array(lab)
    B[c] = dict(separation=float(sc[lab == 1].mean() - sc[lab == 0].mean()), auroc=auroc(sc, lab))
out["B"] = B; print("B", json.dumps(B), flush=True)

# ---------------- C
def collect(seed, n_per_w):
    rows = []  # (paragraph id, kind, target, z dict)
    pid = 0
    for W in (40, 60, 80):
        got = 0
        for p in train_paragraphs(seed=seed):
            if got >= n_per_w: break
            lines = wrap(p, W)
            if lines is None or len(lines) < 5: continue
            ids, recs = word_end_positions(tok, lines, W)
            if len(ids) > 700: continue
            ST["capture"] = True; run(ids); ST["capture"] = False
            Z = {l: CAP[l].cpu().numpy() for l in range(LM)}
            for r in recs: rows.append((pid, 0, r["used"], r["i"], Z))
            nls = [i for i, t in enumerate(ids) if t == nl]
            for li, i in enumerate(nls):
                if li >= 1: rows.append((pid, 1, len(lines[li]), i, Z))
            pid += 1; got += 1
    return rows
rowsC = collect(25, 30)
def feats(heads):
    return np.array([np.concatenate([Z[l][i, h * dh:(h + 1) * dh] for l, h in heads]) for _, _, _, i, Z in rowsC])
pidC = np.array([r[0] for r in rowsC]); kind = np.array([r[1] for r in rowsC]); y = np.array([r[2] for r in rowsC], float)
half = pidC % 2 == 0
def ridge(X, y, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(X.shape[1]), Xs.T @ (y - y.mean()))
    return lambda Z: ((Z - mu) / sd) @ w + y.mean()
def transfer(heads):
    X = feats(heads); tr = half & (kind == 0)
    f = ridge(X[tr], y[tr])
    te_nl = (~half) & (kind == 1); te_we = (~half) & (kind == 0)
    pn = f(X[te_nl]); pw = f(X[te_we])
    fn = ridge(X[half & (kind == 1)], y[half & (kind == 1)])
    return dict(r_nl=float(np.corrcoef(pn, y[te_nl])[0, 1]), slope_nl=float(np.polyfit(pn, y[te_nl], 1)[0]),
                bias_nl=float(np.mean(pn - y[te_nl])), r_wordend=float(np.corrcoef(pw, y[te_we])[0, 1]),
                r_nl_within=float(np.corrcoef(fn(X[te_nl]), y[te_nl])[0, 1]), n_nl=int(te_nl.sum()))
C = {"COUNT4": transfer(COUNT4)}
for q, r in enumerate(RAND): C[f"rand{q}"] = transfer(r)
out["C"] = C; print("C", json.dumps(C), flush=True)
del rowsC

# ---------------- D
def spline(rem): return np.c_[rem, *[np.maximum(rem - k, 0) for k in [2, 5, 8, 12, 16, 24, 32]]]
Dd = {}
for W in (40, 60):
    items = []
    for p in train_paragraphs(seed=26):
        if len(items) >= 40: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        nls = [i for i, t in enumerate(ids) if t == nl]
        items.append((ids, recs, nls))
    Dd[W] = {}
    for c, (hs, wh) in {"intact": ((), None), "L5H7_nl": ([(5, 7)], "nl"), "L5H11_nl": ([(5, 11)], "nl")}.items():
        X, yy, g = [], [], []
        for k, (ids, recs, nls) in enumerate(items):
            lp = run(ids, hs, wh)
            for r in recs:
                if r["remaining"] > 30: continue
                li = r["line"]
                prev_ntok = nls[li - 1] - (nls[li - 2] + 1 if li >= 2 else 0)
                v = lp[r["i"]]; X.append((r["remaining"], r["ntok_line"], r["prevlen"], prev_ntok)); yy.append(v - np.log1p(-np.exp(min(v, -1e-6)))); g.append(k)
        X = np.array(X, float); yy = np.array(yy); g = np.array(g)
        M = np.c_[np.ones(len(yy)), spline(X[:, 0]), X[:, 1:]]
        w = np.linalg.lstsq(M, yy, rcond=None)[0]
        br = np.random.default_rng(0); bs = []
        groups = [np.where(g == k)[0] for k in range(len(items))]
        for _ in range(300):
            ii = np.concatenate([groups[j] for j in br.integers(0, len(items), len(items))])
            bs.append(np.linalg.lstsq(M[ii], yy[ii], rcond=None)[0][-3:])
        bs = np.array(bs)
        Dd[W][c] = dict(n=len(yy), ntok_line=float(w[-3]), prevlen=float(w[-2]), prev_ntok=float(w[-1]),
                        prev_ntok_ci=[float(np.percentile(bs[:, 2], 2.5)), float(np.percentile(bs[:, 2], 97.5))],
                        prevlen_ci=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))])
        print("D", W, c, json.dumps(Dd[W][c]), flush=True)
out["D"] = Dd
save_json(out, "exp21_newline_reuse_pilot.json")
