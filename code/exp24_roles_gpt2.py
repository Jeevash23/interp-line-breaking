"""exp24: which line-breaking roles does GPT-2 Small lack? Exploratory (round 4, item 4).
WikiText-2 train only. Run once per model with the same code: gpt2 (the target) and pythia-410m
(the reference). Width 60 unless stated.

(a) Behaviour: AUROC, and the token coefficient (newline log-odds on a linear spline in characters
    remaining, knots 2,5,8,12,16,24,32, plus tokens on the line; points with <= 30 characters left,
    line 3 on; 40 paragraphs at widths 40 and 60).
(b) Count: ridge probes (5-fold R^2) for characters used and for tokens on the line, from the
    residual after each layer at decision points (30 paragraphs, width 60).
(c) Count use: at the decision token, add the class-mean shift for +6 characters used to the
    residual after layer l, for each l; effect on log P(newline) (60 items, 4-14 characters left).
(d) Width: items as in exp14 (previous line shortened by >= 10 characters; up to 100). The
    behavioural width effect, and the effect of patching only the line-start newline's residual
    after layer l with its value from the short run, for each l (a stored width that is used).
(e) Readers and counters by attention: for every head, the share of attention from decision points
    to the line-start newline and its correlation with characters used (30 paragraphs).
Run: python3 exp24_roles_gpt2.py <gpt2|pythia-410m> [--smoke]
"""
import sys, json, random, numpy as np, torch
from common import *

MN = sys.argv[1]; SMOKE = "--smoke" in sys.argv
NP = 6 if SMOKE else 40; NPP = 6 if SMOKE else 30; NIT = 8 if SMOKE else 100
tok, model = load(MN, attn="eager")
nl = newline_id(tok)
if MN == "gpt2":
    blocks = model.transformer.h; heads_attr = lambda b: b.attn
else:
    blocks = model.gpt_neox.layers; heads_attr = lambda b: b.attention
NL = len(blocks); Hn = model.config.num_attention_heads

RES = {}; ATT = {}; EDIT = {}
def bhook(l):
    def f(m, a, o):
        x = o[0] if isinstance(o, tuple) else o
        if l in EDIT:
            x = x.clone()
            for pos, vec, mode in EDIT[l]:
                if mode == "add": x[:, pos] += vec
                else: x[:, pos] = vec
            RES[l] = x.detach()
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        RES[l] = x.detach()
    return f
for l, b in enumerate(blocks): b.register_forward_hook(bhook(l))

def run(ids, attn=False):
    with torch.no_grad():
        out = model(torch.tensor([ids], device="cuda"), output_attentions=attn)
        lp = torch.log_softmax(out.logits[0].float(), -1)[:, nl]
    return lp, (out.attentions if attn else None)

def logodds(lp): return (lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-6))).cpu().numpy()
def spline(rem): return np.c_[rem, *[np.maximum(rem - k, 0) for k in [2, 5, 8, 12, 16, 24, 32]]]

paras = train_paragraphs(seed=24)
res = dict(model=MN)
# (a) behaviour and token coefficient
for W in (40, 60):
    X, y, sc, lab = [], [], [], []; n = 0
    for p in paras:
        if n >= NP: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        n += 1
        lp, _ = run(ids); lo = logodds(lp)
        for r in recs:
            sc.append(lo[r["i"]]); lab.append(r["label"])
            if r["remaining"] <= 30: X.append([r["remaining"], r["ntok_line"]]); y.append(lo[r["i"]])
    X = np.array(X, float); y = np.array(y)
    Mx = np.c_[np.ones(len(y)), spline(X[:, 0]), X[:, 1]]
    beta, *_ = np.linalg.lstsq(Mx, y, rcond=None)
    res.setdefault("behaviour", {})[W] = dict(auroc=auroc(sc, lab), token_coef=float(beta[-1]), points=len(y))
print("behaviour", res["behaviour"], flush=True)

# (b) probes and (e) attention, width 60
W = 60; feats = {l: [] for l in range(NL)}; used, ntok = [], []
share = []; n = 0
for p in paras[200:]:
    if n >= NPP: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 5: continue
    ids, recs = word_end_positions(tok, lines, W)
    if len(ids) > 700: continue
    n += 1
    _, att = run(ids, attn=True)
    nlpos = [i for i, t in enumerate(ids) if t == nl]
    for r in recs:
        prev = [q for q in nlpos if q < r["i"]]
        if not prev: continue
        q0 = prev[-1]
        for l in range(NL): feats[l].append(RES[l][0, r["i"]].float().cpu().numpy())
        used.append(r["used"]); ntok.append(r["ntok_line"])
        share.append(torch.stack([a[0, :, r["i"], q0] for a in att]).float().cpu().numpy())  # [NL, H]
def r2cv(X, y, lam=10.0):
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); out = []
    for f in range(5):
        te = idx[f::5]; tr = np.setdiff1d(idx, te); mu = X[tr].mean(0); Xt = X[tr] - mu; ym = y[tr].mean()
        w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (y[tr] - ym))
        pr = (X[te] - mu) @ w + ym; out.append(1 - ((y[te] - pr) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    return float(np.mean(out))
used = np.array(used, float); ntok = np.array(ntok, float)
res["probe_used"] = [r2cv(np.array(feats[l]), used) for l in range(NL)]
res["probe_ntok"] = [r2cv(np.array(feats[l]), ntok) for l in range(NL)]
S = np.array(share)  # [N, NL, H]
heads = []
for l in range(NL):
    for h in range(Hn):
        s = S[:, l, h]
        heads.append(dict(head=f"L{l}H{h}", share=float(s.mean()), r_used=float(np.corrcoef(s, used)[0, 1]) if s.std() > 0 else 0.0))
res["dilution_heads"] = sorted([d for d in heads if d["share"] >= 0.05], key=lambda d: d["r_used"])[:8]
res["reader_like_heads"] = sorted([d for d in heads if d["share"] >= 0.02], key=lambda d: -d["r_used"])[:8]
print("probes", [round(x, 2) for x in res["probe_used"]], flush=True)

# class means of each layer's residual by characters used, for (c)
sums = {l: {} for l in range(NL)}; cnts = {}
for p in train_paragraphs(seed=2):
    if sum(cnts.values()) > (1500 if SMOKE else 9000): break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    ids, recs = word_end_positions(tok, lines, W)
    if len(ids) > 700: continue
    run(ids)
    for r in recs:
        u = r["used"]; cnts[u] = cnts.get(u, 0) + 1
        for l in range(NL): sums[l][u] = sums[l].get(u, 0) + RES[l][0, r["i"]]
means = {l: {u: s / cnts[u] for u, s in sums[l].items() if cnts[u] >= 8} for l in range(NL)}

# (c) count use and (d) width
rng = random.Random(14); items = []
for p in train_paragraphs(seed=14):
    if len(items) >= NIT: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1))
    words = lines[k].split(" ")
    js = [j for j in range(2, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    prefix = " ".join(words[: js[0] + 1]); u = len(prefix)
    if u not in means[0] or (u + 6) not in means[0]: continue
    before = lines[:k]; pw = before[-1].split(" "); short = before[-1]
    while len(pw) > 2 and len(before[-1]) - len(short) < 10:
        pw = pw[:-1]; short = " ".join(pw)
    if len(before[-1]) - len(short) < 10: continue
    ib = tok("\n".join(before) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    is_ = tok("\n".join(before[:-1] + [short]) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    nb = [i for i, t in enumerate(ib) if t == nl][-1]; ns = [i for i, t in enumerate(is_) if t == nl][-1]
    if len(ib) - nb != len(is_) - ns: continue
    EDIT.clear()
    lb = float(run(ib)[0][-1]); ls_ = float(run(is_)[0][-1]); snl = {l: RES[l][0, ns].clone() for l in range(NL)}
    cnt, wid = [], []
    for l in range(NL):
        EDIT.clear(); EDIT[l] = [(len(ib) - 1, means[l][u + 6] - means[l][u], "add")]
        cnt.append(float(run(ib)[0][-1]) - lb)
        EDIT.clear(); EDIT[l] = [(nb, snl[l], "set")]
        wid.append(float(run(ib)[0][-1]) - lb)
    EDIT.clear()
    items.append(dict(u=u, width_behaviour=ls_ - lb, count_shift=cnt, width_patch=wid))
CS = np.array([d["count_shift"] for d in items]); WP = np.array([d["width_patch"] for d in items]); WB = np.array([d["width_behaviour"] for d in items])
res["n_items"] = len(items)
res["width_behaviour"] = [float(WB.mean()), float(WB.std() / np.sqrt(len(WB)))]
res["count_shift_by_layer"] = [[float(CS[:, l].mean()), float(CS[:, l].std() / np.sqrt(len(CS)))] for l in range(NL)]
res["width_patch_by_layer"] = [[float(WP[:, l].mean()), float(WP[:, l].std() / np.sqrt(len(WP)))] for l in range(NL)]
save_json(dict(summary=res, items=items), f"exp24_roles_{MN}" + ("_smoke" if SMOKE else "") + ".json")
print(json.dumps(res, indent=1))
