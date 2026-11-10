"""D1-X exp e03 (EXPLORATORY, CHARTER.md): which heads carry the character count, and which carry
the finished line's length at the newline? Plus each head's general damage when ablated.
Usage: xm_e03_probe.py <model> <part> [--smoke]
 1 Count probe at decision points: each head's o-proj input slice at letter-ending word ends
   (<= 30 characters remaining), 40 paragraphs at each of widths 40/60/80. Target: characters
   used on the line, with tokens-on-the-line partialled out (residual of a linear fit on
   ntok_line within each width). Ridge (alpha = 10 x mean feature variance x dh), 5-fold CV by
   paragraph; out-of-fold R^2. Also the raw R^2 for characters used.
 2 Stored-length probe at the newline: each head's slice at line-start marker positions (width 60,
   the same paragraphs, lines 2 on). Target: the character length of the line just ended,
   with its token count partialled out. Same ridge and CV.
 3 General damage: increase in mean next-token cross-entropy over non-decision positions of 10
   width-60 paragraphs when the head is mean-ablated at all positions (means as e01).
"""
import sys, json, numpy as np, torch
from xm_common import *

MODEL, PART = sys.argv[1], sys.argv[2]; SMOKE = "--smoke" in sys.argv
m = M(MODEL); tok = m.tok
paras = load_paras(PART, seed=0)
NP, ND = (4, 3) if SMOKE else (40, 10)

def ridge_cv(X, y, g, k=5):
    X = X.astype(np.float64); y = y.astype(np.float64)
    ug = np.unique(g); fold = {u: i % k for i, u in enumerate(np.random.default_rng(0).permutation(ug))}
    f = np.array([fold[u] for u in g]); pred = np.zeros_like(y)
    for q in range(k):
        tr, te = f != q, f == q
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xt = (X[tr] - mu) / sd; ym = y[tr].mean()
        a = 10.0 * Xt.shape[1]
        w = np.linalg.solve(Xt.T @ Xt + a * np.eye(Xt.shape[1]), Xt.T @ (y[tr] - ym))
        pred[te] = ((X[te] - mu) / sd) @ w + ym
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

def partial(y, z, grp):
    r = np.zeros_like(y, dtype=float)
    for u in np.unique(grp):
        s = grp == u; M_ = np.c_[np.ones(s.sum()), z[s]]
        r[s] = y[s] - M_ @ np.linalg.lstsq(M_, y[s], rcond=None)[0]
    return r

# ---- 1 and 2: collect
Fd = {(l, h): [] for l, h in m.heads()}; Fn = {(l, h): [] for l, h in m.heads()}
used, ntok, wid, gd = [], [], [], []
llen, lntok, gn = [], [], []
for W in (40, 60, 80):
    for k, (ids, recs, lines) in enumerate(wrapped_items(paras, W, NP, tok=tok)):
        m.lp(ids, capture=True)
        pts = [r for r in recs if r["remaining"] <= 30]
        ii = torch.tensor([r["i"] for r in pts], device="cuda")
        for l in range(m.L):
            z = m.cap[l][ii].float().cpu().numpy()
            for h in range(m.H): Fd[(l, h)].append(z[:, h * m.dh:(h + 1) * m.dh])
        used += [r["used"] for r in pts]; ntok += [r["ntok_line"] for r in pts]; wid += [W] * len(pts); gd += [W * 1000 + k] * len(pts)
        if W == 60:
            nls = m.nlpos(ids)
            # line j (0-based) ends at marker nls[j]; keep lines 1.. so the previous marker exists
            sel = [(j, nls[j]) for j in range(1, min(len(nls), len(lines) - 1))]
            if not sel: continue
            jj = torch.tensor([p for _, p in sel], device="cuda")
            for l in range(m.L):
                z = m.cap[l][jj].float().cpu().numpy()
                for h in range(m.H): Fn[(l, h)].append(z[:, h * m.dh:(h + 1) * m.dh])
            for j, p in sel:
                llen.append(len(lines[j])); lntok.append(p - nls[j - 1])  # tokens after the previous marker up to and incl. this one
                gn.append(k)
used, ntok, wid, gd = map(np.array, (used, ntok, wid, gd)); llen, lntok, gn = map(np.array, (llen, lntok, gn))
yd = partial(used.astype(float), ntok.astype(float), wid)
yn = partial(llen.astype(float), lntok.astype(float), np.zeros_like(gn))
R = {}
for l, h in m.heads():
    Xd = np.concatenate(Fd[(l, h)]); Xn = np.concatenate(Fn[(l, h)])
    R[f"L{l}H{h}"] = dict(count_r2_partial=ridge_cv(Xd, yd, gd), count_r2_raw=ridge_cv(Xd, used.astype(float), gd),
                         stored_r2_partial=ridge_cv(Xn, yn, gn), stored_r2_raw=ridge_cv(Xn, llen.astype(float), gn))
print("probes done", len(used), len(llen), flush=True)

# ---- 3 general damage
acc = {l: [] for l in range(m.L)}
for q, p in enumerate(train_paragraphs(seed=6)[:30]):
    lines = wrap(p, (40, 60, 80)[q % 3])
    if lines is None: continue
    ids = tok("\n".join(lines), add_special_tokens=False)["input_ids"][:700]
    m.lp(ids, capture=True)
    for l in range(m.L): acc[l].append(m.cap[l].float().mean(0))
ZM = {l: torch.stack(acc[l]).mean(0) for l in range(m.L)}
dm = wrapped_items(paras, 60, ND, tok=tok)
def ce():
    tot, n = 0.0, 0
    for ids, recs, _ in dm:
        dec = {r["i"] for r in recs}
        with torch.no_grad():
            lg = torch.log_softmax(m.model(torch.tensor([ids], device="cuda")).logits[0].float(), -1)
        t = torch.tensor(ids[1:], device="cuda"); nll = -lg[:-1].gather(1, t[:, None])[:, 0].cpu().numpy()
        keep = np.array([i not in dec for i in range(len(ids) - 1)])
        tot += nll[keep].sum(); n += keep.sum()
    return tot / n
ce0 = ce()
for l, h in m.heads():
    m.patch = {l: [(h, slice(None), ZM[l][h * m.dh:(h + 1) * m.dh])]}
    R[f"L{l}H{h}"]["ce_increase"] = float(ce() - ce0); m.patch = {}
out = dict(model=MODEL, part=PART, smoke=SMOKE, n_decision=len(used), n_newline=len(llen), ce_intact=float(ce0), heads=R,
           top_count=sorted(R, key=lambda k: -R[k]["count_r2_partial"])[:15], top_stored=sorted(R, key=lambda k: -R[k]["stored_r2_partial"])[:15])
for k in out["top_count"][:10]: print("count", k, {a: round(b, 3) for a, b in R[k].items()}, flush=True)
for k in out["top_stored"][:10]: print("stored", k, {a: round(b, 3) for a, b in R[k].items()}, flush=True)
save_json(out, f"xm_e03_{MODEL}_{PART}{'_smoke' if SMOKE else ''}.json")
