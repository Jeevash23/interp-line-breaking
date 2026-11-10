"""D1-X exp e02 (EXPLORATORY, CHARTER.md): where the previous line's length enters, on a model's
exploration part. Usage: xm_e02_width.py <model> <part> [--smoke]
Width items (xm_common.width_items: width 60, previous line >= 10 characters shorter, letter-ending
decision with 4..14 left), up to 100.
 A total effect: log P(newline) at the decision, short-previous-line prompt minus base.
 B residual patch: the base run's line-start marker residual after layer l replaced by its
   state in the short run, for every l.
 C writers: one head's o-proj input slice at the line-start marker patched from the short run,
   for every head (all layers).
 D readers: with the marker residual after layer Lw patched (Lw = the layer in the first half
   of the model with the largest B effect), each head in layers > Lw frozen at the decision token
   to its base-run value; mediation share = 1 - mean(frozen effect) / mean(patch effect).
"""
import sys, json, numpy as np, torch
from xm_common import *

MODEL, PART = sys.argv[1], sys.argv[2]; SMOKE = "--smoke" in sys.argv
m = M(MODEL); tok = m.tok
items = width_items(load_paras(PART, seed=0), m, 6 if SMOKE else 100)
HEADS = m.heads(); CH = 32
tot, B, C = [], [], []
cache = []
for ib, is_, nb, ns in items:
    m.patch, m.rpatch = {}, {}
    lb = m.lp(ib, capture=True)[-1]; zb = {l: v.clone() for l, v in m.cap.items()}; rb = {l: v.clone() for l, v in m.rcap.items()}
    ls = m.lp(is_, capture=True)[-1]; zs = {l: v.clone() for l, v in m.cap.items()}; rs = {l: v.clone() for l, v in m.rcap.items()}
    tot.append(ls - lb)
    m.rpatch = {l: [(nb, rs[l][ns], l)] for l in range(m.L)}
    B.append(m.lp_last_batch(ib, m.L) - lb); m.rpatch = {}
    row = np.zeros(len(HEADS))
    for s in range(0, len(HEADS), CH):
        part = HEADS[s:s + CH]; m.patch = {}
        for r, (l, h) in enumerate(part): m.patch.setdefault(l, []).append((h, nb, zs[l][ns, h * m.dh:(h + 1) * m.dh], r))
        row[s:s + len(part)] = m.lp_last_batch(ib, len(part)) - lb
    m.patch = {}; C.append(row)
    cache.append((ib, nb, ns, lb, {l: zb[l][-1].clone() for l in range(m.L)}, {l: rs[l][ns].clone() for l in range(m.L)}))
tot = np.array(tot); B = np.array(B); C = np.array(C)
Bm = B.mean(0); Lw = int(np.argmax(Bm[: m.L // 2]))
print("total", tot.mean(), "resid by layer", np.round(Bm, 3).tolist(), "Lw", Lw, flush=True)
RH = [(l, h) for l, h in HEADS if l > Lw]
Pw, Dm = [], []
for ib, nb, ns, lb, zdec, rsn in cache:
    m.rpatch = {Lw: [(nb, rsn[Lw])]}; m.patch = {}
    pw = m.lp(ib)[-1] - lb; Pw.append(pw)
    row = np.zeros(len(RH))
    for s in range(0, len(RH), CH):
        part = RH[s:s + CH]; m.patch = {}
        for r, (l, h) in enumerate(part): m.patch.setdefault(l, []).append((h, len(ib) - 1, zdec[l][h * m.dh:(h + 1) * m.dh], r))
        row[s:s + len(part)] = m.lp_last_batch(ib, len(part)) - lb
    Dm.append(row); m.patch, m.rpatch = {}, {}
Pw = np.array(Pw); Dm = np.array(Dm)
share = 1 - Dm.mean(0) / Pw.mean()
names = [f"L{l}H{h}" for l, h in HEADS]; rnames = [f"L{l}H{h}" for l, h in RH]
Cm = C.mean(0); Cse = C.std(0) / np.sqrt(len(C))
out = dict(model=MODEL, part=PART, smoke=SMOKE, n=len(items), total=float(tot.mean()), total_se=float(tot.std() / np.sqrt(len(tot))),
           resid_by_layer=Bm.tolist(), Lw=Lw, patch_effect_Lw=float(Pw.mean()),
           writers_top=[(names[i], float(Cm[i]), float(Cse[i])) for i in np.argsort(-np.abs(Cm))[:20]],
           writers_all=dict(zip(names, Cm.tolist())),
           readers_top=[(rnames[i], float(share[i])) for i in np.argsort(-share)[:20]], readers_all=dict(zip(rnames, share.tolist())))
print(json.dumps({k: out[k] for k in ("n", "total", "resid_by_layer", "Lw", "patch_effect_Lw", "writers_top", "readers_top")}, indent=0), flush=True)
save_json(out, f"xm_e02_{MODEL}_{PART}{'_smoke' if SMOKE else ''}.json")
