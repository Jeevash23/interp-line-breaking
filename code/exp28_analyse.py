"""exp28_analyse (EXPLORATORY, CPU): analysis of outputs/exp28_tokenbias.npz.
 1. Token coefficient (newline log-odds on a linear spline in characters remaining, knots 2,5,8,12,16,24,32,
    plus tokens on the line; points with <= 30 characters left) per text and width: intact, L5H7 ablated,
    L5H7+L4H14 ablated; 2,000 paragraph bootstraps (seed 0). Mean tokens on the line and characters per token.
 2. The same coefficient for the L5H6 and L5H7 count projections and the residual after layer 5, with the
    spline in characters used (the counters see characters used, not remaining).
 3. How well tokens on the line and characters used each explain L5H6's line-start attention.
Usage: python exp28_analyse.py [npz]"""
import sys, json, numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "outputs"
f = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "exp28_tokenbias.npz"
z = np.load(f); cols = list(z["cols"]); C = {c: i for i, c in enumerate(cols)}
def spline(x): return np.c_[x, *[np.maximum(x - k, 0) for k in [2, 5, 8, 12, 16, 24, 32]]]
def coef(x, t, y):
    M = np.c_[np.ones(len(y)), spline(x), t]; return float(np.linalg.lstsq(M, y, rcond=None)[0][-1])
def boot(d, ycol, xcol="remaining", sel=None, nb=2000):
    d = d if sel is None else d[sel]
    x, t, y, g = d[:, C[xcol]], d[:, C["ntok"]], d[:, C[ycol]], d[:, C["para"]].astype(int)
    est = coef(x, t, y); ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}; rs = np.random.default_rng(0); b = []
    for _ in range(nb):
        ii = np.concatenate([idx[k] for k in rs.choice(ug, len(ug))]); b.append(coef(x[ii], t[ii], y[ii]))
    return est, [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def r2(x, y, knots):
    M = np.c_[np.ones(len(y)), x, *[np.maximum(x - k, 0) for k in knots]]
    res = y - M @ np.linalg.lstsq(M, y, rcond=None)[0]; return float(1 - res.var() / y.var())
res = {}
for key in z.files:
    if key == "cols": continue
    d = z[key]; sel = d[:, C["remaining"]] <= 30; s = d[sel]
    nb = 200 if "smoke" in f.name else 2000
    r = dict(n_points=int(sel.sum()), n_paragraphs=int(len(np.unique(d[:, C["para"]]))),
             mean_ntok=float(s[:, C["ntok"]].mean()), chars_per_tok=float((s[:, C["used"]] / s[:, C["ntok"]]).mean()),
             mean_used=float(s[:, C["used"]].mean()))
    for y in ("lo", "lo_abl7", "lo_abl7_14"): r[y] = boot(d, y, sel=sel, nb=nb)
    for y in ("p6", "p7", "p5res"): r[y] = coef(s[:, C["used"]], s[:, C["ntok"]], s[:, C[y]])
    r["p6p7"] = coef(s[:, C["used"]], s[:, C["ntok"]], s[:, C["p6"]] + s[:, C["p7"]])
    a6 = d[:, C["att6"]]
    r["att6_r2_ntok"] = r2(d[:, C["ntok"]], a6, [3, 6, 10, 15, 20]); r["att6_r2_used"] = r2(d[:, C["used"]], a6, [10, 20, 30, 45, 60, 80])
    r["att6_r2_both"] = float(1 - np.var(a6 - np.c_[np.ones(len(a6)), d[:, C["ntok"]], 1 / (1 + d[:, C["ntok"]]), d[:, C["used"]]] @ np.linalg.lstsq(np.c_[np.ones(len(a6)), d[:, C["ntok"]], 1 / (1 + d[:, C["ntok"]]), d[:, C["used"]]], a6, rcond=None)[0]) / np.var(a6))
    res[key.replace("__", "|")] = r
    print(f"{key:22s} n={r['n_points']:5d} ntok={r['mean_ntok']:5.1f} c/t={r['chars_per_tok']:.2f} intact={r['lo'][0]:+.3f} [{r['lo'][1][0]:+.3f},{r['lo'][1][1]:+.3f}] "
          f"abl7={r['lo_abl7'][0]:+.3f} abl7+14={r['lo_abl7_14'][0]:+.3f} | heads tokcoef p6={r['p6']:+.3f} p7={r['p7']:+.3f} sum={r['p6p7']:+.3f} res5={r['p5res']:+.3f} | att6 R2 ntok={r['att6_r2_ntok']:.2f} used={r['att6_r2_used']:.2f}")
(OUT / f.name.replace(".npz", "_analysis.json")).write_text(json.dumps(res, indent=1))
