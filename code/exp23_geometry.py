"""exp23: the geometry of the character count in Pythia-410M, against the Claude 3.5 Haiku account
(Gurnee et al., arXiv 2601.04480: count on a curved low-dimensional manifold, boundary heads
rotating it). Exploratory (round 4, item 3). CPU only; reads exp03's saved activations
(WikiText-2 train, widths 40 and 60, 40 paragraphs each; residual after layers -1..12 in float16).

(a) Class means of the residual at decision points by characters used (classes with >= 20 points),
    after layers 3..12. For each layer:
    - participation ratio and variance share of the top 1, 2, 3 and 6 principal components of the
      centred class means (a scalar code puts nearly everything on PC1);
    - "ringing": the cosine-similarity matrix of the centred class means; a curved manifold gives
      negative similarity at intermediate distances, a straight line gives similarity falling
      monotonically to -1 at the ends;
    - tortuosity: path length of the class-mean trajectory divided by its end-to-end distance
      (1 for a straight line);
    - R^2 of predicting characters used from the top-k PCs of the means projected onto the points
      (k = 1, 2, 6), against a ridge probe on the full residual.
(b) Reader-head attention to the line-start newline (exp03's attention target 0) as a function of
    characters remaining: is it monotone in the count (a scalar read) or peaked at a head-specific
    distance from the edge (a Haiku-style boundary head)? Heads: L5H6, L5H7, L6H2, L8H15, L17H1,
    control L8H11. Peak location and the share of variance explained by a monotone (isotonic) fit.
(c) Added after the first run, because class means of ~20-200 points in 1,024 dimensions are noisy
    (tortuosity 11-28 and participation ratios in (a) are dominated by that noise): a split-half
    test. Points are split by paragraph parity into halves A and B. The top-6 principal directions
    of A's centred class means are fixed; B's class means are projected onto them. In each direction,
    B's projections are fitted by a straight line in characters used and by a natural cubic spline
    (knots at the 10/30/50/70/90% quantiles of the classes). Reported: the variance-weighted R^2 of
    each fit, the A-B reliability of each direction (correlation of projected means), and the same
    numbers with the used labels shuffled across points (seed 0). A straight line explaining nearly
    all reliable variance means a linear (scalar) code; a spline adding a lot means a curved one.
Run: python3 -I exp23_geometry.py   (from code/)
"""
import json, numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "outputs"
z = np.load(OUT / "exp03_pythia-410m.npz")
meta = z["meta"]; cols = list(z["cols"])
C = {c: meta[:, i] for i, c in enumerate(cols)}
R = np.load(OUT / "exp03_pythia-410m_resid_L0-12.npy", mmap_mode="r")  # [14, N, D]; index k = after layer k-1

def pca_stats(M):
    Mc = M - M.mean(0)
    s = np.linalg.svd(Mc, compute_uv=False) ** 2
    v = s / s.sum()
    pr = float(s.sum() ** 2 / (s ** 2).sum())
    return pr, [float(v[:k].sum()) for k in (1, 2, 3, 6)]

def ringing(M):
    Mc = M - M.mean(0)
    Mn = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
    S = Mn @ Mn.T
    n = len(S)
    prof = [float(np.mean([S[i, i + d] for i in range(n - d)])) for d in range(n)]
    # non-monotone profile: a local minimum followed by a rise of > 0.1
    mins = [d for d in range(1, n - 1) if prof[d] < prof[d - 1] and prof[d] < prof[d + 1]]
    rise = max([max(prof[d:]) - prof[d] for d in mins], default=0.0)
    return prof, float(rise)

def tortuosity(M):
    seg = np.linalg.norm(np.diff(M, axis=0), axis=1).sum()
    return float(seg / np.linalg.norm(M[-1] - M[0]))

def r2(X, y, lam=10.0, folds=5, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); out = []
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        mu = X[tr].mean(0); Xt = X[tr] - mu; ym = y[tr].mean()
        w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (y[tr] - ym))
        pred = (X[te] - mu) @ w + ym
        out.append(1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    return float(np.mean(out))

res = {"geometry": {}, "attention": {}}
for W in (40, 60):
    sel = np.where(C["W"] == W)[0]
    used = C["used"][sel].astype(int)
    classes = [u for u in range(W + 1) if (used == u).sum() >= 20]
    res["geometry"][W] = {"classes": [int(classes[0]), int(classes[-1]), len(classes)], "layers": {}}
    for L in range(3, 13):
        X = np.asarray(R[L + 1][sel], dtype=np.float32)
        M = np.stack([X[used == u].mean(0) for u in classes])
        pr, var = pca_stats(M)
        prof, rise = ringing(M)
        Mc = M - M.mean(0); _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
        keep = np.isin(used, classes)
        Xp = X[keep] - M.mean(0)
        r2k = {k: r2(Xp @ Vt[:k].T, used[keep].astype(float)) for k in (1, 2, 6)}
        Xr = Xp @ Vt[:min(len(classes) - 1, 64)].T
        res["geometry"][W]["layers"][L] = dict(participation_ratio=pr, var_top1_2_3_6=var, ringing_rise=rise,
            sim_profile=[round(x, 3) for x in prof[::4]], tortuosity=tortuosity(M), r2_topk=r2k,
            r2_mean_subspace=r2(Xr, used[keep].astype(float)))
    A = z["att"][sel][:, :, :, 0]  # attention to the previous (line-start) newline
    rem = C["remaining"][sel].astype(int); lab = C["label"][sel]
    pts = (lab == 0) | (lab == 1)
    for name, (l, h) in dict(L5H6=(5, 6), L5H7=(5, 7), L6H2=(6, 2), L8H15=(8, 15), L17H1=(17, 1), L8H11=(8, 11)).items():
        a = A[:, l, h]
        rr = [r for r in range(0, W) if (rem == r).sum() >= 15]
        prof = np.array([a[rem == r].mean() for r in rr])
        # monotone fit (pool-adjacent-violators, both directions), share of between-class variance it explains
        def pava(y):
            y = list(y); w = [1] * len(y); i = 0
            blocks = [[v, 1] for v in y]
            out = []
            for v in y:
                out.append([v, 1])
                while len(out) > 1 and out[-2][0] > out[-1][0]:
                    b = out.pop(); a2 = out.pop(); out.append([(a2[0] * a2[1] + b[0] * b[1]) / (a2[1] + b[1]), a2[1] + b[1]])
            return np.concatenate([[v] * n for v, n in out])
        fits = [pava(prof), -pava(-prof)]
        ss = ((prof - prof.mean()) ** 2).sum()
        mono = max(1 - ((prof - f) ** 2).sum() / ss for f in fits) if ss > 0 else 1.0
        res["attention"].setdefault(W, {})[name] = dict(remaining=[rr[0], rr[-1]], peak_at_remaining=int(rr[int(np.argmax(prof))]),
            min_at_remaining=int(rr[int(np.argmin(prof))]), profile=[round(float(x), 4) for x in prof],
            monotone_r2=float(mono), corr_with_remaining=float(np.corrcoef(rem, a)[0, 1]))
def spline_basis(x, knots):
    x = np.asarray(x, float); K = len(knots)
    d = lambda k: (np.clip(x - knots[k], 0, None) ** 3 - np.clip(x - knots[-1], 0, None) ** 3) / (knots[-1] - knots[k])
    cols = [np.ones_like(x), x] + [d(k) - d(K - 2) for k in range(K - 2)]
    return np.stack(cols, 1)
def fitr2(Y, X):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None); E = Y - X @ beta
    return 1 - (E ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
def split_half(X, used, para, classes, shuffle=False):
    u = used.copy()
    if shuffle: u = np.random.default_rng(0).permutation(u)
    A = para % 2 == 0; B = ~A
    ok = [c for c in classes if (u[A] == c).sum() >= 8 and (u[B] == c).sum() >= 8]
    MA = np.stack([X[A & (u == c)].mean(0) for c in ok]); MB = np.stack([X[B & (u == c)].mean(0) for c in ok])
    mu = MA.mean(0); _, _, Vt = np.linalg.svd(MA - mu, full_matrices=False); V = Vt[:6]
    YA = (MA - mu) @ V.T; YB = (MB - mu) @ V.T
    xs = np.array(ok, float); knots = np.quantile(xs, [0.1, 0.3, 0.5, 0.7, 0.9])
    lin = fitr2(YB, np.stack([np.ones_like(xs), xs], 1)); spl = fitr2(YB, spline_basis(xs, knots))
    w = YB.var(0); w = w / w.sum()
    rel = [float(np.corrcoef(YA[:, k], YB[:, k])[0, 1]) for k in range(6)]
    return dict(n_classes=len(ok), r2_linear=float((w * lin).sum()), r2_spline=float((w * spl).sum()),
                r2_linear_by_pc=[round(float(x), 3) for x in lin], r2_spline_by_pc=[round(float(x), 3) for x in spl],
                reliability_by_pc=[round(x, 3) for x in rel], var_share_B=[round(float(x), 3) for x in w])
res["split_half"] = {}
for W in (40, 60):
    sel = np.where(C["W"] == W)[0]; used = C["used"][sel].astype(int); para = C["para"][sel].astype(int)
    classes = [u for u in range(W + 1) if (used == u).sum() >= 20]
    for L in (4, 5, 6, 8, 10, 12):
        X = np.asarray(R[L + 1][sel], dtype=np.float32)
        res["split_half"].setdefault(W, {})[L] = dict(real=split_half(X, used, para, classes), shuffled=split_half(X, used, para, classes, True))
(OUT / "exp23_geometry.json").write_text(json.dumps(res, indent=1))
for W in (40, 60):
    print("W", W, res["geometry"][W]["classes"])
    for L, d in res["geometry"][W]["layers"].items():
        print(f" L{L}: PR {d['participation_ratio']:.2f} var1/2/3/6 {[round(x,2) for x in d['var_top1_2_3_6']]} ring {d['ringing_rise']:.2f} tort {d['tortuosity']:.2f} r2 top1/2/6 {[round(v,2) for v in d['r2_topk'].values()]} meanspace {d['r2_mean_subspace']:.2f}")
    for h, d in res["attention"][W].items():
        print(f" {h}: peak@rem {d['peak_at_remaining']} min@rem {d['min_at_remaining']} mono {d['monotone_r2']:.2f} r {d['corr_with_remaining']:.2f}")
for W in (40, 60):
    for L, d in res["split_half"][W].items():
        r, s = d["real"], d["shuffled"]
        print(f"W{W} L{L}: lin {r['r2_linear']:.2f} spline {r['r2_spline']:.2f} rel {r['reliability_by_pc']} varB {r['var_share_B']} | lin by pc {r['r2_linear_by_pc']} spl by pc {r['r2_spline_by_pc']} | shuffled lin {s['r2_linear']:.2f} spline {s['r2_spline']:.2f} rel {s['reliability_by_pc'][:3]}")
