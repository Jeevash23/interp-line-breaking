"""exp02: which cues does the newline probability track? Exploratory, CPU only,
reads exp01 records.

(a) Hazard curve: mean P(newline) by characters remaining, against the empirical
    break rate at that remaining count.
(b) Characters or tokens: least-squares fits of the model's newline logit on
    candidate cue sets, compared by R^2 (5-fold, within width).
(c) Width or previous-line length: does the logit follow (prevlen - used) beyond
    (width - used)?
"""
import json, numpy as np
from common import OUT

def logit_from_logp(lp):
    lp = np.minimum(lp, -1e-6)
    return lp - np.log1p(-np.exp(lp))

def cv_r2(X, y, k=5, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); folds = np.array_split(idx, k)
    ss_res = ss_tot = 0.0
    for f in folds:
        tr = np.setdiff1d(idx, f)
        A = np.c_[X[tr], np.ones(len(tr))]; w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred = np.c_[X[f], np.ones(len(f))] @ w
        ss_res += ((y[f] - pred) ** 2).sum(); ss_tot += ((y[f] - y[tr].mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

def onehot_bins(v, edges):
    b = np.digitize(v, edges); M = np.zeros((len(v), len(edges) + 1)); M[np.arange(len(v)), b] = 1; return M[:, 1:]

out = {}
for m in ["pythia-410m", "pythia-1b", "gpt2"]:
    z = np.load(OUT / f"exp01_records_{m}.npz"); d = z["data"]; c = list(z["cols"])
    col = lambda n: d[:, c.index(n)]
    W, used, rem, lab, nxt, prev, ntok, lp = (col(n) for n in ["W", "used", "remaining", "label", "nextlen", "prevlen", "ntok_line", "logp_nl"])
    y = logit_from_logp(lp)
    res = {}
    for w in [40, 60, 80]:
        s = W == w
        # (a) hazard curve
        hz = {}
        for r in range(0, int(w) + 1, 2):
            t = s & (rem >= r) & (rem < r + 2)
            if t.sum() >= 20:
                hz[r] = dict(n=int(t.sum()), p_model=float(np.exp(lp[t]).mean()), p_true=float(lab[t].mean()))
        # (b) cue sets, restricted to the boundary zone (remaining <= 20) where the decision is hard
        z_ = s & (rem <= 20)
        R = rem[z_]; T = ntok[z_]; U = used[z_]; P = prev[z_]; Y = y[z_]
        edges = np.arange(1, 21)
        fits = {
            "chars_remaining(bins)": cv_r2(onehot_bins(R, edges), Y),
            "tokens_on_line(bins)": cv_r2(onehot_bins(T, np.arange(1, 30)), Y),
            "chars+tokens": cv_r2(np.c_[onehot_bins(R, edges), onehot_bins(T, np.arange(1, 30))], Y),
            "prevgap(bins)": cv_r2(onehot_bins(P - U, np.arange(-20, 21)), Y),
            "chars+prevgap": cv_r2(np.c_[onehot_bins(R, edges), onehot_bins(P - U, np.arange(-20, 21))], Y),
            "true_fit(nextlen+1>rem)": cv_r2(((nxt[z_] + 1) > R).astype(float)[:, None], Y),
            "chars+true_fit": cv_r2(np.c_[onehot_bins(R, edges), ((nxt[z_] + 1) > R).astype(float)], Y),
        }
        res[w] = dict(hazard=hz, r2=fits, n_zone=int(z_.sum()))
    out[m] = res
    print(m)
    for w, r in res.items():
        print("  W", w, "n", r["n_zone"], {k: round(v, 3) for k, v in r["r2"].items()})
(OUT / "exp02_cues.json").write_text(json.dumps(out, indent=1))
# print hazard for 410m W=60
for m in ["pythia-410m", "gpt2"]:
    print(m, "W=60 hazard (rem: p_model / p_true)")
    for r, v in out[m][60]["hazard"].items():
        if int(r) <= 24: print("   ", r, round(v["p_model"], 3), round(v["p_true"], 3), v["n"])
