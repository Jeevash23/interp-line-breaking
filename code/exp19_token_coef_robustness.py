"""exp19: robustness of exp18's P2 result, plus one more prediction. Exploratory.

exp18 P2: on natural text (width 60), the coefficient of tokens-on-the-line in the newline
log-odds (given a spline in characters remaining) is 0.058 intact and 0.57 with L5H7
mean-ablated; controls L5H11 and L5H3 leave it unchanged. Its bootstrap resampled points,
not paragraphs.

Here: widths 40, 60, 80, new paragraphs, bootstrap over paragraphs. Conditions: intact,
L5H7 ablated, L5H11 ablated (control), L4H14 ablated, L4H14 and L5H7 both ablated.
Prediction written before the run (P3): if L4H14 is the source of the per-token bias that L5H7
removes, then (i) ablating L4H14 alone gives a token coefficient at or below the intact one
(L5H7 now over-corrects), and (ii) ablating both gives a coefficient well below the
L5H7-only value. Failure: the both-ablated coefficient is as large as L5H7-only.
"""
import json, numpy as np, torch
from common import *

tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
cap = {}
zmean = {}
for l in (4, 5):
    h_ = layers[l].attention.dense.register_forward_pre_hook(lambda m, a: cap.__setitem__("z", a[0].detach()))
    zs = []
    with torch.no_grad():
        for p in train_paragraphs(seed=6)[:30]:
            model.gpt_neox(torch.tensor([tok(p, add_special_tokens=False)["input_ids"][:500]], device="cuda")); zs.append(cap["z"][0].mean(0))
    h_.remove(); zmean[l] = torch.stack(zs).mean(0)
ABL = {"list": []}
def mk(l):
    def f(m, a):
        hs = [h for ll, h in ABL["list"] if ll == l]
        if not hs: return None
        z = a[0].clone()
        for h in hs: z[..., h * dh:(h + 1) * dh] = zmean[l][h * dh:(h + 1) * dh]
        return (z,)
    return f
for l in (4, 5): layers[l].attention.dense.register_forward_pre_hook(mk(l))
CONDS = {"intact": [], "abl_L5H7": [(5, 7)], "abl_L5H11": [(5, 11)], "abl_L4H14": [(4, 14)], "abl_L4H14_L5H7": [(4, 14), (5, 7)]}

def spline(rem):
    return np.c_[rem, *[np.maximum(rem - k, 0) for k in [2, 5, 8, 12, 16, 24, 32]]]
res = {}
for W in (40, 60, 80):
    items = []
    for p in train_paragraphs(seed=21):
        if len(items) >= 40: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        items.append((ids, recs))
    res[W] = {}
    for cname, lst in CONDS.items():
        ABL["list"] = lst
        X, y, g = [], [], []
        for k, (ids, recs) in enumerate(items):
            with torch.no_grad():
                lp = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0].float(), -1)[:, nl].cpu().numpy()
            for r in recs:
                if r["remaining"] > 30: continue
                v = lp[r["i"]]; X.append((r["remaining"], r["ntok_line"])); y.append(v - np.log1p(-np.exp(min(v, -1e-6)))); g.append(k)
        X = np.array(X, float); y = np.array(y); g = np.array(g)
        M = np.c_[np.ones(len(y)), spline(X[:, 0]), X[:, 1]]
        w = np.linalg.lstsq(M, y, rcond=None)[0]
        br = np.random.default_rng(2); bs = []
        groups = [np.where(g == k)[0] for k in range(len(items))]
        for _ in range(500):
            ii = np.concatenate([groups[j] for j in br.integers(0, len(items), len(items))])
            bs.append(np.linalg.lstsq(M[ii], y[ii], rcond=None)[0][-1])
        res[W][cname] = dict(n=len(y), token_coef=float(w[-1]), ci95_paragraph_bootstrap=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        print(W, cname, json.dumps(res[W][cname]), flush=True)
ABL["list"] = []
save_json(res, "exp19_token_coef_robustness.json")
