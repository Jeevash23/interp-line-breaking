"""exp08: does L5H6 count characters by attention dilution? Exploratory.

Idea: from any position, L5H6 attends to the previous newline and to the tokens of the
current line. If each line token's attention weight (relative to the newline's) grows
with its character length, the newline's share 1 / (1 + sum_t w_t) falls as characters
accumulate, and the head's output (a mix of the newline's value and the line tokens'
values) carries a character count.

(a) For each decision point and each current-line token t: w_t = a_t / a_newline.
    Fit log w_t ~ token character length (+ distance in tokens), report slopes and R^2.
(b) Does sum_t w_t track characters used better than tokens used? (R^2 by least squares)
(c) Same at the newline token itself: at a newline, does L5H6 attend to the newline
    before it and to the previous line's tokens the same way (so its output there
    carries the previous line's length)?
(d) Causal: probe 'characters used' (ridge, 5-fold) from the residual after layer 5,
    with and without L5H6 mean-ablated; and from L5H6's own output.
"""
import json, numpy as np, torch
from common import *

W = 60
tok, model = load("pythia-410m", attn="eager")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
L_, H_ = 5, 6

cap = {}
layers[L_].attention.dense.register_forward_pre_hook(lambda m, a: cap.__setitem__("z", a[0].detach()))
layers[L_].register_forward_hook(lambda m, a, o: cap.__setitem__("res5", (o[0] if isinstance(o, tuple) else o).detach()))
Wd = layers[L_].attention.dense.weight.detach().float()

rows_tok, rows_pt, rows_nl = [], [], []
Xres, Xhead, Yused = [], [], []
paras = [p for p in train_paragraphs(seed=8)]
n = 0
zsum = torch.zeros(D, device="cuda"); zc = 0
items = []
for p in paras:
    if n >= 40: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    ids, recs = word_end_positions(tok, lines, W)
    if len(ids) > 700: continue
    n += 1; items.append((lines, ids, recs))

for lines, ids, recs in items:
    text = "\n".join(lines)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); offs = enc["offset_mapping"]
    with torch.no_grad():
        out = model.gpt_neox(torch.tensor([ids], device="cuda"), output_attentions=True)
    A = out.attentions[L_][0, H_].float().cpu().numpy()  # (T, T)
    z = cap["z"][0]; zsum += z.sum(0); zc += z.shape[0]
    head_out = (z[:, H_ * dh:(H_ + 1) * dh].float() @ Wd[:, H_ * dh:(H_ + 1) * dh].T).cpu().numpy()
    res5 = cap["res5"][0].float().cpu().numpy()
    nls = [i for i, t in enumerate(ids) if t == nl]
    clen = [len(text[s:e]) for s, e in offs]
    for r in recs:
        i = r["i"]; pn = [x for x in nls if x < i][-1]
        a_nl = A[i, pn]
        ws = []
        for t in range(pn + 1, i + 1):
            w = A[i, t] / a_nl
            rows_tok.append((np.log(w + 1e-12), clen[t], i - t, clen[t] - (1 if text[offs[t][0]] == " " else 0)))
            ws.append(w)
        rows_pt.append((sum(ws), r["used"], r["ntok_line"], a_nl, r["remaining"]))
        Xres.append(res5[i]); Xhead.append(head_out[i]); Yused.append(r["used"])
    # (c) at each newline token (from line 2 on): attention to the newline before it, relative weights of the previous line's tokens
    for a_, b_ in zip(nls[:-1], nls[1:]):
        if a_ < 1: continue
        a_nl = A[b_, a_]
        s = sum(A[b_, t] / a_nl for t in range(a_ + 1, b_ + 1))
        rows_nl.append((s, len(text[offs[a_ + 1][0]:offs[b_][0]]), b_ - a_, a_nl))

T = np.array(rows_tok); P = np.array(rows_pt); N = np.array(rows_nl)
def r2(X, y):
    A_ = np.c_[X, np.ones(len(y))]; w, *_ = np.linalg.lstsq(A_, y, rcond=None); return 1 - ((y - A_ @ w) ** 2).sum() / ((y - y.mean()) ** 2).sum(), w
res = {}
r2_len, w_len = r2(T[:, [1]], T[:, 0]); r2_both, w_both = r2(T[:, [1, 2]], T[:, 0])
res["a_logw_vs_charlen"] = dict(r2=float(r2_len), slope_per_char=float(w_len[0]), r2_with_distance=float(r2_both), slope_char=float(w_both[0]), slope_dist=float(w_both[1]), n=len(T))
# mean log w by char length
res["a_mean_logw_by_len"] = {int(c): float(T[T[:, 1] == c, 0].mean()) for c in range(1, 13) if (T[:, 1] == c).sum() > 30}
res["b_sumw"] = dict(r2_used=float(r2(P[:, [0]], P[:, 1])[0]), r2_ntok=float(r2(P[:, [0]], P[:, 2])[0]),
                     r2_newline_share_vs_used=float(r2(P[:, [3]], P[:, 1])[0]), r2_newline_share_vs_remaining=float(r2(P[:, [3]], P[:, 4])[0]),
                     corr_sumw_used=float(np.corrcoef(P[:, 0], P[:, 1])[0, 1]), corr_sumw_ntok=float(np.corrcoef(P[:, 0], P[:, 2])[0, 1]))
# partial: does sum w add to used beyond ntok? fit used ~ ntok + sumw
res["b_partial"] = dict(r2_used_from_ntok=float(r2(P[:, [2]], P[:, 1])[0]), r2_used_from_ntok_and_sumw=float(r2(P[:, [2, 0]], P[:, 1])[0]),
                        r2_used_from_log_newline_share=float(r2(np.log(P[:, [3]]), P[:, 1])[0]))
res["c_at_newline"] = dict(n=len(N), corr_sumw_prevlinelen=float(np.corrcoef(N[:, 0], N[:, 1])[0, 1]), corr_sumw_ntok=float(np.corrcoef(N[:, 0], N[:, 2])[0, 1]),
                           mean_att_to_prev_newline=float(N[:, 3].mean()))

# (d) probes
def ridge_cv(X, y, lam=100.0, k=5):
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); folds = np.array_split(idx, k); ssr = sst = 0
    for f in folds:
        tr = np.setdiff1d(idx, f); mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
        A_ = (X[tr] - mu) / sd; B_ = (X[f] - mu) / sd
        w = np.linalg.solve(A_.T @ A_ + lam * np.eye(A_.shape[1]), A_.T @ (y[tr] - y[tr].mean()))
        ssr += ((y[f] - (B_ @ w + y[tr].mean())) ** 2).sum(); sst += ((y[f] - y[tr].mean()) ** 2).sum()
    return 1 - ssr / sst
Y = np.array(Yused, float)
res["d_probe_used"] = dict(res_after_L5=float(ridge_cv(np.array(Xres), Y)), L5H6_output=float(ridge_cv(np.array(Xhead), Y)))
# with L5H6 ablated
zmean = (zsum / zc)
def abl(m, a):
    z = a[0].clone(); z[..., H_ * dh:(H_ + 1) * dh] = zmean[H_ * dh:(H_ + 1) * dh]; return (z,)
h = layers[L_].attention.dense.register_forward_pre_hook(abl)
Xa = []
for lines, ids, recs in items:
    with torch.no_grad(): model.gpt_neox(torch.tensor([ids], device="cuda"))
    r5 = cap["res5"][0].float().cpu().numpy()
    for r in recs: Xa.append(r5[r["i"]])
h.remove()
res["d_probe_used"]["res_after_L5_with_L5H6_ablated"] = float(ridge_cv(np.array(Xa), Y))
save_json(res, "exp08_dilution.json")
print(json.dumps(res, indent=1))
