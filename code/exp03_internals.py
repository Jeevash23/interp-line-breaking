"""exp03: first look inside Pythia-410M at word-end decision points. Exploratory.

(a) Probes: ridge regression from the residual stream after each layer to
    characters used, characters remaining and tokens used on the line (5-fold R^2).
(b) Direct logit attribution (DLA) of each head and MLP to the newline logit,
    averaged by characters remaining.
(c) Attention: for each head, the mass placed on the previous newline token, on the
    first token of the current line, on the rest of the current line, on the
    previous line, and on position 0.
"""
import sys, json, numpy as np, torch
from common import *

MNAME = sys.argv[1] if len(sys.argv) > 1 else "pythia-410m"
WIDTHS = [40, 60]
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 40

tok, model = load(MNAME, attn="eager")
nl = newline_id(tok)
cfg = model.config
L, H, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
dh = D // H
layers = model.gpt_neox.layers
Wu = model.embed_out.weight[nl].float() if hasattr(model, "embed_out") else model.lm_head.weight[nl].float()
lnf = model.gpt_neox.final_layer_norm
g = lnf.weight.float()

cache = {}
hooks = []
for l, layer in enumerate(layers):
    hooks.append(layer.attention.dense.register_forward_pre_hook(lambda m, a, l=l: cache.__setitem__(("z", l), a[0].detach())))
    hooks.append(layer.mlp.register_forward_hook(lambda m, a, o, l=l: cache.__setitem__(("mlp", l), o.detach())))
    hooks.append(layer.register_forward_hook(lambda m, a, o, l=l: cache.__setitem__(("res", l), (o[0] if isinstance(o, tuple) else o).detach())))
hooks.append(model.gpt_neox.embed_in.register_forward_hook(lambda m, a, o: cache.__setitem__(("res", -1), o.detach())))

paras = train_paragraphs()
meta, RES, DLA_H, DLA_M, ATT = [], [], [], [], []
Wd = [layer.attention.dense.weight.float() for layer in layers]  # (D, D): out = z @ W.T
for W in WIDTHS:
    used_p = 0
    for pi, p in enumerate(paras):
        if used_p >= NP: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        ids, recs = word_end_positions(tok, lines, W)
        if not recs or len(ids) > 1200: continue
        used_p += 1
        cache.clear()
        with torch.no_grad():
            out = model(torch.tensor([ids], device="cuda"), output_attentions=True)
        att = out.attentions  # tuple of (1, H, T, T)
        ids_t = np.array(ids)
        nl_pos = np.where(ids_t == nl)[0]
        idx = [r["i"] for r in recs]
        I = torch.tensor(idx, device="cuda")
        xf = cache[("res", L - 1)][0, I].float()
        mu = xf.mean(-1, keepdim=True); sd = xf.std(-1, unbiased=False, keepdim=True)
        def dla(c):  # c: (n, D)
            c = c.float(); c = c - c.mean(-1, keepdim=True)
            return ((c / sd) * g) @ Wu
        with torch.no_grad():
            res = torch.stack([cache[("res", l)][0, I] for l in range(-1, L)], 0).float().cpu().numpy()  # (L+1, n, D)
            dh_ = np.zeros((len(idx), L, H), np.float32); dm_ = np.zeros((len(idx), L), np.float32)
            at_ = np.zeros((len(idx), L, H, 5), np.float32)
            for l in range(L):
                z = cache[("z", l)][0, I].float()  # (n, D)
                for h in range(H):
                    ch = z[:, h * dh:(h + 1) * dh] @ Wd[l][:, h * dh:(h + 1) * dh].T
                    dh_[:, l, h] = dla(ch).cpu().numpy()
                dm_[:, l] = dla(cache[("mlp", l)][0, I]).cpu().numpy()
                A = att[l][0].float()  # (H, T, T)
                for k, r in enumerate(recs):
                    i = r["i"]
                    prev_nl = nl_pos[nl_pos < i]
                    if len(prev_nl) < 2: continue
                    a = A[:, i, : i + 1]
                    pn, ppn = prev_nl[-1], prev_nl[-2]
                    at_[k, l, :, 0] = a[:, pn].cpu().numpy()                      # previous newline
                    at_[k, l, :, 1] = a[:, pn + 1].cpu().numpy() if pn + 1 <= i else 0  # first token of line
                    at_[k, l, :, 2] = a[:, pn + 2: i + 1].sum(-1).cpu().numpy()   # rest of current line
                    at_[k, l, :, 3] = a[:, ppn + 1: pn].sum(-1).cpu().numpy()     # previous line
                    at_[k, l, :, 4] = a[:, 0].cpu().numpy()                       # position 0
        for r in recs:
            meta.append((W, pi, r["i"], r["used"], r["remaining"], r["label"], r["nextlen"], r["prevlen"], r["ntok_line"]))
        RES.append(res); DLA_H.append(dh_); DLA_M.append(dm_); ATT.append(at_)
        del out, att; cache.clear(); torch.cuda.empty_cache()

meta = np.array(meta, float)
RES = np.concatenate(RES, 1); DLA_H = np.concatenate(DLA_H); DLA_M = np.concatenate(DLA_M); ATT = np.concatenate(ATT)
np.savez_compressed(OUT / f"exp03_{MNAME}.npz", meta=meta, dla_h=DLA_H, dla_m=DLA_M, att=ATT,
                    cols=np.array("W,para,i,used,remaining,label,nextlen,prevlen,ntok_line".split(",")))
np.save(OUT / f"exp03_{MNAME}_resid_L0-12.npy", RES[:14].astype(np.float16))  # layers -1..12 only, to save disk
print("points", len(meta), "resid", RES.shape, flush=True)

# ---- probes
def ridge_cv(X, y, lam=10.0, k=5, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); folds = np.array_split(idx, k)
    ssr = sst = 0.0
    for f in folds:
        tr = np.setdiff1d(idx, f)
        mu = X[tr].mean(0); sdv = X[tr].std(0) + 1e-6
        A = (X[tr] - mu) / sdv; B = (X[f] - mu) / sdv
        w = np.linalg.solve(A.T @ A + lam * len(tr) * 0 + lam * np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean()))
        pred = B @ w + y[tr].mean()
        ssr += ((y[f] - pred) ** 2).sum(); sst += ((y[f] - y[tr].mean()) ** 2).sum()
    return 1 - ssr / sst

probe = {}
targets = {"used": meta[:, 3], "remaining": meta[:, 4], "ntok_line": meta[:, 8], "width": meta[:, 0]}
sub = np.random.default_rng(1).choice(len(meta), min(4000, len(meta)), replace=False)
for l in range(L + 1):
    X = RES[l, sub].astype(np.float32)
    probe[l] = {k: round(float(ridge_cv(X, v[sub], lam=100.0)), 3) for k, v in targets.items()}
    print("layer", l - 1, probe[l], flush=True)

# ---- DLA by remaining bins
rem = meta[:, 4]; lab = meta[:, 5]
near = rem <= 4; far = rem >= 20
dlah_diff = DLA_H[near].mean(0) - DLA_H[far].mean(0)
dlam_diff = DLA_M[near].mean(0) - DLA_M[far].mean(0)
top = np.dstack(np.unravel_index(np.argsort(-np.abs(dlah_diff).ravel())[:20], dlah_diff.shape))[0]
summary = dict(points=int(len(meta)), probe=probe,
               top_heads_dla_near_minus_far=[(f"L{l}H{h}", round(float(dlah_diff[l, h]), 3)) for l, h in top],
               mlp_dla_near_minus_far={f"L{l}": round(float(v), 3) for l, v in enumerate(dlam_diff)},
               total_logit_near_minus_far=round(float(DLA_H[near].sum((1, 2)).mean() + DLA_M[near].sum(1).mean() - DLA_H[far].sum((1, 2)).mean() - DLA_M[far].sum(1).mean()), 3))
# attention: heads with most mass on previous newline / line start
attm = ATT.mean(0)  # (L, H, 5)
names = ["prev_newline", "line_first_tok", "rest_of_line", "prev_line", "pos0"]
for j, nme in enumerate(names):
    t = np.dstack(np.unravel_index(np.argsort(-attm[:, :, j].ravel())[:12], attm.shape[:2]))[0]
    summary[f"top_att_{nme}"] = [(f"L{l}H{h}", round(float(attm[l, h, j]), 3)) for l, h in t]
save_json(summary, f"exp03_{MNAME}_summary.json")
print(json.dumps({k: v for k, v in summary.items() if k != "probe"}, indent=0))
