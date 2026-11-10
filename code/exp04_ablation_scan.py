"""exp04: mean-ablation scan of every attention head and every MLP in Pythia-410M.
Exploratory. Which components does the break decision depend on?

For each component, its output is replaced by its mean over all token positions of
the scan set (computed from clean runs), and we record, at word-end decision points:
  - separation S = mean newline logit at breaks minus at non-breaks with <= 12 chars left
  - AUROC of the newline log-prob (breaks vs non-breaks)
  - mean change of the newline log-prob at breaks and at near non-breaks.
"""
import sys, json, time, numpy as np, torch
from common import *

MNAME = "pythia-410m"
tok, model = load(MNAME)
nl = newline_id(tok)
cfg = model.config; L, H, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size; dh = D // H
layers = model.gpt_neox.layers

paras = train_paragraphs(seed=1)  # different shuffle from exp01/03
items = []
for W, n in [(40, 12), (60, 12)]:
    k = 0
    for p in paras:
        if k >= n: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 6: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        items.append((ids, recs)); k += 1

# clean run: means of z (per layer) and of mlp out (per layer)
zsum = [torch.zeros(D, device="cuda") for _ in range(L)]; msum = [torch.zeros(D, device="cuda") for _ in range(L)]; cnt = 0
cap = {}
hs = []
for l, layer in enumerate(layers):
    hs.append(layer.attention.dense.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(("z", l), a[0].detach())))
    hs.append(layer.mlp.register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("m", l), o.detach())))
with torch.no_grad():
    for ids, _ in items:
        model(torch.tensor([ids], device="cuda"))
        for l in range(L):
            zsum[l] += cap[("z", l)][0].sum(0); msum[l] += cap[("m", l)][0].sum(0)
        cnt += len(ids)
for h_ in hs: h_.remove()
zmean = [z / cnt for z in zsum]; mmean = [m / cnt for m in msum]

def measure(hook_fn=None, target=None):
    lps, labs, rems = [], [], []
    handle = None
    if target is not None:
        kind, l, h = target
        if kind == "head":
            def pre(m, a, h=h, l=l):
                z = a[0].clone(); z[..., h * dh:(h + 1) * dh] = zmean[l][h * dh:(h + 1) * dh]; return (z,)
            handle = layers[l].attention.dense.register_forward_pre_hook(pre)
        else:
            def fwd(m, a, o, l=l):
                return mmean[l].expand_as(o).clone()
            handle = layers[l].mlp.register_forward_hook(fwd)
    with torch.no_grad():
        for ids, recs in items:
            lp = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0].float(), -1)
            I = torch.tensor([r["i"] for r in recs], device="cuda")
            lps.append(lp[I, nl].cpu().numpy()); labs += [r["label"] for r in recs]; rems += [r["remaining"] for r in recs]
    if handle: handle.remove()
    return np.concatenate(lps), np.array(labs), np.array(rems)

def stats(lp, lab, rem, base=None):
    lg = lp - np.log1p(-np.exp(np.minimum(lp, -1e-6)))
    near0 = (lab == 0) & (rem <= 12)
    d = dict(S=float(lg[lab == 1].mean() - lg[near0].mean()), auroc=auroc(lp, lab),
             break_acc_proxy=float((lp[lab == 1] > np.log(0.5)).mean()))
    if base is not None:
        d["d_lp_break"] = float((lp - base)[lab == 1].mean()); d["d_lp_near_nonbreak"] = float((lp - base)[near0].mean())
    return d

t0 = time.time()
lp0, lab, rem = measure()
base = stats(lp0, lab, rem)
print("clean", base, "points", len(lab), "breaks", int(lab.sum()), flush=True)
res = {"clean": base, "heads": {}, "mlps": {}}
for l in range(L):
    lp, _, _ = measure(target=("mlp", l, None)); res["mlps"][f"L{l}"] = stats(lp, lab, rem, lp0)
    for h in range(H):
        lp, _, _ = measure(target=("head", l, h)); res["heads"][f"L{l}H{h}"] = stats(lp, lab, rem, lp0)
    print("layer", l, "done", round(time.time() - t0), "s", flush=True)
save_json(res, "exp04_ablation_scan.json")
hd = sorted(res["heads"].items(), key=lambda kv: kv[1]["S"])[:20]
md = sorted(res["mlps"].items(), key=lambda kv: kv[1]["S"])
print("clean S", round(base["S"], 3), "auroc", round(base["auroc"], 4))
print("heads with largest drop in S:")
for k, v in hd: print("  ", k, {a: round(b, 3) for a, b in v.items()})
print("MLPs by S:")
for k, v in md: print("  ", k, {a: round(b, 3) for a, b in v.items()})
