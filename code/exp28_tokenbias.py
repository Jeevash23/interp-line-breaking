"""exp28 (EXPLORATORY, round 5, AMENDMENTS.md Amendment 2, items 2 and 4): where does the residual token
weight come from? Raw per-point data across widths and text types, for analysis on the CPU (exp28_analyse.py).
Text: WikiText-2 train at widths 20, 30, 40, 60, 80, 100, 120, and the exploration parts of the code and
German text at widths 40 and 60. Per (text, width): the first 60 paragraphs (seed 100) that wrap with >= 5
lines and tokenise to <= 700 tokens.
Per word-end decision point (common.word_end_positions; all points, line 3 on): width, characters used and
remaining, tokens on the line, label, last character a letter, newline log-odds intact, with L5H7
mean-ablated at every position, and with L5H7 and L4H14 mean-ablated (means over positions from 30 train
paragraphs, seed 6, width 60); attention of L5H6 and L5H7 to the line-start newline; projections of
L5H6's and L5H7's outputs and of the residual after layer 5 on the count direction (the mean residual
after layer 5 at points with >= 40 characters used minus that at <= 20 used, width 60, 20 train
paragraphs, seed 2, unit length).
Run: python3 exp28_tokenbias.py [--smoke]
"""
import sys, json, numpy as np, torch
from common import *
from newtext5 import load as LOAD

SMOKE = "--smoke" in sys.argv
NP = 4 if SMOKE else 60
tok, model = load("pythia-410m", attn="eager")
nl = newline_id(tok)
layers = model.gpt_neox.layers; Dm = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = Dm // Hn
Z = {}; R5 = {}; ABL = {}
def zh(l):
    def f(m, a):
        Z[l] = a[0][0].detach()
        if l in ABL:
            z = a[0].clone()
            for h, v in ABL[l]: z[:, :, h * dh:(h + 1) * dh] = v
            return (z,)
    return f
for l in (4, 5): layers[l].attention.dense.register_forward_pre_hook(zh(l))
layers[5].register_forward_hook(lambda m, a, o: R5.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[0].detach()))
WO5 = layers[5].attention.dense.weight.detach()

def wrapped(paras, W, n):
    out = []
    for p in paras:
        if len(out) >= n: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        text = "\n".join(lines); offs = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        for r in recs: r["alpha"] = int(text[offs[r["i"]][1] - 1].isalpha())
        out.append((ids, recs))
    return out

with torch.no_grad():
    zs = {4: 0, 5: 0}; nn = 0
    for ids, _ in wrapped(train_paragraphs(seed=6), 60, 6 if SMOKE else 30):
        model(torch.tensor([ids], device="cuda"))
        for l in (4, 5): zs[l] = zs[l] + Z[l].sum(0)
        nn += len(ids)
    ZM = {l: zs[l] / nn for l in (4, 5)}
    lo_, hi_ = 0, 0; nlo, nhi = 0, 0
    for ids, recs in wrapped(train_paragraphs(seed=2), 60, 6 if SMOKE else 20):
        model(torch.tensor([ids], device="cuda"))
        for r in recs:
            if r["used"] >= 40: hi_ = hi_ + R5["x"][r["i"]]; nhi += 1
            elif r["used"] <= 20: lo_ = lo_ + R5["x"][r["i"]]; nlo += 1
    wdir = hi_ / nhi - lo_ / nlo; wdir = wdir / wdir.norm()

def lo_of(ids, abl):
    ABL.clear()
    if "L5H7" in abl: ABL.setdefault(5, []).append((7, ZM[5][7 * dh:8 * dh]))
    if "L4H14" in abl: ABL.setdefault(4, []).append((14, ZM[4][14 * dh:15 * dh]))
    out = model(torch.tensor([ids], device="cuda"), output_attentions=not abl)
    ABL.clear()
    lp = torch.log_softmax(out.logits[0].float(), -1)[:, nl]
    return (lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-6))).cpu().numpy(), out

COLS = ["width", "used", "remaining", "ntok", "label", "alpha", "para", "lo", "lo_abl7", "lo_abl7_14", "att6", "att7", "p6", "p7", "p5res"]
data = {}
srcs = [("train", W) for W in (20, 30, 40, 60, 80, 100, 120)] + [(s, W) for s in ("explore-code", "explore-de") for W in (40, 60)]
with torch.no_grad():
    for split, W in srcs:
        its = wrapped(LOAD(split, seed=100), W, NP); rows = []
        for q, (ids, recs) in enumerate(its):
            lo, out = lo_of(ids, ()); A = out.attentions[5][0]
            nlpos = [i for i, t in enumerate(ids) if t == nl]
            o6 = Z[5][:, 6 * dh:7 * dh] @ WO5[:, 6 * dh:7 * dh].T; o7 = Z[5][:, 7 * dh:8 * dh] @ WO5[:, 7 * dh:8 * dh].T
            p6 = (o6 @ wdir).cpu().numpy(); p7 = (o7 @ wdir).cpu().numpy(); pr = (R5["x"] @ wdir).cpu().numpy()
            l7, _ = lo_of(ids, ("L5H7",)); l714, _ = lo_of(ids, ("L5H7", "L4H14"))
            for r in recs:
                i = r["i"]; ls = max([j for j in nlpos if j < i], default=None)
                if ls is None: continue
                rows.append([W, r["used"], r["remaining"], r["ntok_line"], r["label"], r["alpha"], q, lo[i], l7[i], l714[i],
                             float(A[6, i, ls]), float(A[7, i, ls]), p6[i], p7[i], pr[i]])
        data[f"{split}|{W}"] = np.array(rows, dtype=np.float64)
        print(split, W, len(its), "paragraphs", len(rows), "points", flush=True)
np.savez_compressed(OUT / ("exp28_tokenbias" + ("_smoke" if SMOKE else "") + ".npz"), cols=np.array(COLS), **{k.replace("|", "__"): v for k, v in data.items()})
print("saved")
