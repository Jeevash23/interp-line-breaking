"""conf01 (CONFIRMATORY, PROTOCOL.md H1): token correction by L5H7 in Pythia-410M.

Usage: conf01_token.py validation        (the frozen run)
       conf01_token.py train --smoke     (pipeline check only; 5 paragraphs per width)
Measure: coefficient of tokens-on-the-line in the newline log-odds at word-end decision points
(<= 30 characters remaining, line 3 on), given a linear spline in characters remaining with
knots 2, 5, 8, 12, 16, 24, 32. Head ablation: mean of the head's attention-output input over
positions, from 30 train paragraphs (seed 6), at all positions. 95% interval: 2,000 paragraph
bootstrap resamples, seed 0. Paragraphs: the first 60 per width, in seed-100 order, that wrap with
>= 5 lines and tokenise to <= 700 tokens.
"""
import sys, json, numpy as np, torch
from common import *

SPLIT = sys.argv[1]; SMOKE = "--smoke" in sys.argv
assert SPLIT in ("train", "validation")
NPARA = 5 if SMOKE else 60
NBOOT = 50 if SMOKE else 2000
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
cap = {}
h_ = layers[5].attention.dense.register_forward_pre_hook(lambda m, a: cap.__setitem__("z", a[0].detach()))
zs = []
with torch.no_grad():
    for p in train_paragraphs(seed=6)[:30]:
        model.gpt_neox(torch.tensor([tok(p, add_special_tokens=False)["input_ids"][:500]], device="cuda")); zs.append(cap["z"][0].mean(0))
h_.remove(); zmean = torch.stack(zs).mean(0)
ABL = {"h": None}
def abl(m, a):
    if ABL["h"] is None: return None
    h = ABL["h"]; z = a[0].clone(); z[..., h * dh:(h + 1) * dh] = zmean[h * dh:(h + 1) * dh]; return (z,)
layers[5].attention.dense.register_forward_pre_hook(abl)
CONDS = {"intact": None, "abl_L5H7": 7, "abl_L5H3": 3, "abl_L5H11": 11, "abl_L5H14": 14}
def spline(rem): return np.c_[rem, *[np.maximum(rem - k, 0) for k in [2, 5, 8, 12, 16, 24, 32]]]

res = {}
paras = split_paragraphs(SPLIT, seed=100)
for W in (40, 60, 80):
    items = []
    for p in paras:
        if len(items) >= NPARA: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        items.append((ids, recs))
    res[W] = {}
    for cname, h in CONDS.items():
        ABL["h"] = h
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
        br = np.random.default_rng(0); bs = []
        groups = [np.where(g == k)[0] for k in range(len(items))]
        for _ in range(NBOOT):
            ii = np.concatenate([groups[j] for j in br.integers(0, len(items), len(items))])
            bs.append(np.linalg.lstsq(M[ii], y[ii], rcond=None)[0][-1])
        res[W][cname] = dict(n_points=len(y), n_paragraphs=len(items), token_coef=float(w[-1]),
                             ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        print(W, cname, json.dumps(res[W][cname]), flush=True)
ABL["h"] = None
# verdicts (PROTOCOL.md H1)
v = {}
for W in res:
    i = res[W]["intact"]; a = res[W]["abl_L5H7"]
    v[W] = dict(H1a=bool(i["ci95"][0] > -0.10 and i["ci95"][1] < 0.10),
                H1b=bool(a["token_coef"] >= 0.30 and a["ci95"][0] > 0.20),
                H1c=bool(all(abs(res[W][c]["token_coef"] - i["token_coef"]) < 0.10 for c in ("abl_L5H3", "abl_L5H11", "abl_L5H14"))))
H1 = all(all(x.values()) for x in v.values())
out = dict(split=SPLIT, smoke=SMOKE, results=res, verdicts=v, H1_pass=H1)
save_json(out, f"conf01_token_{SPLIT}{'_smoke' if SMOKE else ''}.json")
print(json.dumps(dict(verdicts=v, H1_pass=H1), indent=1))
