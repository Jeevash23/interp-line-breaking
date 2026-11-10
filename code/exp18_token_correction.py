"""exp18: is L5H7 a token-count corrector? Exploratory test of a prediction.

exp16 (640 item-by-family pairs): at the decision token, the residual after layer 4 makes
the model over-count by about +0.17 log P(newline) per extra token from a same-length
re-spelling; after layer 5 the slope is about 0 (-0.006). L5H7's vote falls by about 0.20
per extra token, while L4H14 and L5H6 rise. Reading: L4H14 and L5H6 count characters with
a per-token bias, and L5H7 removes it.

Predictions (could fail):
 P1  With L5H7 mean-ablated (all positions), the behaviour under re-spellings becomes
     token-sensitive: slope of the change in log P(newline) on extra tokens rises from about 0
     to clearly positive (of the order of +0.1 to +0.2 per token). Control: mean-ablating
     L5H11 or L5H3 leaves the slope near 0.
 P2  On natural text (width 60), regressing the newline log-odds at decision points on
     characters remaining (spline) plus tokens used on the line: the token coefficient is
     about 0 in the intact model (as exp02) and becomes positive with L5H7 ablated, but not
     with the control heads ablated.
"""
import json, random, string, numpy as np, torch
from common import *

W = 60
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn

# mean z of layer 5 over positions (separate paragraphs)
cap = {}
hz = layers[5].attention.dense.register_forward_pre_hook(lambda m, a: cap.__setitem__("z", a[0].detach()))
zs = []
with torch.no_grad():
    for p in train_paragraphs(seed=6)[:30]:
        model.gpt_neox(torch.tensor([tok(p, add_special_tokens=False)["input_ids"][:500]], device="cuda")); zs.append(cap["z"][0].mean(0))
hz.remove()
zmean = torch.stack(zs).mean(0)
ABL = {"h": None}
def abl(m, a):
    if ABL["h"] is None: return None
    h = ABL["h"]; z = a[0].clone(); z[..., h * dh:(h + 1) * dh] = zmean[h * dh:(h + 1) * dh]; return (z,)
layers[5].attention.dense.register_forward_pre_hook(abl)
CONDS = {"intact": None, "abl_L5H7": 7, "abl_L5H11": 11, "abl_L5H3": 3, "abl_L5H6": 6}

def lp_last(ids):
    with torch.no_grad():
        hid = model.gpt_neox(torch.tensor([ids], device="cuda")).last_hidden_state[0, -1]
        return torch.log_softmax(model.lm_head(hid).float(), -1)[nl].item()

rng_l = random.Random(99)
FAM = {
    "dotted": lambda w: "".join(c if k % 2 == 0 else "." for k, c in enumerate(w)),
    "dashed": lambda w: "".join(c if k % 2 == 0 else "-" for k, c in enumerate(w)),
    "upper": lambda w: w.upper(),
    "camel": lambda w: "".join(c.upper() if k % 2 == 0 else c.lower() for k, c in enumerate(w)),
    "digits": lambda w: "".join(rng_l.choice("123456789") for _ in w),
    "scramble": lambda w: "".join(rng_l.choice(string.ascii_lowercase) for _ in w),
}
# P1
rng = random.Random(18)
items = []
for p in train_paragraphs(seed=19):
    if len(items) >= 100: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1))
    words = lines[k].split(" ")
    js = [j for j in range(3, len(words)) if 3 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    j = js[0]
    cand = [i for i in range(j) if len(words[i]) >= 5 and words[i].isalpha()]
    if len(cand) < 2: continue
    pick = rng.sample(cand, 2)
    before = "\n".join(lines[:k]) + "\n"
    base = before + " ".join(words[: j + 1])
    vers = {}
    for f, fn in FAM.items():
        ws = list(words[: j + 1])
        for i in pick: ws[i] = fn(ws[i])
        vers[f] = before + " ".join(ws)
    items.append((base, vers))
P1 = {}
for cname, h in CONDS.items():
    ABL["h"] = h
    pts = []
    for base, vers in items:
        ib = tok(base, add_special_tokens=False)["input_ids"]; lb = lp_last(ib)
        for f, t in vers.items():
            ir = tok(t, add_special_tokens=False)["input_ids"]
            pts.append((len(ir) - len(ib), lp_last(ir) - lb, f))
    a = np.array([(x, y) for x, y, _ in pts])
    slope, icpt = np.polyfit(a[:, 0], a[:, 1], 1)
    # bootstrap CI over items
    nI = len(items); nF = len(FAM); A3 = a.reshape(nI, nF, 2); br = np.random.default_rng(0); bs = []
    for _ in range(1000):
        ii = br.integers(0, nI, nI); b = A3[ii].reshape(-1, 2); bs.append(np.polyfit(b[:, 0], b[:, 1], 1)[0])
    P1[cname] = dict(slope=float(slope), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], intercept=float(icpt),
                     by_family={f: float(np.mean([y for x, y, ff in pts if ff == f])) for f in FAM})
    print("P1", cname, json.dumps(P1[cname]), flush=True)

# P2
nat = []
for p in train_paragraphs(seed=20):
    if len(nat) >= 40: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    ids, recs = word_end_positions(tok, lines, W)
    if len(ids) > 700: continue
    nat.append((ids, recs))
def spline(rem):
    knots = [2, 5, 8, 12, 16, 24, 32]
    return np.c_[rem, *[np.maximum(rem - k, 0) for k in knots]]
P2 = {}
for cname, h in CONDS.items():
    ABL["h"] = h
    X, y = [], []
    for ids, recs in nat:
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0].float(), -1)[:, nl].cpu().numpy()
        for r in recs:
            if r["remaining"] > 30: continue
            v = lp[r["i"]]; X.append((r["remaining"], r["ntok_line"])); y.append(v - np.log1p(-np.exp(min(v, -1e-6))))
    X = np.array(X, float); y = np.array(y)
    M = np.c_[np.ones(len(y)), spline(X[:, 0]), X[:, 1]]
    w, *_ = np.linalg.lstsq(M, y, rcond=None)
    br = np.random.default_rng(1); bs = []
    for _ in range(500):
        ii = br.integers(0, len(y), len(y)); bs.append(np.linalg.lstsq(M[ii], y[ii], rcond=None)[0][-1])
    P2[cname] = dict(n=len(y), token_coef=float(w[-1]), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    print("P2", cname, json.dumps(P2[cname]), flush=True)
ABL["h"] = None
save_json(dict(P1=P1, P2=P2), "exp18_token_correction.json")
