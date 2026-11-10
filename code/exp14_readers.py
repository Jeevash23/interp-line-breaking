"""exp14: who reads the layer-5 count, and who brings in the width? Exploratory.

Two sources of a change in log P(newline), both introduced after layer 5:
  COUNT : at the decision token, add the class-mean shift for d = +6 characters used to
          the residual after layer 5 (exp05 method; exp05 gave about +2.1).
  WIDTH : patch the residual of the line-start newline token after layer 5 with its state
          from a prompt whose previous line is >= 10 characters shorter (exp07 gave +0.69).
For each source, freeze ONE downstream component at the decision token (each head of
layers 6..23 via its slice of the attention-output input, each MLP of layers 6..23) to its
value in the unedited run, and measure how much of the source's effect remains.
Mediation share = 1 - (effect with the component frozen) / (effect).

Readers common to both sources would be where the count is compared with the width.
"""
import json, random, numpy as np, torch
from common import *

W = 60
L0 = 6
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
NL = len(layers)
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
COMPS = [("h", l, h) for l in range(L0, NL) for h in range(Hn)] + [("m", l, None) for l in range(L0, NL)]

cap = {}
for l in range(L0, NL):
    layers[l].attention.dense.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(("z", l), a[0][:, -1].detach().clone()))
    layers[l].mlp.register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("m", l), o[:, -1].detach().clone()))
res5 = {}
layers[5].register_forward_hook(lambda m, a, o: res5.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().clone()))

PATCH = {}
EDIT = {}  # applied to the residual after layer 5: list of (pos, vec) for all rows
def zhook(l):
    def f(m, a):
        lst = PATCH.get(("z", l))
        if not lst: return None
        z = a[0].clone()
        for row, h, vec in lst: z[row, -1, h * dh:(h + 1) * dh] = vec
        return (z,)
    return f
def mhook(l):
    def f(m, a, o):
        lst = PATCH.get(("m", l))
        if not lst: return None
        o = o.clone()
        for row, _, vec in lst: o[row, -1] = vec
        return o
    return f
def edit5(m, a, o):
    if not EDIT: return None
    x = o[0] if isinstance(o, tuple) else o
    x = x.clone()
    for pos, vec, mode in EDIT["list"]:
        if mode == "add": x[:, pos] += vec
        else: x[:, pos] = vec
    return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
for l in range(L0, NL):
    layers[l].attention.dense.register_forward_pre_hook(zhook(l))
    layers[l].mlp.register_forward_hook(mhook(l))
layers[5].register_forward_hook(edit5)

def lp_of(ids, B=1):
    with torch.no_grad():
        hid = model.gpt_neox(torch.tensor([ids] * B, device="cuda")).last_hidden_state[:, -1]
        return torch.log_softmax(model.lm_head(hid).float(), -1)[:, nl].cpu().numpy()

def frozen_runs(ids, src, chunk=64):
    out = np.zeros(len(COMPS))
    for s in range(0, len(COMPS), chunk):
        part = COMPS[s:s + chunk]
        PATCH.clear()
        for row, (kind, l, h) in enumerate(part):
            if kind == "h": PATCH.setdefault(("z", l), []).append((row, h, src[("z", l)][h * dh:(h + 1) * dh]))
            else: PATCH.setdefault(("m", l), []).append((row, None, src[("m", l)]))
        out[s:s + len(part)] = lp_of(ids, len(part))
    PATCH.clear()
    return out

def snapshot(): return {k: v[0].clone() for k, v in cap.items()}

# class means of the residual after layer 5 by characters used (exp05 recipe, same seed)
sums, cnts = {}, {}
for p in train_paragraphs(seed=2):
    if sum(cnts.values()) > 9000: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    ids, recs = word_end_positions(tok, lines, W)
    if len(ids) > 700: continue
    lp_of(ids)
    for r in recs:
        u = r["used"]; v = res5["x"][0, r["i"]]
        sums[u] = sums.get(u, 0) + v; cnts[u] = cnts.get(u, 0) + 1
means = {u: s / cnts[u] for u, s in sums.items() if cnts[u] >= 8}

rng = random.Random(14)
rows = []
for p in train_paragraphs(seed=14):
    if len(rows) >= 100: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1))
    words = lines[k].split(" ")
    js = [j for j in range(2, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    j = js[0]
    prefix = " ".join(words[: j + 1]); u = len(prefix)
    if u not in means or (u + 6) not in means: continue
    before = lines[:k]
    pw = before[-1].split(" "); short = before[-1]
    while len(pw) > 2 and len(before[-1]) - len(short) < 10:
        pw = pw[:-1]; short = " ".join(pw)
    if len(before[-1]) - len(short) < 10: continue
    ib = tok("\n".join(before) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    is_ = tok("\n".join(before[:-1] + [short]) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    nb = [i for i, t in enumerate(ib) if t == nl][-1]; ns = [i for i, t in enumerate(is_) if t == nl][-1]
    if len(ib) - nb != len(is_) - ns: continue  # current line must tokenise identically
    # clean base
    EDIT.clear(); PATCH.clear()
    lb = lp_of(ib)[0]; cb = snapshot()
    # COUNT source
    vec = means[u + 6] - means[u]
    EDIT["list"] = [(len(ib) - 1, vec, "add")]
    lc = lp_of(ib)[0]
    fc = frozen_runs(ib, cb)
    # WIDTH source: newline state after layer 5 from the short run
    EDIT.clear(); lp_of(is_); vnl = res5["x"][0, ns].clone()
    EDIT["list"] = [(nb, vnl, "set")]
    lw = lp_of(ib)[0]
    fw = frozen_runs(ib, cb)
    EDIT.clear()
    rows.append(dict(u=u, eff_count=float(lc - lb), eff_width=float(lw - lb),
                     frozen_count=(fc - lb).tolist(), frozen_width=(fw - lb).tolist()))
    if len(rows) % 20 == 0: print(len(rows), "items", flush=True)

names = [f"L{l}H{h}" if kind == "h" else f"MLP{l}" for kind, l, h in COMPS]
EC = np.array([r["eff_count"] for r in rows]); EW = np.array([r["eff_width"] for r in rows])
FC = np.array([r["frozen_count"] for r in rows]); FW = np.array([r["frozen_width"] for r in rows])
med_c = 1 - FC.mean(0) / EC.mean(); med_w = 1 - FW.mean(0) / EW.mean()
def top(med, n=25):
    o = np.argsort(-med)
    return [dict(comp=names[i], share=float(med[i])) for i in o[:n]]
summ = dict(n=len(rows), eff_count=float(EC.mean()), eff_width=float(EW.mean()),
            top_count=top(med_c), top_width=top(med_w),
            corr_shares=float(np.corrcoef(med_c, med_w)[0, 1]))
save_json(dict(summary=summ, names=names, med_count=med_c.tolist(), med_width=med_w.tolist(), rows=rows), "exp14_readers.json")
print(json.dumps(summ, indent=1))
