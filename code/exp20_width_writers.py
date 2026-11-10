"""exp20: which early components write the previous line's length into the line-start newline
token? Exploratory (train split, new seed), after the confirmation run.

exp07/exp14/conf02: the width effect (previous line >= 10 characters shorter) reaches the
decision through the newline token's residual after layer 5, read by L8H15, L17H1 and L6H2.
Here: patch the output of ONE component (each head of layers 0..5 via its attention-output
input slice, each MLP 0..5) at the line-start newline position, from the short-previous-line
run into the base run; change in log P(newline) at the decision point. Also the whole
residual after each layer 0..5 at that position.
Expectation written before the run: L5H6 (which attends 0.41 to the newline before it and
counts by dilution) writes much of it; failure would be a diffuse or MLP-only pattern.
"""
import json, random, numpy as np, torch
from common import *

W = 60; LMAX = 6
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
COMPS = [("h", l, h) for l in range(LMAX) for h in range(Hn)] + [("m", l, None) for l in range(LMAX)] + [("r", l, None) for l in range(LMAX)]
cap = {}
for l in range(LMAX):
    layers[l].attention.dense.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(("z", l), a[0][0].detach().clone()))
    layers[l].mlp.register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("m", l), o[0].detach().clone()))
    layers[l].register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("r", l), (o[0] if isinstance(o, tuple) else o)[0].detach().clone()))
P = {"pos": None, "list": {}}
def zh(l):
    def f(m, a):
        lst = P["list"].get(("z", l))
        if not lst: return None
        z = a[0].clone()
        for row, h, vec in lst: z[row, P["pos"], h * dh:(h + 1) * dh] = vec
        return (z,)
    return f
def mh(l):
    def f(m, a, o):
        lst = P["list"].get(("m", l))
        if not lst: return None
        o = o.clone()
        for row, _, vec in lst: o[row, P["pos"]] = vec
        return o
    return f
def rh(l):
    def f(m, a, o):
        lst = P["list"].get(("r", l))
        if not lst: return None
        x = (o[0] if isinstance(o, tuple) else o).clone()
        for row, _, vec in lst: x[row, P["pos"]] = vec
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
    return f
for l in range(LMAX):
    layers[l].attention.dense.register_forward_pre_hook(zh(l)); layers[l].mlp.register_forward_hook(mh(l)); layers[l].register_forward_hook(rh(l))
def lp_of(ids, B=1):
    with torch.no_grad():
        hid = model.gpt_neox(torch.tensor([ids] * B, device="cuda")).last_hidden_state[:, -1]
        return torch.log_softmax(model.lm_head(hid).float(), -1)[:, nl].cpu().numpy()

rng = random.Random(20)
rows = []
for p in train_paragraphs(seed=22):
    if len(rows) >= 100: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1))
    words = lines[k].split(" ")
    js = [j for j in range(2, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    prefix = " ".join(words[: js[0] + 1]); before = lines[:k]
    pw = before[-1].split(" "); short = before[-1]
    while len(pw) > 2 and len(before[-1]) - len(short) < 10:
        pw = pw[:-1]; short = " ".join(pw)
    if len(before[-1]) - len(short) < 10: continue
    ib = tok("\n".join(before) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    is_ = tok("\n".join(before[:-1] + [short]) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    nb = [i for i, t in enumerate(ib) if t == nl][-1]; ns = [i for i, t in enumerate(is_) if t == nl][-1]
    if len(ib) - nb != len(is_) - ns: continue
    P["list"] = {}; lb = lp_of(ib)[0]
    ls = lp_of(is_)[0]; src = {k_: v[ns].clone() for k_, v in cap.items()}
    P["pos"] = nb
    out = np.zeros(len(COMPS))
    for s in range(0, len(COMPS), 40):
        part = COMPS[s:s + 40]; P["list"] = {}
        for row, (kind, l, h) in enumerate(part):
            if kind == "h": P["list"].setdefault(("z", l), []).append((row, h, src[("z", l)][h * dh:(h + 1) * dh]))
            else: P["list"].setdefault((kind, l), []).append((row, None, src[(kind, l)]))
        out[s:s + len(part)] = lp_of(ib, len(part)) - lb
    P["list"] = {}
    rows.append(dict(total=float(ls - lb), patch=out.tolist()))
    if len(rows) % 25 == 0: print(len(rows), "items", flush=True)
names = [f"L{l}H{h}" if k_ == "h" else (f"MLP{l}" if k_ == "m" else f"resid_after_L{l}") for k_, l, h in COMPS]
Pm = np.array([r["patch"] for r in rows]).mean(0); se = np.array([r["patch"] for r in rows]).std(0) / np.sqrt(len(rows))
comp_idx = [i for i, n in enumerate(names) if not n.startswith("resid")]
top = sorted(comp_idx, key=lambda i: -abs(Pm[i]))[:15]
summ = dict(n=len(rows), total=float(np.mean([r["total"] for r in rows])),
            resid={names[i]: float(Pm[i]) for i, n in enumerate(names) if n.startswith("resid")},
            top=[dict(comp=names[i], effect=float(Pm[i]), se=float(se[i])) for i in top])
save_json(dict(summary=summ, names=names, mean=Pm.tolist(), rows=rows), "exp20_width_writers.json")
print(json.dumps(summ, indent=1))
