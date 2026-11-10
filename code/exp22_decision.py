"""exp22: where are the count and the width combined into the break decision? Exploratory
(round 4, AMENDMENTS.md Amendment 1, item 1). WikiText-2 train only.

Sources, as in exp14 (width 60, items built from train paragraphs, seed 14):
  C  : at the decision token, add the class-mean shift for +6 characters used to the residual
       after layer 5.
  Wd : set the line-start newline's residual after layer 5 to its value in a prompt whose previous
       line is >= 10 characters shorter.
  CW : both.
Recorded at the decision token for base, C, Wd, CW:
  - the residual after every layer 5..23 (change vectors per layer, cosine between the C and Wd
    changes: if the readers write the width onto the count's own axis, the two align);
  - each component's output (heads of layers 6..23 through their slice of the output projection,
    MLPs 6..23), its change under C and Wd, and its direct effect on log P(newline), linearised
    at the base run's final residual;
  - interaction: effect(CW) - effect(C) - effect(Wd).
Causal sets, frozen at the decision token to their base values (each source separately): blocks of
layers, all MLPs, all heads, named candidates (L17H1; MLP14+16+18; MLP23; readers), and 10 random
sets of three MLPs from layers 6..23 outside {14, 16, 18, 23}, drawn with seed 0.
Run: python3 exp22_decision.py [--smoke]
"""
import sys, json, random, numpy as np, torch
from common import *

SMOKE = "--smoke" in sys.argv
NITEMS = 6 if SMOKE else 100
W = 60; L0 = 6
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers; NL = len(layers)
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
COMPS = [("h", l, h) for l in range(L0, NL) for h in range(Hn)] + [("m", l, None) for l in range(L0, NL)]
names = [f"L{l}H{h}" if k == "h" else f"MLP{l}" for k, l, h in COMPS]

cap = {}
for l in range(L0, NL):
    layers[l].attention.dense.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(("z", l), a[0][:, -1].detach().clone()))
    layers[l].mlp.register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("m", l), o[:, -1].detach().clone()))
for l in range(5, NL):
    layers[l].register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("r", l), (o[0] if isinstance(o, tuple) else o)[:, -1].detach().clone()))
res5 = {}
layers[5].register_forward_hook(lambda m, a, o: res5.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().clone()))

PATCH = {}; EDIT = {}
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
WO = {l: layers[l].attention.dense.weight.detach() for l in range(L0, NL)}

def lp_of(ids, B=1):
    with torch.no_grad():
        hid = model.gpt_neox(torch.tensor([ids] * B, device="cuda")).last_hidden_state[:, -1]
        return torch.log_softmax(model.lm_head(hid).float(), -1)[:, nl].cpu().numpy()

def comp_outputs():
    """Per-component outputs at the decision token (row 0), in residual space."""
    out = []
    for k, l, h in COMPS:
        if k == "h": out.append(cap[("z", l)][0, h * dh:(h + 1) * dh] @ WO[l][:, h * dh:(h + 1) * dh].T)
        else: out.append(cap[("m", l)][0])
    return torch.stack(out)  # [n_comp, D]

def resid_stack(): return torch.stack([cap[("r", l)][0] for l in range(5, NL)])  # [19, D]

def grad_final():
    h = cap[("r", NL - 1)][0].clone().requires_grad_(True)
    lp = torch.log_softmax(model.lm_head(model.gpt_neox.final_layer_norm(h)).float(), -1)[nl]
    g, = torch.autograd.grad(lp, h)
    return g

def frozen_set_run(ids, src, sets):
    """One batch: row r freezes every component in sets[r] to its value in src."""
    PATCH.clear()
    for row, S in enumerate(sets):
        for (k, l, h) in S:
            if k == "h": PATCH.setdefault(("z", l), []).append((row, h, src[("z", l)][0, h * dh:(h + 1) * dh]))
            else: PATCH.setdefault(("m", l), []).append((row, None, src[("m", l)][0]))
    out = lp_of(ids, len(sets)); PATCH.clear()
    return out

def snapshot(): return {k: v.clone() for k, v in cap.items()}

H = lambda l, h: ("h", l, h); M = lambda l: ("m", l, None)
SETS = {
    "layers6-11": [c for c in COMPS if 6 <= c[1] <= 11],
    "layers12-17": [c for c in COMPS if 12 <= c[1] <= 17],
    "layers18-23": [c for c in COMPS if 18 <= c[1] <= 23],
    "all_mlps": [c for c in COMPS if c[0] == "m"],
    "all_heads": [c for c in COMPS if c[0] == "h"],
    "readers3": [H(8, 15), H(17, 1), H(6, 2)],
    "L17H1": [H(17, 1)],
    "MLP14_16_18": [M(14), M(16), M(18)],
    "MLP23": [M(23)],
    "MLP6_14_16": [M(6), M(14), M(16)],
    "late_mlps18-23": [M(l) for l in range(18, 24)],
    "mid_mlps12-17": [M(l) for l in range(12, 18)],
}
pool = [l for l in range(L0, NL) if l not in (14, 16, 18, 23)]
rs = np.random.default_rng(0)
for q in range(10): SETS[f"random_mlp3_{q}"] = [M(int(l)) for l in rs.choice(pool, 3, replace=False)]
SNAMES = list(SETS)

# class means of the residual after layer 5 by characters used (exp05/exp14 recipe)
sums, cnts = {}, {}
for p in train_paragraphs(seed=2):
    if sum(cnts.values()) > (1500 if SMOKE else 9000): break
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
rows = []; dR = {c: [] for c in ("C", "Wd", "CW")}; dO = {c: [] for c in ("C", "Wd")}; DLA = {c: [] for c in ("C", "Wd")}
for p in train_paragraphs(seed=14):
    if len(rows) >= NITEMS: break
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
    if len(ib) - nb != len(is_) - ns: continue
    EDIT.clear(); PATCH.clear()
    lb = lp_of(ib)[0]; cb = snapshot(); Rb = resid_stack(); Ob = comp_outputs(); g = grad_final()
    vec = means[u + 6] - means[u]
    lp_of(is_); vnl = res5["x"][0, ns].clone()
    out = dict(u=u, base=float(lb))
    for cond, lst in (("C", [(len(ib) - 1, vec, "add")]), ("Wd", [(nb, vnl, "set")]), ("CW", [(len(ib) - 1, vec, "add"), (nb, vnl, "set")])):
        EDIT["list"] = lst
        out[cond] = float(lp_of(ib)[0] - lb)
        dR[cond].append((resid_stack() - Rb).cpu())
        if cond in dO:
            dOc = comp_outputs() - Ob
            dO[cond].append(dOc.cpu()); DLA[cond].append((dOc @ g).cpu())
            sets = [SETS[s] for s in SNAMES]
            fr = np.concatenate([frozen_set_run(ib, cb, sets[i:i + 12]) for i in range(0, len(sets), 12)])
            out[f"frozen_{cond}"] = (fr - lb).tolist()
    EDIT.clear()
    rows.append(out)
    if len(rows) % 20 == 0: print(len(rows), "items", flush=True)

n = len(rows)
eff = {c: np.array([r[c] for r in rows]) for c in ("C", "Wd", "CW")}
inter = eff["CW"] - eff["C"] - eff["Wd"]
mdR = {c: torch.stack(v).mean(0) for c, v in dR.items()}   # [19, D]
cos_layer = [float(torch.nn.functional.cosine_similarity(mdR["C"][i], mdR["Wd"][i], dim=0)) for i in range(mdR["C"].shape[0])]
norm_layer = {c: [float(x) for x in mdR[c].norm(dim=-1)] for c in mdR}
mdO = {c: torch.stack(v).mean(0) for c, v in dO.items()}  # [n_comp, D]
mDLA = {c: torch.stack(v).mean(0).numpy() for c, v in DLA.items()}
cos_comp = torch.nn.functional.cosine_similarity(mdO["C"], mdO["Wd"], dim=-1).numpy()
def topdla(c, k=20):
    o = np.argsort(-np.abs(mDLA[c]))[:k]
    return [dict(comp=names[i], dla=float(mDLA[c][i]), other=float(mDLA["Wd" if c == "C" else "C"][i]), cos=float(cos_comp[i])) for i in o]
fro = {}
for c in ("C", "Wd"):
    F = np.array([r[f"frozen_{c}"] for r in rows]); E = eff[c].mean()
    fro[c] = {s: float(1 - F[:, q].mean() / E) for q, s in enumerate(SNAMES)}
    fro[c]["random_mlp3_mean"] = float(np.mean([fro[c][f"random_mlp3_{q}"] for q in range(10)]))
summ = dict(n=n, eff={c: float(v.mean()) for c, v in eff.items()}, interaction=float(inter.mean()),
            interaction_sd=float(inter.std()), cos_layer=dict(zip([f"after L{l}" for l in range(5, NL)], cos_layer)),
            norm_layer=norm_layer, dla_sum={c: float(mDLA[c].sum()) for c in mDLA},
            top_dla_C=topdla("C"), top_dla_Wd=topdla("Wd"), mediation_sets=fro)
save_json(dict(summary=summ, names=names, dla=dict((c, v.tolist()) for c, v in mDLA.items()), cos_comp=cos_comp.tolist(), rows=rows,
               sets={s: [names[COMPS.index(c)] for c in SETS[s]] if len(SETS[s]) < 30 else len(SETS[s]) for s in SNAMES}),
          "exp22_decision" + ("_smoke" if SMOKE else "") + ".json")
torch.save(dict(mdR=mdR, mdO=mdO), OUT / ("exp22_decision" + ("_smoke" if SMOKE else "") + "_means.pt"))
print(json.dumps(summ, indent=1))
