"""exp29 (EXPLORATORY, round 5, AMENDMENTS.md Amendment 2, item 3): what do MLPs 21 and 23 do?
WikiText-2 train only. Items and sources as exp22/p4_01 (width 60, seed 14, up to 100 items):
  C  : add the class-mean shift for +6 characters used to the residual after layer 5 at the decision token;
  Wd : set the line-start newline's residual after layer 5 to its value from the short-previous-line prompt.
(a) Route. For each source, the effect on log P(newline) at the decision token: intact; MLPs 21 and 23
    frozen to their base values at the decision token; the final LayerNorm's mean and scale frozen to
    their base values; both. If the damping acts through the final LayerNorm (the entropy-neuron route),
    freezing the LayerNorm should remove it.
(b) Neurons. The change of each MLP-21 and MLP-23 neuron's activation under each source, times its
    output weight's effect on the newline log-odds (gradient at the base run's final residual, with the
    LayerNorm free: total; with its statistics frozen: direct). Properties of every neuron: output-weight
    norm; share of the output weight in the unembedding's 24 weakest directions (null space; unembedding
    scaled by the final LayerNorm weight, centred over the vocabulary); cosine with the newline's centred
    unembedding direction.
(c) Generic or specific. On 30 natural paragraphs (width 60, seed 25), at every position: the intact
    model's top token and its log-probability, and the entropy; again with MLPs 21 and 23 mean-ablated at
    every position, and with only the 20 neurons that carry most of the damping mean-ablated. Generic
    confidence regulation predicts that removing them raises confidence for confident predictions of any
    token; a newline-specific role predicts it mainly for the newline.
Run: python3 exp29_damping.py [--smoke]
"""
import sys, json, random, numpy as np, torch
from common import *

SMOKE = "--smoke" in sys.argv
NITEMS, NNAT = (6, 4) if SMOKE else (100, 30)
W = 60; DL = (21, 23)
tok, model = load("pythia-410m")
for p_ in model.parameters(): p_.requires_grad_(False)
nl = newline_id(tok)
layers = model.gpt_neox.layers; NL = len(layers); D = model.config.hidden_size
FLN = model.gpt_neox.final_layer_norm
res5 = {}; EDIT = {}; MPATCH = {}; ACT = {}; ACTPATCH = {}; LNFIX = {}; MOUT = {}; LNIN = {}
def edit5(m, a, o):
    x = o[0] if isinstance(o, tuple) else o
    res5["x"] = x.detach()
    if not EDIT: return None
    x = x.clone()
    for pos, vec, mode in EDIT["list"]:
        if mode == "add": x[:, pos] += vec
        else: x[:, pos] = vec
    return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
layers[5].register_forward_hook(edit5)
def mhook(l):
    def f(m, a, o):
        MOUT[l] = o[:, -1].detach().clone()
        if l in MPATCH:
            o = o.clone(); o[:, -1] = MPATCH[l]; return o
    return f
def acthook(l):
    def f(m, a):
        x = a[0]; ACT[l] = x.detach().clone()
        if l in ACTPATCH:
            x = x.clone(); idx, val, pos = ACTPATCH[l]
            if pos is None: x[..., idx] = val
            else: x[:, pos][..., idx] = val
            return (x,)
    return f
for l in DL:
    layers[l].mlp.register_forward_hook(mhook(l))
    layers[l].mlp.dense_4h_to_h.register_forward_pre_hook(acthook(l))
def fln_pre(m, a):
    LNIN["x"] = a[0].detach().clone()
def fln_post(m, a, o):
    if not LNFIX: return None
    x = a[0]; mu, sd = LNFIX["mu"], LNFIX["sd"]
    o = o.clone(); o[:, -1] = (x[:, -1] - mu) / sd * m.weight + m.bias
    return o
FLN.register_forward_pre_hook(fln_pre); FLN.register_forward_hook(fln_post)

def lp_last(ids):
    with torch.no_grad():
        lg = model(torch.tensor([ids], device="cuda")).logits[0, -1].float()
    return float(torch.log_softmax(lg, -1)[nl])

WU = model.lm_head.weight.detach().float()  # [V, D]
gamma = FLN.weight.detach().float(); beta = FLN.bias.detach().float()
def grads(h):
    """d log P(newline) / d final residual: total (LayerNorm free) and direct (statistics frozen)."""
    h = h.clone().float().requires_grad_(True)
    lp = torch.log_softmax(FLN(h) @ WU.T, -1)[nl]; gt, = torch.autograd.grad(lp, h)
    mu, sd = h.detach().mean(), (h.detach().var(unbiased=False) + FLN.eps).sqrt()
    h2 = h.detach().clone().requires_grad_(True)
    lp2 = torch.log_softmax(((h2 - mu) / sd * gamma + beta) @ WU.T, -1)[nl]; gd, = torch.autograd.grad(lp2, h2)
    return gt, gd

# neuron properties
WUc = WU * gamma; WUc = WUc - WUc.mean(0, keepdim=True)
ev, evec = torch.linalg.eigh(WUc.T @ WUc)  # ascending
null = evec[:, :24]
unl = WUc[nl] / WUc[nl].norm()
props = {}
for l in DL:
    Wo = layers[l].mlp.dense_4h_to_h.weight.detach().float()  # [D, 4D]
    nrm = Wo.norm(dim=0); props[l] = dict(norm=nrm.cpu().numpy(), null=((null.T @ Wo).norm(dim=0) ** 2 / nrm ** 2).cpu().numpy(),
                                            cos_nl=((unl @ Wo) / nrm).cpu().numpy())

# class means after layer 5 (exp05/exp14 recipe)
sums, cnts = {}, {}
for p in train_paragraphs(seed=2):
    if sum(cnts.values()) > (1500 if SMOKE else 9000): break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    ids, recs = word_end_positions(tok, lines, W)
    if len(ids) > 700: continue
    with torch.no_grad(): model(torch.tensor([ids], device="cuda"))
    for r in recs:
        u = r["used"]; v = res5["x"][0, r["i"]]; sums[u] = sums.get(u, 0) + v; cnts[u] = cnts.get(u, 0) + 1
means = {u: s / cnts[u] for u, s in sums.items() if cnts[u] >= 8}

rng = random.Random(14); rows = []
contrib = {c: {l: {"tot": 0, "dir": 0} for l in DL} for c in ("C", "Wd")}
for p in train_paragraphs(seed=14):
    if len(rows) >= NITEMS: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1)); words = lines[k].split(" ")
    js = [j for j in range(2, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    prefix = " ".join(words[: js[0] + 1]); u = len(prefix)
    if u not in means or (u + 6) not in means: continue
    before = lines[:k]; pw = before[-1].split(" "); short = before[-1]
    while len(pw) > 2 and len(before[-1]) - len(short) < 10: pw = pw[:-1]; short = " ".join(pw)
    if len(before[-1]) - len(short) < 10: continue
    ib = tok("\n".join(before) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    is_ = tok("\n".join(before[:-1] + [short]) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    nb = [i for i, t in enumerate(ib) if t == nl][-1]; ns = [i for i, t in enumerate(is_) if t == nl][-1]
    if len(ib) - nb != len(is_) - ns: continue
    EDIT.clear(); MPATCH.clear(); LNFIX.clear()
    lb = lp_last(ib); mb = {l: MOUT[l].clone() for l in DL}; ab = {l: ACT[l][0, -1].clone() for l in DL}
    hb = LNIN["x"][0, -1].float().clone()
    mu_b = hb.mean(); sd_b = (hb.var(unbiased=False) + FLN.eps).sqrt()
    gt, gd = grads(hb)
    lp_of = lambda: lp_last(ib)
    with torch.no_grad(): lp_last(is_)
    vnl = res5["x"][0, ns].clone(); vec = means[u + 6] - means[u]
    out = dict(u=u, base=lb)
    for cond, lst in (("C", [(len(ib) - 1, vec, "add")]), ("Wd", [(nb, vnl, "set")])):
        EDIT["list"] = lst
        e = lp_of() - lb; ac = {l: ACT[l][0, -1].clone() for l in DL}
        for l in DL: MPATCH[l] = mb[l]
        eF = lp_of() - lb; MPATCH.clear()
        LNFIX.update(mu=mu_b, sd=sd_b); eLN = lp_of() - lb
        for l in DL: MPATCH[l] = mb[l]
        eFLN = lp_of() - lb; MPATCH.clear(); LNFIX.clear()
        out[cond] = dict(e=e, frozen_mlp=eF, frozen_ln=eLN, frozen_both=eFLN)
        for l in DL:
            Wo = layers[l].mlp.dense_4h_to_h.weight.detach().float(); da = (ac[l] - ab[l]).float()
            contrib[cond][l]["tot"] = contrib[cond][l]["tot"] + (da * (gt @ Wo)).cpu().numpy()
            contrib[cond][l]["dir"] = contrib[cond][l]["dir"] + (da * (gd @ Wo)).cpu().numpy()
        EDIT.clear()
    rows.append(out)
    if len(rows) % 20 == 0: print(len(rows), "items", flush=True)
n = len(rows)
def share(cond, key): E = np.mean([r[cond]["e"] for r in rows]); return float(1 - np.mean([r[cond][key] for r in rows]) / E)
route = {c: dict(effect=float(np.mean([r[c]["e"] for r in rows])), effect_ln_frozen=float(np.mean([r[c]["frozen_ln"] for r in rows])),
                 mlp_share=share(c, "frozen_mlp"),
                 mlp_share_ln_frozen=float(1 - np.mean([r[c]["frozen_both"] for r in rows]) / np.mean([r[c]["frozen_ln"] for r in rows])))
         for c in ("C", "Wd")}
print("route", json.dumps(route), flush=True)
neur = {}
allc = []
for l in DL:
    for c in ("C", "Wd"):
        for k in ("tot", "dir"): contrib[c][l][k] = contrib[c][l][k] / n
    t = contrib["C"][l]["tot"] + contrib["Wd"][l]["tot"]
    for i in range(len(t)): allc.append((float(t[i]), l, i))
allc.sort()  # most negative (damping) first
top = allc[:40]
topn = [dict(neuron=f"MLP{l}.{i}", contrib_C=float(contrib["C"][l]["tot"][i]), contrib_Wd=float(contrib["Wd"][l]["tot"][i]),
             direct_C=float(contrib["C"][l]["dir"][i]), direct_Wd=float(contrib["Wd"][l]["dir"][i]),
             norm=float(props[l]["norm"][i]), norm_pct=float((props[l]["norm"] < props[l]["norm"][i]).mean()),
             null=float(props[l]["null"][i]), cos_nl=float(props[l]["cos_nl"][i])) for _, l, i in top]
tot_neg = {c: float(sum(contrib[c][l]["tot"].sum() for l in DL)) for c in ("C", "Wd")}
dir_neg = {c: float(sum(contrib[c][l]["dir"].sum() for l in DL)) for c in ("C", "Wd")}
conc = {k: float(sum(x[0] for x in allc[:k]) / sum(x[0] for x in allc if x[0] < 0)) for k in (5, 10, 20, 50, 200)}
pop = {l: dict(norm_median=float(np.median(props[l]["norm"])), null_median=float(np.median(props[l]["null"])),
               null_p99=float(np.percentile(props[l]["null"], 99)), cos_nl_absmax=float(np.abs(props[l]["cos_nl"]).max())) for l in DL}
print("top neurons", json.dumps(topn[:10]), flush=True)

# (c) natural text, all positions
MEANACT = {}; MEANM = {}
accA = {l: 0 for l in DL}; accM = {l: 0 for l in DL}; nn_ = 0
hooksM = {}
def paras_ok(seed, nmax):
    out = []
    for p in train_paragraphs(seed=seed):
        if len(out) >= nmax: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        ids = tok("\n".join(lines), add_special_tokens=False)["input_ids"]
        if len(ids) > 700: continue
        out.append(ids)
    return out
FULLM = {}
def mfull(l):
    def f(m, a, o):
        FULLM[l] = o[0].detach()
        if ("abl", l) in FULLM: o = o.clone(); o[:] = FULLM[("abl", l)]; return o
    return f
hh = [layers[l].mlp.register_forward_hook(mfull(l)) for l in DL]
with torch.no_grad():
    for ids in paras_ok(6, 6 if SMOKE else 30):
        model(torch.tensor([ids], device="cuda"))
        for l in DL: accA[l] = accA[l] + ACT[l][0].sum(0); accM[l] = accM[l] + FULLM[l].sum(0)
        nn_ += len(ids)
for l in DL: MEANACT[l] = accA[l] / nn_; MEANM[l] = accM[l] / nn_
topidx = {l: [i for _, ll, i in top[:20] if ll == l] for l in DL}
recs = []
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
def run_cond(x, cond):
    MPATCH.clear(); ACTPATCH.clear()
    for k in [k for k in FULLM if isinstance(k, tuple)]: del FULLM[k]
    if cond == "mlp_abl":
        for l in DL: FULLM[("abl", l)] = MEANM[l]
    if cond == "neur_abl":
        for l in DL:
            if topidx[l]: ACTPATCH[l] = (torch.tensor(topidx[l], device="cuda"), MEANACT[l][topidx[l]], None)
    L = torch.log_softmax(model(x).logits[0].float(), -1)
    for k in [k for k in FULLM if isinstance(k, tuple)]: del FULLM[k]
    ACTPATCH.clear()
    return L
with torch.no_grad():
    for ids in paras_ok(25, NNAT):
        x = torch.tensor([ids], device="cuda")
        L0 = run_cond(x, "intact"); t = L0.argmax(-1); p0 = L0.gather(-1, t[:, None])[:, 0]; ent0 = -(L0.exp() * L0).sum(-1)
        for cond in ("mlp_abl", "neur_abl"):
            L1 = run_cond(x, cond); p1 = L1.gather(-1, t[:, None])[:, 0]; ent1 = -(L1.exp() * L1).sum(-1)
            for i in range(len(t)):
                recs.append((cond, int(t[i] == nl), float(p0[i].exp()), float(p1[i] - p0[i]), float(ent1[i] - ent0[i])))
for h in hh: h.remove()
summ_nat = {}
for cond in ("mlp_abl", "neur_abl"):
    for isnl in (1, 0):
        rr = [r for r in recs if r[0] == cond and r[1] == isnl]
        for a, b in zip(bins[:-1], bins[1:]):
            s = [r for r in rr if a <= r[2] < b]
            if s: summ_nat[f"{cond}|{'newline' if isnl else 'other'}|{a:.1f}-{b:.1f}"] = dict(n=len(s), dlogp=float(np.mean([r[3] for r in s])), dent=float(np.mean([r[4] for r in s])))
print("natural", json.dumps(summ_nat, indent=0), flush=True)
save_json(dict(n=n, route=route, total_contrib=tot_neg, direct_contrib=dir_neg, concentration=conc, top_neurons=topn, population=pop,
               natural=summ_nat, top20=[f"MLP{l}.{i}" for _, l, i in top[:20]], rows=rows), "exp29_damping" + ("_smoke" if SMOKE else "") + ".json")
