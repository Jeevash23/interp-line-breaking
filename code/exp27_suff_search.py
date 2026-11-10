"""exp27 (EXPLORATORY, round 5, AMENDMENTS.md Amendment 2, item 1): what else does the circuit need?
WikiText-2 train only. Ablation machinery copied from p4_02_sufficiency.py (mean ablation; EARLY = heads
of layers 1-5 outside the kept set at every position, layer-0 heads kept; DEC = heads and MLPs of layers
6-23 outside the kept set at decision points; BOTH = the two).
 1. Roles of the 96 heads of layers 0-5, from attention on 12 paragraphs (width 60, seed 3): mean share
    of attention from each position to itself, to the previous token, to the rest of the current line, to
    the line-start newline, to earlier lines and to the first token (a partition). A head's label is its
    largest share.
 2. Role classes: the round-4 circuit (BOTH scope) plus every early head with a given label, one class
    at a time and cumulatively in the order self, prev, local, line, earlier, first.
 3. Greedy search, early heads: from the circuit, add one at a time the head of layers 1-5 that most
    raises the separation kept (BOTH scope) on 20 search paragraphs (seed 25); 16 steps.
 4. Greedy search, late components: then, with the early additions kept, add one at a time the head or
    MLP of layers 6-23 that most raises the separation kept; candidates preselected each step as the 40
    with the largest first-order (attribution) gain; 10 steps.
 5. Report set (40 paragraphs, seed 27, and 80 width items, seed 14; disjoint from the search
    paragraphs): separation kept, AUROC and width effect kept along the greedy path; the same number of
    random early heads (and random late components) added instead, 5 draws each, at 4, 8, 12, 16 (and
    +5, +10 late); resample ablation for the final set.
Run: python3 exp27_suff_search.py [--smoke]
"""
import sys, json, random, numpy as np, torch
from common import *

SMOKE = "--smoke" in sys.argv
NS, NR, NIT = (4, 4, 6) if SMOKE else (20, 40, 80)
GE, GL, PRE = (2, 2, 6) if SMOKE else (16, 10, 40)
W = 60
tok, model = load("pythia-410m", attn="eager")
for p_ in model.parameters(): p_.requires_grad_(False)
nl = newline_id(tok)
layers = model.gpt_neox.layers; NL = len(layers)
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
EARLY_KEEP = {(2, 7), (2, 11), (2, 13), (4, 10), (4, 15), (5, 6), (5, 7), (4, 14)}
LATE_KEEP = {(6, 2), (8, 15), (17, 1), (6, 11), (15, 10), (19, 0), (20, 6)}
MLP_KEEP = {6, 14, 15, 16, 18, 21, 23}

ST = dict(head_abl={}, mlp_abl=set(), dec_mask=None, early_mask=None, mode="mean", donor_dec=None, donor_early=None, alpha=None)
MEAN_ALL, MEAN_DEC, CAPZ, CAPM = {}, {}, {}, {}
CAPTURE = dict(on=False)
def zhook(l):
    def f(m, a):
        z = a[0]
        if CAPTURE["on"]: CAPZ[l] = z[0].detach().clone()
        hs = ST["head_abl"].get(l)
        if not hs: return None
        early = l <= 5
        mask = ST["early_mask"] if early else ST["dec_mask"]
        if mask is None: return None
        z = z.clone()
        for h in hs:
            sl = slice(h * dh, (h + 1) * dh)
            if ST["mode"] == "mean":
                mv = (MEAN_ALL if early else MEAN_DEC)[("z", l)][sl]
                al = ST["alpha"].get(("h", l, h)) if ST["alpha"] is not None else None
                if al is None: z[:, mask, sl] = mv
                else: z[:, mask, sl] = mv + al * (z[:, mask, sl] - mv)
            else:
                src = ST["donor_early"] if early else ST["donor_dec"]
                z[:, mask, sl] = src[("z", l)][mask][:, sl] if early else src[("z", l)][:, sl]
        return (z,)
    return f
def mhook(l):
    def f(m, a, o):
        if CAPTURE["on"]: CAPM[l] = o[0].detach().clone()
        if l not in ST["mlp_abl"] or ST["dec_mask"] is None: return None
        o = o.clone(); mask = ST["dec_mask"]
        if ST["mode"] == "mean":
            mv = MEAN_DEC[("m", l)]
            al = ST["alpha"].get(("m", l, None)) if ST["alpha"] is not None else None
            o[:, mask] = mv if al is None else mv + al * (o[:, mask] - mv)
        else: o[:, mask] = ST["donor_dec"][("m", l)]
        return o
    return f
ATT = {}
for l in range(NL):
    layers[l].attention.dense.register_forward_pre_hook(zhook(l))
    layers[l].mlp.register_forward_hook(mhook(l))

def lp_seq(ids, grad=False):
    with torch.set_grad_enabled(grad):
        lp = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0].float(), -1)[:, nl]
        lo = lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-6))
    return lo if grad else lo.cpu().numpy()

def paras_ok(seed, n, minlines=5):
    out = []
    for p in train_paragraphs(seed=seed):
        if len(out) >= n: break
        lines = wrap(p, W)
        if lines is None or len(lines) < minlines: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        out.append((ids, recs))
    return out

# ---- 1. head roles (layers 0-5): a partition of each query's attention by key position
RN = ["self", "prev", "local", "line", "earlier", "first"]
roles = {}
with torch.no_grad():
    acc = np.zeros((6, Hn, len(RN))); cnt = 0
    for ids, recs in paras_ok(3, 3 if SMOKE else 12):
        out = model(torch.tensor([ids], device="cuda"), output_attentions=True)
        T = len(ids); lab_ = torch.full((T, T), -1, dtype=torch.long); lastnl = -1
        for i in range(T):
            for j in range(i + 1):
                if j == i: c = 0
                elif j == lastnl: c = 3
                elif j == i - 1: c = 1
                elif j == 0: c = 5
                elif j > lastnl: c = 2
                else: c = 4
                lab_[i, j] = c
            if ids[i] == nl: lastnl = i
        onehot = torch.stack([(lab_ == c).float() for c in range(len(RN))])  # [6, T, T]
        sel = torch.arange(1, T)
        for l in range(6):
            A = out.attentions[l][0].float().cpu()  # [H, T, T]
            for c in range(len(RN)): acc[l, :, c] += (A * onehot[c]).sum(-1)[:, sel].mean(-1).numpy()
        cnt += 1
    acc /= cnt
for l in range(6):
    for h in range(Hn):
        k = int(np.argmax(acc[l, h])); roles[f"L{l}H{h}"] = dict(label=RN[k], shares=[float(x) for x in acc[l, h]])
print("role counts (layers 1-5):", {r: sum(1 for k, v in roles.items() if v["label"] == r and not k.startswith("L0")) for r in RN}, flush=True)

# ---- means
CAPTURE["on"] = True
acc_all, acc_dec, nall, ndec = {}, {}, 0, 0; pool = []
with torch.no_grad():
    for ids, recs in paras_ok(6, 6 if SMOKE else 30):
        lp_seq(ids)
        di = torch.tensor([r["i"] for r in recs], device="cuda")
        for l in range(NL):
            for key, X in ((("z", l), CAPZ[l]), (("m", l), CAPM[l])):
                acc_all[key] = acc_all.get(key, 0) + X.sum(0); acc_dec[key] = acc_dec.get(key, 0) + X[di].sum(0)
        nall += len(ids); ndec += len(di)
        for r in recs[::3]:
            if len(pool) < 1500: pool.append({k: (CAPZ[k[1]] if k[0] == "z" else CAPM[k[1]])[r["i"]].clone() for k in acc_all})
for k in acc_all: MEAN_ALL[k] = acc_all[k] / nall; MEAN_DEC[k] = acc_dec[k] / ndec
donor_paras = paras_ok(7, 60)
CAPTURE["on"] = False
prng = random.Random(25)

def set_scope(keep_early, keep_late, keep_mlp):
    ST["head_abl"] = {}; ST["mlp_abl"] = set()
    for l in range(1, 6):
        for h in range(Hn):
            if (l, h) not in keep_early: ST["head_abl"].setdefault(l, []).append(h)
    for l in range(6, NL):
        for h in range(Hn):
            if (l, h) not in keep_late: ST["head_abl"].setdefault(l, []).append(h)
        if l not in keep_mlp: ST["mlp_abl"].add(l)

def prepare(ids, dec_positions, mode, didx, ablate=True):
    ST["mode"] = mode
    ST["dec_mask"] = torch.tensor(dec_positions, device="cuda", dtype=torch.long) if ablate else None
    ST["early_mask"] = torch.arange(len(ids), device="cuda") if ablate else None
    if mode == "resample" and ablate:
        d = pool[prng.randrange(len(pool))]
        ST["donor_dec"] = {k: v.unsqueeze(0).expand(len(dec_positions), -1) for k, v in d.items()}
        dids = donor_paras[didx % len(donor_paras)][0]
        CAPTURE["on"] = True; save = (ST["head_abl"], ST["mlp_abl"], ST["dec_mask"], ST["early_mask"])
        ST["head_abl"] = {}; ST["mlp_abl"] = set(); ST["dec_mask"] = ST["early_mask"] = None
        lp_seq(dids)
        ST["head_abl"], ST["mlp_abl"], ST["dec_mask"], ST["early_mask"] = save; CAPTURE["on"] = False
        n = len(ids); ST["donor_early"] = {}
        for l in range(1, 6):
            z = MEAN_ALL[("z", l)].unsqueeze(0).repeat(n, 1); m = min(n, CAPZ[l].shape[0]); z[:m] = CAPZ[l][:m]
            ST["donor_early"][("z", l)] = z

SEARCH = paras_ok(25, NS); REP = paras_ok(27, NR)
def sep_of(nat, keep, mode="mean", ablate=True):
    br, nb, sc, lab = [], [], [], []
    for q, (ids, recs) in enumerate(nat):
        set_scope(*keep); prepare(ids, [r["i"] for r in recs], mode, q, ablate)
        lo = lp_seq(ids)
        for r in recs:
            if r["label"] == 1: br.append(lo[r["i"]])
            elif r["remaining"] <= 12: nb.append(lo[r["i"]])
            sc.append(lo[r["i"]]); lab.append(r["label"])
    ST["dec_mask"] = ST["early_mask"] = None
    return float(np.mean(br) - np.mean(nb)), auroc(sc, lab)

def grad_gain(nat, keep, cands):
    """First-order gain in separation from restoring each candidate (alpha from 0 towards 1)."""
    ST["alpha"] = {c: torch.zeros((), device="cuda", requires_grad=True) for c in cands}
    tot = {c: 0.0 for c in cands}
    for q, (ids, recs) in enumerate(nat):
        set_scope(*keep); prepare(ids, [r["i"] for r in recs], "mean", q)
        lo = lp_seq(ids, grad=True)
        bi = [r["i"] for r in recs if r["label"] == 1]; ni = [r["i"] for r in recs if r["label"] == 0 and r["remaining"] <= 12]
        if not bi or not ni: continue
        s = lo[bi].mean() - lo[ni].mean()
        gs = torch.autograd.grad(s, [ST["alpha"][c] for c in cands], allow_unused=True)
        for c, g in zip(cands, gs): tot[c] += 0.0 if g is None else float(g)
    ST["alpha"] = None; ST["dec_mask"] = ST["early_mask"] = None
    return tot

# width items (exp14 construction, seed 14)
rng = random.Random(14); items = []
for p in train_paragraphs(seed=14):
    if len(items) >= NIT: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1)); words = lines[k].split(" ")
    js = [j for j in range(2, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    prefix = " ".join(words[: js[0] + 1]); before = lines[:k]; pw = before[-1].split(" "); short = before[-1]
    while len(pw) > 2 and len(before[-1]) - len(short) < 10: pw = pw[:-1]; short = " ".join(pw)
    if len(before[-1]) - len(short) < 10: continue
    items.append((tok("\n".join(before) + "\n" + prefix, add_special_tokens=False)["input_ids"],
                  tok("\n".join(before[:-1] + [short]) + "\n" + prefix, add_special_tokens=False)["input_ids"]))
def width_of(keep, mode="mean", ablate=True):
    e = []
    for q, (ib, is_) in enumerate(items):
        v = []
        for ids in (ib, is_):
            set_scope(*keep); prepare(ids, [len(ids) - 1], mode, q, ablate); v.append(lp_seq(ids)[-1])
        e.append(v[1] - v[0])
    ST["dec_mask"] = ST["early_mask"] = None
    return float(np.mean(e))

res = dict(roles=roles)
with torch.no_grad():
    s0 = sep_of(SEARCH, (set(), set(), set()), ablate=False)[0]
    r0 = sep_of(REP, (set(), set(), set()), ablate=False); w0 = width_of((set(), set(), set()), ablate=False)
res["intact"] = dict(search_sep=s0, report_sep=r0[0], report_auroc=r0[1], width=w0)
CIRC = (set(EARLY_KEEP), set(LATE_KEEP), set(MLP_KEEP))
def report(keep, mode="mean"):
    with torch.no_grad():
        s, a = sep_of(REP, keep, mode); w = width_of(keep, mode)
    return dict(sep_kept=s / r0[0], auroc=a, width_kept=w / w0)
res["circuit"] = report(CIRC); print("circuit", res["circuit"], flush=True)

# ---- 2. role classes
early_out = [(l, h) for l in range(1, 6) for h in range(Hn) if (l, h) not in EARLY_KEEP]
lab = {(l, h): roles[f"L{l}H{h}"]["label"] for l, h in early_out}
res["classes"] = {}; cum = set(EARLY_KEEP)
for r in RN:
    cls = {x for x in early_out if lab[x] == r}
    if not cls: continue
    one = report((EARLY_KEEP | cls, LATE_KEEP, MLP_KEEP)); cum |= cls
    allc = report((set(cum), LATE_KEEP, MLP_KEEP))
    res["classes"][r] = dict(n=len(cls), heads=[f"L{l}H{h}" for l, h in sorted(cls)], alone=one, cumulative=allc)
    print("class", r, len(cls), one["sep_kept"], allc["sep_kept"], flush=True)

# ---- 3. greedy early
keepE = set(EARLY_KEEP); pathE = []
with torch.no_grad():
    for step in range(GE):
        cands = [x for x in early_out if x not in keepE]
        sc = {x: sep_of(SEARCH, (keepE | {x}, LATE_KEEP, MLP_KEEP))[0] / s0 for x in cands}
        best = max(sc, key=sc.get); keepE.add(best)
        pathE.append(dict(add=f"L{best[0]}H{best[1]}", role=lab[best], search_kept=sc[best]))
        print("greedy early", step, pathE[-1], flush=True)
# ---- 4. greedy late (attribution preselection)
late_out = [("h", l, h) for l in range(6, NL) for h in range(Hn) if (l, h) not in LATE_KEEP] + [("m", l, None) for l in range(6, NL) if l not in MLP_KEEP]
keepL, keepM = set(LATE_KEEP), set(MLP_KEEP); pathL = []
for step in range(GL):
    cands = [c for c in late_out if not ((c[0] == "h" and (c[1], c[2]) in keepL) or (c[0] == "m" and c[1] in keepM))]
    g = grad_gain(SEARCH, (keepE, keepL, keepM), cands)
    pre = sorted(cands, key=lambda c: -g[c])[:PRE]
    with torch.no_grad():
        sc = {}
        for c in pre:
            kl, km = set(keepL), set(keepM)
            (kl.add((c[1], c[2])) if c[0] == "h" else km.add(c[1]))
            sc[c] = sep_of(SEARCH, (keepE, kl, km))[0] / s0
    best = max(sc, key=sc.get)
    (keepL.add((best[1], best[2])) if best[0] == "h" else keepM.add(best[1]))
    nm = f"L{best[1]}H{best[2]}" if best[0] == "h" else f"MLP{best[1]}"
    pathL.append(dict(add=nm, search_kept=sc[best], grad_rank=pre.index(best)))
    print("greedy late", step, pathL[-1], flush=True)

# ---- 5. report-set curve and random controls
curve = []
for k in [0, 2, 4, 8, 12, 16][: (3 if SMOKE else 6)]:
    ke = set(EARLY_KEEP) | {tuple(int(x) for x in p["add"][1:].split("H")) for p in pathE[:k]}
    curve.append(dict(early_added=k, late_added=0, **report((ke, LATE_KEEP, MLP_KEEP))))
    print("curve", curve[-1], flush=True)
def parse(n): return ("m", int(n[3:]), None) if n.startswith("MLP") else ("h",) + tuple(int(x) for x in n[1:].split("H"))
for k in [2, 5, 10][: (1 if SMOKE else 3)]:
    kl, km = set(LATE_KEEP), set(MLP_KEEP)
    for p in pathL[:k]:
        c = parse(p["add"]); (kl.add((c[1], c[2])) if c[0] == "h" else km.add(c[1]))
    curve.append(dict(early_added=len(pathE), late_added=k, **report((keepE, kl, km))))
    print("curve", curve[-1], flush=True)
rs = np.random.default_rng(0); rand = []
for k in [4, 8, 12, 16][: (1 if SMOKE else 4)]:
    for d in range(2 if SMOKE else 5):
        add = {early_out[i] for i in rs.choice(len(early_out), k, replace=False)}
        rand.append(dict(kind="early", k=k, **report((EARLY_KEEP | add, LATE_KEEP, MLP_KEEP))))
for k in [5, 10][: (1 if SMOKE else 2)]:
    for d in range(2 if SMOKE else 5):
        add = [late_out[i] for i in rs.choice(len(late_out), k, replace=False)]
        kl, km = set(LATE_KEEP), set(MLP_KEEP)
        for c in add: (kl.add((c[1], c[2])) if c[0] == "h" else km.add(c[1]))
        rand.append(dict(kind="late", k=k, **report((keepE, kl, km))))
print("random", [(r["kind"], r["k"], round(r["sep_kept"], 3)) for r in rand], flush=True)
res.update(greedy_early=pathE, greedy_late=pathL, curve=curve, random=rand,
           final_resample=report((keepE, keepL, keepM), "resample"),
           final_sets=dict(early=sorted(f"L{l}H{h}" for l, h in keepE), late=sorted(f"L{l}H{h}" for l, h in keepL), mlps=sorted(keepM)))
print("final resample", res["final_resample"], flush=True)
save_json(res, "exp27_suff_search" + ("_smoke" if SMOKE else "") + ".json")
