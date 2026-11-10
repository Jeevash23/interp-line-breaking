"""exp27b (EXPLORATORY, round 5, AMENDMENTS.md Amendment 2, item 1): a fuller circuit by fidelity.
exp27 added components by the separation they restore and overshot: with the rest mean-ablated, late
components pushed the separation far above the intact model's (1.68 of it) while AUROC fell. Here the target
is fidelity to the intact model: F = 1 - sum (lo_circuit - lo_intact)^2 / sum (lo_intact - mean lo_intact)^2
over all decision points (newline log-odds), so overshoot and shortfall both count against a set.
Same machinery and paragraphs as exp27 (BOTH scope; mean ablation; search set 20 paragraphs seed 25; report
set 40 paragraphs seed 27 and 80 width items seed 14). Greedy: from the round-4 circuit, at each step the
candidate (any head of layers 1-5 at every position, or any head or MLP of layers 6-23 at decision points)
that most raises F on the search set is added; candidates preselected as the 40 with the largest
first-order gain in F; 24 steps. Report set along the path (0, 2, 4, 8, 12, 16, 20, 24 added): F, separation
kept, AUROC, width effect kept, with mean and resample ablation; random additions of the same number and
layer group (early/late), 5 draws, at 4, 8, 12, 16, 24 (mean and resample).
Run: python3 exp27b_fidelity.py [--smoke]
"""
import sys, json, random, numpy as np, torch
from common import *

SMOKE = "--smoke" in sys.argv
NS, NR, NIT = (4, 4, 6) if SMOKE else (20, 40, 80)
GS, PRE = (3, 6) if SMOKE else (24, 40)
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
def los(nat, keep, mode="mean", ablate=True):
    out = []
    for q, (ids, recs) in enumerate(nat):
        set_scope(*keep); prepare(ids, [r["i"] for r in recs], mode, q, ablate)
        lo = lp_seq(ids); out.append(np.array([lo[r["i"]] for r in recs]))
    ST["dec_mask"] = ST["early_mask"] = None
    return out
with torch.no_grad():
    LO_S = los(SEARCH, (set(), set(), set()), ablate=False); LO_R = los(REP, (set(), set(), set()), ablate=False)
def fid(lo, ref):
    a = np.concatenate(lo); b = np.concatenate(ref); return float(1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum())
def sep_auc(nat, lo):
    br, nb, sc, lab = [], [], [], []
    for (ids, recs), l in zip(nat, lo):
        for r, v in zip(recs, l):
            if r["label"] == 1: br.append(v)
            elif r["remaining"] <= 12: nb.append(v)
            sc.append(v); lab.append(r["label"])
    return float(np.mean(br) - np.mean(nb)), auroc(sc, lab)
REFS = sep_auc(REP, LO_R)[0]
def grad_gain(nat, ref, keep, cands):
    ST["alpha"] = {c: torch.zeros((), device="cuda", requires_grad=True) for c in cands}
    tot = {c: 0.0 for c in cands}
    for q, (ids, recs) in enumerate(nat):
        set_scope(*keep); prepare(ids, [r["i"] for r in recs], "mean", q)
        lo = lp_seq(ids, grad=True)
        loss = -((lo[[r["i"] for r in recs]] - torch.tensor(ref[q], device="cuda")) ** 2).sum()
        gs = torch.autograd.grad(loss, [ST["alpha"][c] for c in cands], allow_unused=True)
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

def with_add(keep, c):
    ke, kl, km = set(keep[0]), set(keep[1]), set(keep[2])
    if c[0] == "e": ke.add((c[1], c[2]))
    elif c[0] == "h": kl.add((c[1], c[2]))
    else: km.add(c[1])
    return (ke, kl, km)
def cname(c): return f"L{c[1]}H{c[2]}" if c[0] in ("e", "h") else f"MLP{c[1]}"
with torch.no_grad(): w0 = width_of((set(), set(), set()), ablate=False)
def report(keep, mode="mean"):
    with torch.no_grad():
        lo = los(REP, keep, mode); s_, a_ = sep_auc(REP, lo); w = width_of(keep, mode)
    return dict(fid=fid(lo, LO_R), sep_kept=s_ / REFS, auroc=a_, width_kept=w / w0)
CIRC = (set(EARLY_KEEP), set(LATE_KEEP), set(MLP_KEEP))
res = dict(intact=dict(report_sep=REFS, width=w0))
with torch.no_grad(): f0 = fid(los(SEARCH, CIRC), LO_S)
res["circuit"] = dict(mean=report(CIRC), resample=report(CIRC, "resample"), search_fid=f0); print("circuit", res["circuit"], flush=True)
EARLY_C = [("e", l, h) for l in range(1, 6) for h in range(Hn) if (l, h) not in EARLY_KEEP]
LATE_C = [("h", l, h) for l in range(6, NL) for h in range(Hn) if (l, h) not in LATE_KEEP] + [("m", l, None) for l in range(6, NL) if l not in MLP_KEEP]
keep = CIRC; path = []
for step in range(GS):
    cands = [c for c in EARLY_C + LATE_C if cname(c) not in {p["add"] for p in path}]
    # gradient preselection: early candidates enter the alpha dict under the hook key ("h", l, h)
    key = {c: (("h", c[1], c[2]) if c[0] in ("e", "h") else ("m", c[1], None)) for c in cands}
    g = grad_gain(SEARCH, LO_S, keep, list(key.values()))
    pre = sorted(cands, key=lambda c: -g[key[c]])[:PRE]
    with torch.no_grad():
        sc = {c: fid(los(SEARCH, with_add(keep, c)), LO_S) for c in pre}
    best = max(sc, key=sc.get); keep = with_add(keep, best)
    path.append(dict(add=cname(best), group="early" if best[0] == "e" else "late", search_fid=sc[best], grad_rank=pre.index(best)))
    print("greedy", step, path[-1], flush=True)
def keep_at(k):
    kk = CIRC
    for p in path[:k]:
        n_ = p["add"]
        if n_.startswith("MLP"): kk = with_add(kk, ("m", int(n_[3:]), None))
        else:
            l, h = (int(x) for x in n_[1:].split("H")); kk = with_add(kk, ("e" if l <= 5 else "h", l, h))
    return kk
curve = []
for k in [2, 4, 8, 12, 16, 20, 24][: (2 if SMOKE else 7)]:
    kk = keep_at(k); curve.append(dict(added=k, n_early=sum(p["group"] == "early" for p in path[:k]), mean=report(kk), resample=report(kk, "resample")))
    print("curve", curve[-1], flush=True)
rs = np.random.default_rng(0); rand = []
for k in [4, 8, 12, 16, 24][: (1 if SMOKE else 5)]:
    ne = sum(p["group"] == "early" for p in path[:k]); nlt = k - ne
    for d in range(2 if SMOKE else 5):
        kk = CIRC
        for i in rs.choice(len(EARLY_C), ne, replace=False): kk = with_add(kk, EARLY_C[i])
        for i in rs.choice(len(LATE_C), nlt, replace=False): kk = with_add(kk, LATE_C[i])
        rand.append(dict(added=k, mean=report(kk), resample=report(kk, "resample")))
    print("random", k, [(round(r["mean"]["fid"], 3), round(r["mean"]["sep_kept"], 3)) for r in rand if r["added"] == k], flush=True)
res.update(path=path, curve=curve, random=rand)
save_json(res, "exp27b_fidelity" + ("_smoke" if SMOKE else "") + ".json")
