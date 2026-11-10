"""p5_01 (round 5, CONFIRMATORY under PROTOCOL-5.md when run on reserve-pythia410m): sufficiency of the fuller
circuit found by exp27b (EXPLORATORY). Machinery copied from exp27b_fidelity.py: BOTH scope (heads of layers
1-5 outside the kept set ablated at every position, layer-0 heads kept; heads and MLPs of layers 6-23 outside
the kept set ablated at decision points); mean ablation (means over 30 train paragraphs, seed 6) and resample
ablation (donors from train, seeds 7 and 25), as in rounds 4 and exploration.
Sets: the round-4 circuit; the round-4 circuit plus the first 4, 8, 16 and 24 components of exp27b's path
(fixed below); 5 random sets (numpy seed 0) of the round-4 circuit plus 24 components with the path's
composition (13 heads of layers 1-5, 11 heads or MLPs of layers 6-23), drawn from the same candidates.
Material: 40 paragraphs (seed 27, width 60) and up to 80 width items (seed 14) from the named split.
Saved per paragraph (newline log-odds at every decision point) and per item (width effect), for each set and
ablation, so report5_tables.py can bootstrap fidelity, separation kept, AUROC and width effect kept.
Run: python3 p5_01_sufficiency.py <train|reserve-pythia410m> [--smoke]
"""
import sys, json, random, numpy as np, torch
from common import *
from newtext5 import load as LOAD
SPLIT = sys.argv[1]; assert SPLIT in ("train", "reserve-pythia410m")
PATH = ["MLP9", "L4H8", "L3H4", "MLP13", "L14H8", "L1H13", "L20H12", "L4H5", "L1H14", "MLP19", "L1H12", "L1H9", "L21H2", "L4H9", "L2H5", "L9H10", "L3H12", "L2H8", "L3H11", "L13H12", "L7H15", "L11H1", "L21H12", "L1H1"]

SMOKE = "--smoke" in sys.argv
NS, NR, NIT = (4, 4, 6) if SMOKE else (20, 40, 80)
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

def paras_ok(seed, n, minlines=5, split="train"):
    out = []
    for p in (train_paragraphs(seed=seed) if split == "train" else LOAD(split, seed)):
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

REP = paras_ok(27, NR, split=SPLIT)
def los(nat, keep, mode="mean", ablate=True):
    out = []
    for q, (ids, recs) in enumerate(nat):
        set_scope(*keep); prepare(ids, [r["i"] for r in recs], mode, q, ablate)
        lo = lp_seq(ids); out.append(np.array([lo[r["i"]] for r in recs]))
    ST["dec_mask"] = ST["early_mask"] = None
    return out
with torch.no_grad():
    LO_R = los(REP, (set(), set(), set()), ablate=False)
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
for p in (train_paragraphs(seed=14) if SPLIT == "train" else LOAD(SPLIT, 14)):
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
    return np.array(e, dtype=float)

def with_add(keep, c):
    ke, kl, km = set(keep[0]), set(keep[1]), set(keep[2])
    if c[0] == "e": ke.add((c[1], c[2]))
    elif c[0] == "h": kl.add((c[1], c[2]))
    else: km.add(c[1])
    return (ke, kl, km)
def cname(c): return f"L{c[1]}H{c[2]}" if c[0] in ("e", "h") else f"MLP{c[1]}"
with torch.no_grad(): W0 = width_of((set(), set(), set()), ablate=False)
RAW = dict(intact=dict(lo=[x.tolist() for x in LO_R], width=W0.tolist()))
def run(name, keep, mode):
    with torch.no_grad():
        lo = los(REP, keep, mode); w = width_of(keep, mode)
    s_, a_ = sep_auc(REP, lo); REFS_ = sep_auc(REP, LO_R)[0]
    RAW[f"{name}|{mode}"] = dict(lo=[x.tolist() for x in lo], width=w.tolist())
    r = dict(fid=fid(lo, LO_R), sep_kept=s_ / REFS_, auroc=a_, width_kept=float(w.mean() / W0.mean()))
    print(name, mode, {k: round(v, 3) for k, v in r.items()}, flush=True); return r
CIRC = (set(EARLY_KEEP), set(LATE_KEEP), set(MLP_KEEP))
EARLY_C = [("e", l, h) for l in range(1, 6) for h in range(Hn) if (l, h) not in EARLY_KEEP]
LATE_C = [("h", l, h) for l in range(6, NL) for h in range(Hn) if (l, h) not in LATE_KEEP] + [("m", l, None) for l in range(6, NL) if l not in MLP_KEEP]
def parse(n_):
    if n_.startswith("MLP"): return ("m", int(n_[3:]), None)
    l, h = (int(x) for x in n_[1:].split("H")); return ("e" if l <= 5 else "h", l, h)
def keep_at(k):
    kk = CIRC
    for n_ in PATH[:k]: kk = with_add(kk, parse(n_))
    return kk
res = dict(split=SPLIT, path=PATH, n_paragraphs=len(REP), n_items=len(items), intact_sep=sep_auc(REP, LO_R)[0], intact_auroc=sep_auc(REP, LO_R)[1], intact_width=float(W0.mean()))
print("intact", res, flush=True)
for mode in ("mean", "resample"): res[f"circuit|{mode}"] = run("circuit", CIRC, mode)
for k in ([4, 24] if SMOKE else [4, 8, 16, 24]):
    for mode in ("mean", "resample"): res[f"plus{k}|{mode}"] = run(f"plus{k}", keep_at(k), mode)
ne = sum(parse(n_)[0] == "e" for n_ in PATH); nlt = len(PATH) - ne
rs = np.random.default_rng(0)
for d in range(2 if SMOKE else 5):
    kk = CIRC
    for i in rs.choice(len(EARLY_C), ne, replace=False): kk = with_add(kk, EARLY_C[i])
    for i in rs.choice(len(LATE_C), nlt, replace=False): kk = with_add(kk, LATE_C[i])
    for mode in ("mean", "resample"): res[f"random{d}|{mode}"] = run(f"random{d}", kk, mode)
res["recs"] = [[dict(label=r["label"], remaining=r["remaining"]) for r in recs] for _, recs in REP]
res["raw"] = RAW
save_json(res, f"p5_01_sufficiency_{SPLIT}" + ("_smoke" if SMOKE else "") + ".json")
