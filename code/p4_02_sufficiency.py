"""p4_02 (round 4, CONFIRMATORY under PROTOCOL-4.md): a copy of exp25_sufficiency.py. Changes: natural text
and width items come from the named split (newtext2.paras); ablation means, the resample pool and the
early-resample donors still come from WikiText-2 train; per-paragraph and per-item values are saved for the
bootstrap in report4_tables.py; output p4_02_sufficiency_<split>.json. Original docstring follows.
exp25: is the found circuit enough?

Circuit (from exp04, exp14, exp16-22):
  early heads (layers 1-5): exp04's eight critical heads L2H7, L2H11, L2H13, L4H10, L4H15, L5H6,
    L5H7, L6H11 (L6H11 counted with the late heads), plus the corrector L4H14;
  late heads (layers 6-23): readers L6H2, L8H15, L17H1, plus L6H11, L15H10, L19H0, L20H6
    (exp22's next-largest direct writers and the width mediator of exp14);
  MLPs: all of layers 0-5 (MLP0 acts as part of the embedding), plus MLP6, 14, 15, 16, 18, 21, 23
    (exp22's direct writers; 21 and 23 damp).
Everything outside the circuit is ablated, in three scopes:
  DEC   : heads and MLPs of layers 6-23 outside the circuit, at decision points only;
  EARLY : heads of layers 1-5 outside the circuit, at every position (layer-0 heads kept);
  BOTH  : DEC and EARLY together.
Two ablations: mean (head inputs to the output projection and MLP outputs replaced by their mean over
  decision points for DEC, over all positions for EARLY; 30 train paragraphs, seed 6) and resample
  (DEC: values from one random decision point of a pool of 1,500 from other paragraphs, the same
  donor point for every component; EARLY: position-aligned values from a different train paragraph
  wrapped at the same width, seed 7, falling back to the mean past its end).
Controls: 5 random circuits of the same size per scope (same number of kept heads per layer block
  1-5 and 6-23, same number of kept MLPs among 6-23), drawn with seed 0, mean ablation.
Measures (width 60): separation (mean newline log-odds at breaks minus at non-breaks with <= 12
  characters left) and AUROC on 40 paragraphs (seed-25 order); the width effect (previous line
  shortened by >= 10 characters) on 80 items, with the same ablation in both prompts.
Run: python3 p4_02_sufficiency.py <train|conf4-pythia410m> [--smoke]
"""
import sys, json, random, numpy as np, torch
from common import *
from newtext2 import paras
SPLIT = sys.argv[1]; assert SPLIT in ("train", "conf4-pythia410m")

SMOKE = "--smoke" in sys.argv
NPARA = 5 if SMOKE else 40; NIT = 6 if SMOKE else 80; NRAND = 1 if SMOKE else 5
W = 60
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers; NL = len(layers)
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
EARLY_KEEP = {(2, 7), (2, 11), (2, 13), (4, 10), (4, 15), (5, 6), (5, 7), (4, 14)}
LATE_KEEP = {(6, 2), (8, 15), (17, 1), (6, 11), (15, 10), (19, 0), (20, 6)}
MLP_KEEP = {6, 14, 15, 16, 18, 21, 23}

# --- state used by hooks
ST = dict(head_abl={}, mlp_abl=set(), dec_mask=None, early_mask=None, mode="mean", donor_dec=None, donor_early=None)
MEAN_ALL, MEAN_DEC, CAPZ, CAPM = {}, {}, {}, {}
CAPTURE = dict(on=False)
def zhook(l):
    def f(m, a):
        z = a[0]
        if CAPTURE["on"]: CAPZ[l] = z[0].detach().clone()
        hs = ST["head_abl"].get(l)
        if not hs: return None
        z = z.clone()
        early = l <= 5
        mask = ST["early_mask"] if early else ST["dec_mask"]
        if mask is None: return None
        for h in hs:
            sl = slice(h * dh, (h + 1) * dh)
            if ST["mode"] == "mean":
                z[:, mask, sl] = (MEAN_ALL if early else MEAN_DEC)[("z", l)][sl]
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
        o[:, mask] = MEAN_DEC[("m", l)] if ST["mode"] == "mean" else ST["donor_dec"][("m", l)]
        return o
    return f
for l in range(NL):
    layers[l].attention.dense.register_forward_pre_hook(zhook(l))
    layers[l].mlp.register_forward_hook(mhook(l))

def lp_seq(ids):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0].float(), -1)[:, nl]
    return (lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-6))).cpu().numpy()

def paras_ok(seed, n, minlines=5, split="train"):
    out = []
    for p in paras(split, seed):
        if len(out) >= n: break
        lines = wrap(p, W)
        if lines is None or len(lines) < minlines: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        out.append((ids, recs))
    return out

# --- means and the resample pool
CAPTURE["on"] = True
acc_all, acc_dec, nall, ndec = {}, {}, 0, 0
pool = []
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

def set_scope(scope, keep_early, keep_late, keep_mlp):
    ST["head_abl"] = {}; ST["mlp_abl"] = set()
    if scope in ("EARLY", "BOTH"):
        for l in range(1, 6):
            for h in range(Hn):
                if (l, h) not in keep_early: ST["head_abl"].setdefault(l, []).append(h)
    if scope in ("DEC", "BOTH"):
        for l in range(6, NL):
            for h in range(Hn):
                if (l, h) not in keep_late: ST["head_abl"].setdefault(l, []).append(h)
            if l not in keep_mlp: ST["mlp_abl"].add(l)

def prepare(ids, dec_positions, scope, mode, didx):
    ST["mode"] = mode
    ST["dec_mask"] = torch.tensor(dec_positions, device="cuda", dtype=torch.long) if scope in ("DEC", "BOTH") and len(dec_positions) else None
    ST["early_mask"] = torch.arange(len(ids), device="cuda") if scope in ("EARLY", "BOTH") else None
    if mode == "resample":
        d = pool[prng.randrange(len(pool))]
        ST["donor_dec"] = {k: v.unsqueeze(0).expand(len(dec_positions), -1) for k, v in d.items()} if ST["dec_mask"] is not None else None
        if ST["early_mask"] is not None:
            dids = donor_paras[didx % len(donor_paras)][0]
            CAPTURE["on"] = True; ST_save = dict(ST); ST["head_abl"] = {}; ST["mlp_abl"] = set()
            lp_seq(dids)
            ST.update(ST_save); CAPTURE["on"] = False
            n = len(ids)
            ST["donor_early"] = {}
            for l in range(1, 6):
                z = MEAN_ALL[("z", l)].unsqueeze(0).repeat(n, 1)
                m = min(n, CAPZ[l].shape[0]); z[:m] = CAPZ[l][:m]
                ST["donor_early"][("z", l)] = z

nat = paras_ok(25, NPARA, split=SPLIT)
def separation(scope, mode, keep):
    br, nb, sc, lab, pbr, pnb, psc = [], [], [], [], [], [], []
    for q, (ids, recs) in enumerate(nat):
        set_scope(scope, *keep); prepare(ids, [r["i"] for r in recs], scope, mode, q)
        lo = lp_seq(ids)
        for r in recs:
            if r["label"] == 1: br.append(lo[r["i"]]); pbr.append(q)
            elif r["remaining"] <= 12: nb.append(lo[r["i"]]); pnb.append(q)
            sc.append(lo[r["i"]]); lab.append(r["label"]); psc.append(q)
    ST["dec_mask"] = ST["early_mask"] = None
    RAW.setdefault("sep", []).append(dict(br=[float(x) for x in br], nb=[float(x) for x in nb], sc=[float(x) for x in sc], lab=lab, pbr=pbr, pnb=pnb, psc=psc))
    return float(np.mean(br) - np.mean(nb)), auroc(sc, lab)

# width items (exp14 construction, seed 14)
rng = random.Random(14); items = []
for p in paras(SPLIT, 14):
    if len(items) >= NIT: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1))
    words = lines[k].split(" ")
    js = [j for j in range(2, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    prefix = " ".join(words[: js[0] + 1])
    before = lines[:k]; pw = before[-1].split(" "); short = before[-1]
    while len(pw) > 2 and len(before[-1]) - len(short) < 10:
        pw = pw[:-1]; short = " ".join(pw)
    if len(before[-1]) - len(short) < 10: continue
    ib = tok("\n".join(before) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    is_ = tok("\n".join(before[:-1] + [short]) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    items.append((ib, is_))
def width_effect(scope, mode, keep):
    e = []
    for q, (ib, is_) in enumerate(items):
        v = []
        for ids in (ib, is_):
            set_scope(scope, *keep); prepare(ids, [len(ids) - 1], scope, mode, q)
            v.append(lp_seq(ids)[-1])
        e.append(v[1] - v[0])
    ST["dec_mask"] = ST["early_mask"] = None
    RAW.setdefault("width", []).append([float(x) for x in e])
    return float(np.mean(e)), float(np.std(e) / np.sqrt(len(e)))

RAW = {}
CIRC = (EARLY_KEEP, LATE_KEEP, MLP_KEEP)
res = dict(n_paragraphs=len(nat), n_items=len(items))
set_scope("NONE", *CIRC)
res["intact"] = dict(sep_auroc=separation("NONE", "mean", CIRC), width=width_effect("NONE", "mean", CIRC))
print("intact", res["intact"], flush=True)
rs = np.random.default_rng(0)
early_pool = [(l, h) for l in range(1, 6) for h in range(Hn)]
late_pool = [(l, h) for l in range(6, NL) for h in range(Hn)]
for scope in ("DEC", "EARLY", "BOTH"):
    for mode in ("mean", "resample"):
        r = dict(sep_auroc=separation(scope, mode, CIRC), width=width_effect(scope, mode, CIRC))
        res.setdefault(scope, {})[mode] = r
        print(scope, mode, r, flush=True)
    rands = []
    for q in range(NRAND):
        ke = {early_pool[i] for i in rs.choice(len(early_pool), len(EARLY_KEEP), replace=False)}
        kl = {late_pool[i] for i in rs.choice(len(late_pool), len(LATE_KEEP), replace=False)}
        km = {int(x) for x in rs.choice(np.arange(6, NL), len(MLP_KEEP), replace=False)}
        rands.append(dict(sep_auroc=separation(scope, "mean", (ke, kl, km)), width=width_effect(scope, "mean", (ke, kl, km))))
    res[scope]["random_mean"] = rands
    print(scope, "random", [(round(x["sep_auroc"][0], 2), round(x["width"][0], 2)) for x in rands], flush=True)
res["order"] = "intact, then per scope: mean, resample, random 0..NRAND-1 (separation and width lists in this order)"
res["raw"] = RAW
save_json(res, f"p4_02_sufficiency_{SPLIT}" + ("_smoke" if SMOKE else "") + ".json")
print(json.dumps({k: v for k, v in res.items() if k != 'raw'}, indent=1))
