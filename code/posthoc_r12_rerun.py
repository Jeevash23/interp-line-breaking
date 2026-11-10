"""POST HOC (not confirmatory). Re-runs the ablation-based measurements of rounds 1-2 on the splits
already read (validation, test), keeping per-paragraph / per-item data so that bootstrap
p-values can be computed for a Holm correction, and adds RESAMPLE ablation alongside mean
ablation. Frozen verdicts are not changed by anything here.

Usage: posthoc_r12_rerun.py validation|test [--smoke]
Samples, seeds and measures follow the frozen scripts (conf01/c2_01 for token regressions,
c2_03 for newline-only ablation, previous-line tokens and the dissociation). Mean ablation uses
the same means as those scripts, so its point estimates should reproduce the frozen outputs.
Resample ablation: the head's attention-output input slice is copied from a forward pass on a
DIFFERENT paragraph (WikiText-2 train, seed 7, wrapped at the same width; the donor for each target
is the next one in a seeded cycle that is long enough). All-position ablation copies position t
from the donor's position t; newline-only ablation copies the target's k-th newline from the
donor's k-th newline. In width items both prompts get the same donor.
Bootstrap: 10,000 resamples, seed 0, identical across conditions (paired). Samples saved to
outputs/posthoc_r12_<split>.npz; points to outputs/posthoc_r12_<split>.json.
"""
import sys, json, random, numpy as np, torch
from common import *

SPLIT = sys.argv[1]; SMOKE = "--smoke" in sys.argv
assert SPLIT in ("validation", "test") or (SMOKE and SPLIT == "train")
NPARA = 4 if SMOKE else 60; NITEMS = 6 if SMOKE else 150; NB = 100 if SMOKE else 10000
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
LM = 6
COUNT4 = [(2, 7), (4, 15), (5, 6), (5, 7)]
EXP04 = {(2, 7), (2, 11), (2, 13), (4, 10), (4, 15), (5, 6), (5, 7), (6, 11)}
pool = [(l, h) for l in range(1, LM) for h in range(Hn) if (l, h) not in EXP04]
rs = np.random.default_rng(0)
RAND = [[pool[i] for i in rs.choice(len(pool), 4, replace=False)] for _ in range(10)]   # as c2_03

CAP = {}; ST = dict(capture=False, patch={})
def hook(l):
    def f(m, a):
        z = a[0]
        if ST["capture"]: CAP[l] = z[0].detach().clone()
        lst = ST["patch"].get(l)
        if not lst: return None
        z = z.clone()
        for h, pos, val in lst: z[:, pos, h * dh:(h + 1) * dh] = val
        return (z,)
    return f
for l in range(LM): layers[l].attention.dense.register_forward_pre_hook(hook(l))
def fwd(ids, capture=False):
    ST["capture"] = capture
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0].float(), -1)[:, nl].cpu().numpy()
    ST["capture"] = False
    return lp

# means, exactly as the frozen scripts
cap1 = {}
hs_ = [layers[l].attention.dense.register_forward_pre_hook(lambda m, a, l=l: cap1.__setitem__(l, a[0].detach())) for l in range(LM)]
zs = {l: [] for l in range(LM)}
with torch.no_grad():
    for p in train_paragraphs(seed=6)[:30]:
        model.gpt_neox(torch.tensor([tok(p, add_special_tokens=False)["input_ids"][:500]], device="cuda"))
        for l in range(LM): zs[l].append(cap1[l][0].mean(0))
for h_ in hs_: h_.remove()
MEAN_ALL = {l: torch.stack(zs[l]).mean(0) for l in range(LM)}
acc = {l: [] for l in range(LM)}
for q, p in enumerate(train_paragraphs(seed=6)[:30]):
    lines = wrap(p, (40, 60, 80)[q % 3])
    if lines is None: continue
    ids = tok("\n".join(lines), add_special_tokens=False)["input_ids"][:700]
    fwd(ids, capture=True)
    pos = [i for i, x in enumerate(ids) if x == nl]
    for l in range(LM): acc[l].append(CAP[l][pos])
MEAN_NL = {l: torch.cat(acc[l]).mean(0) for l in range(LM)}

# donors
DON = {}
def donor_pool(W):
    if W not in DON:
        lst = []
        for p in train_paragraphs(seed=7):
            if len(lst) >= 400: break
            lines = wrap(p, W)
            if lines is None or len(lines) < 4: continue
            ids = tok("\n".join(lines), add_special_tokens=False)["input_ids"]
            if len(ids) <= 900: lst.append(ids)
        DON[W] = dict(lst=lst, k=0)
    return DON[W]
def pick_donor(W, T, n_nl):
    d = donor_pool(W)
    for _ in range(len(d["lst"])):
        ids = d["lst"][d["k"] % len(d["lst"])]; d["k"] += 1
        if len(ids) >= T and sum(1 for x in ids if x == nl) >= n_nl:
            fwd(ids, capture=True)
            return dict(z={l: CAP[l].clone() for l in range(LM)}, nls=[i for i, x in enumerate(ids) if x == nl])
    raise RuntimeError("no donor long enough")

def run(ids, heads=(), where=None, kind=None, donor=None):
    ST["patch"] = {}
    if heads:
        T = len(ids); nls = [i for i, x in enumerate(ids) if x == nl]
        if where == "all": pos = list(range(T)); dpos = pos
        elif where == "nl": pos = nls; dpos = donor["nls"][:len(nls)] if donor else None
        elif where == "last": pos = [nls[-1]]; dpos = [donor["nls"][len(nls) - 1]] if donor else None
        pt = torch.tensor(pos, device="cuda")
        for l, h in heads:
            sl = slice(h * dh, (h + 1) * dh)
            if kind == "mean": val = (MEAN_ALL if where == "all" else MEAN_NL)[l][sl]
            else: val = donor["z"][l][torch.tensor(dpos, device="cuda"), sl]
            ST["patch"].setdefault(l, []).append((h, pt, val))
    lp = fwd(ids); ST["patch"] = {}
    return lp

paras = split_paragraphs(SPLIT, seed=100)
br = np.random.default_rng(0)
def spline(rem): return np.c_[rem, *[np.maximum(rem - k, 0) for k in [2, 5, 8, 12, 16, 24, 32]]]
SAMP, PTS = {}, {}

# ---------------- token regressions (H1/R1, H3, H6)
TOKC = {"intact": ((), None, None)}
for kind in ("mean", "resample"):
    for nm, hs in {"L5H7": [(5, 7)], "L5H3": [(5, 3)], "L5H11": [(5, 11)], "L5H14": [(5, 14)], "L4H14": [(4, 14)]}.items():
        TOKC[f"{kind}_all_{nm}"] = (hs, "all", kind)
    for nm, hs in {"L5H7": [(5, 7)], "L5H11": [(5, 11)]}.items():
        TOKC[f"{kind}_nl_{nm}"] = (hs, "nl", kind)
for W in (40, 60, 80):
    items = []
    for p in paras:
        if len(items) >= NPARA: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        items.append((ids, recs, [i for i, t in enumerate(ids) if t == nl]))
    donors = [pick_donor(W, len(ids), len(nls)) for ids, _, nls in items]
    idx = [br.integers(0, len(items), len(items)) for _ in range(NB)]
    for c, (hs, wh, kind) in TOKC.items():
        X, y, g = [], [], []
        for k, (ids, recs, nls) in enumerate(items):
            lp = run(ids, hs, wh, kind, donors[k] if kind == "resample" else None)
            for r in recs:
                if r["remaining"] > 30: continue
                li = r["line"]; prev_ntok = nls[li - 1] - (nls[li - 2] + 1 if li >= 2 else 0)
                v = lp[r["i"]]; X.append((r["remaining"], r["ntok_line"], r["prevlen"], prev_ntok)); y.append(v - np.log1p(-np.exp(min(v, -1e-6)))); g.append(k)
        X = np.array(X, float); y = np.array(y); g = np.array(g)
        groups = [np.where(g == k)[0] for k in range(len(items))]
        M1 = np.c_[np.ones(len(y)), spline(X[:, 0]), X[:, 1]]                 # token coefficient (H1)
        M6 = np.c_[np.ones(len(y)), spline(X[:, 0]), X[:, 1:]]                # previous-line tokens (H6)
        w1 = np.linalg.lstsq(M1, y, rcond=None)[0][-1]; w6 = np.linalg.lstsq(M6, y, rcond=None)[0][-1]
        b1, b6 = [], []
        for ii_ in idx:
            ii = np.concatenate([groups[j] for j in ii_])
            b1.append(np.linalg.lstsq(M1[ii], y[ii], rcond=None)[0][-1]); b6.append(np.linalg.lstsq(M6[ii], y[ii], rcond=None)[0][-1])
        PTS[f"tok_w{W}_{c}"] = float(w1); PTS[f"prevtok_w{W}_{c}"] = float(w6)
        SAMP[f"tok_w{W}_{c}"] = np.array(b1); SAMP[f"prevtok_w{W}_{c}"] = np.array(b6)
        print(W, c, round(float(w1), 4), round(float(w6), 4), flush=True)

# ---------------- width items (H4)
W = 60; rng = random.Random(200); items = []
for p in paras:
    if len(items) >= NITEMS: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1)); words = lines[k].split(" ")
    js = [j for j in range(2, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    prefix = " ".join(words[: js[0] + 1]); before = lines[:k]
    pw = before[-1].split(" "); short = before[-1]
    while len(pw) > 2 and len(before[-1]) - len(short) < 10: pw = pw[:-1]; short = " ".join(pw)
    if len(before[-1]) - len(short) < 10: continue
    ib = tok("\n".join(before) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    is_ = tok("\n".join(before[:-1] + [short]) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    nb = [i for i, t in enumerate(ib) if t == nl][-1]; ns = [i for i, t in enumerate(is_) if t == nl][-1]
    if len(ib) - nb != len(is_) - ns: continue
    items.append((ib, is_))
donors = [pick_donor(W, len(b), sum(1 for x in b if x == nl)) for b, _ in items]
CONDA = {"intact": ((), None, None)}
for kind in ("mean", "resample"):
    CONDA[f"{kind}_COUNT4_nl"] = (COUNT4, "nl", kind); CONDA[f"{kind}_COUNT4_last"] = (COUNT4, "last", kind)
    for q, r in enumerate(RAND): CONDA[f"{kind}_random{q}_nl"] = (r, "nl", kind)
EFF = {c: np.array([float(run(s, hs, wh, kind, donors[k] if kind == "resample" else None)[-1] - run(b, hs, wh, kind, donors[k] if kind == "resample" else None)[-1])
                    for k, (b, s) in enumerate(items)]) for c, (hs, wh, kind) in CONDA.items()}
n = len(items); idx = [br.integers(0, n, n) for _ in range(NB)]
E = EFF["intact"]
PTS["width_effect"] = float(E.mean()); SAMP["width_effect"] = np.array([E[i].mean() for i in idx])
for c in CONDA:
    if c == "intact": continue
    F = EFF[c]; PTS[f"share_{c}"] = float(1 - F.mean() / E.mean()); SAMP[f"share_{c}"] = np.array([1 - F[i].mean() / E[i].mean() for i in idx])
for kind in ("mean", "resample"):
    ks = [f"share_{kind}_random{q}_nl" for q in range(10)]
    PTS[f"randmean_{kind}"] = float(np.mean([PTS[k] for k in ks])); SAMP[f"randmean_{kind}"] = np.mean([SAMP[k] for k in ks], 0)
print("H4", {k: round(v, 3) for k, v in PTS.items() if k.startswith(("share_mean_COUNT4", "share_resample_COUNT4", "randmean", "width"))}, flush=True)

# ---------------- dissociation (H4d)
nat = []
for p in paras:
    if len(nat) >= NPARA: break
    lines = wrap(p, 60)
    if lines is None or len(lines) < 5: continue
    ids, recs = word_end_positions(tok, lines, 60)
    if len(ids) <= 700: nat.append((ids, recs))
donors = [pick_donor(60, len(ids), sum(1 for x in ids if x == nl)) for ids, _ in nat]
idx = [br.integers(0, len(nat), len(nat)) for _ in range(NB)]
SEP = {}
for c, (hs, wh, kind) in {"intact": ((), None, None), "mean_COUNT4_nl": (COUNT4, "nl", "mean"), "mean_COUNT4_all": (COUNT4, "all", "mean"),
                          "resample_COUNT4_nl": (COUNT4, "nl", "resample"), "resample_COUNT4_all": (COUNT4, "all", "resample")}.items():
    s1, n1, s0, n0 = [], [], [], []
    for k, (ids, recs) in enumerate(nat):
        lp = run(ids, hs, wh, kind, donors[k] if kind == "resample" else None)
        a1 = [lp[r["i"]] for r in recs if r["label"] == 1]; a0 = [lp[r["i"]] for r in recs if r["label"] == 0 and r["remaining"] <= 12]
        lo = lambda v: v - np.log1p(-np.exp(np.minimum(v, -1e-6)))
        s1.append(lo(np.array(a1)).sum() if a1 else 0.0); n1.append(len(a1)); s0.append(lo(np.array(a0)).sum() if a0 else 0.0); n0.append(len(a0))
    s1, n1, s0, n0 = map(np.array, (s1, n1, s0, n0))
    SEP[c] = (s1.sum() / n1.sum() - s0.sum() / n0.sum(), np.array([s1[i].sum() / n1[i].sum() - s0[i].sum() / n0[i].sum() for i in idx]))
for c in SEP:
    if c == "intact": continue
    PTS[f"kept_{c}"] = float(SEP[c][0] / SEP["intact"][0]); SAMP[f"kept_{c}"] = SEP[c][1] / SEP["intact"][1]
PTS["separation_intact"] = float(SEP["intact"][0])
print("H4d", {k: round(v, 3) for k, v in PTS.items() if k.startswith("kept")}, flush=True)

tag = f"posthoc_r12_{SPLIT}{'_smoke' if SMOKE else ''}"
np.savez_compressed(OUT / f"{tag}.npz", **SAMP)
save_json(dict(split=SPLIT, smoke=SMOKE, n_boot=NB, n_width_items=n, points=PTS), f"{tag}.json")
