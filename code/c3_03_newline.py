"""c3_03_newline (CONFIRMATORY, PROTOCOL-3.md): round 3 on new text (Wikipedia articles created 2025-26,
part conf-pythia410m). A copy of the frozen code/c2_03_newline.py; the only changes are this header, the
text loader (newtext.load_paras in place of split_paragraphs), the split name allowed and the
output file name.
Usage: c3_03_newline.py conf-pythia410m   (the frozen run);  c3_03_newline.py train --smoke   (pipeline check only)

Original docstring:
c2_03 (CONFIRMATORY, PROTOCOL-2.md H4 and H6): the counting heads re-used at the newline token.

Usage: c2_03_newline.py test              (the frozen run)
       c2_03_newline.py train --smoke     (pipeline check only; 8 items, 5 paragraphs per width)
COUNT4 = L2H7, L4H15, L5H6, L5H7: the heads that are both among exp04's decision-critical heads
and among exp20's writers of the previous line's length at the newline.
Newline-only ablation: a head's attention-output input slice is replaced by its mean over
newline positions (30 train paragraphs, seed 6, wrapped at widths 40/60/80 in turn, first 700
tokens) at the chosen positions only; at every other position the head runs normally.
H4 items: exactly as code/c2_02_width.py (width 60, one per paragraph, seed-100 paragraph
order, random choices seed 200, up to 150). Width effect = log P(newline) at the decision point
in the short-previous-line prompt minus in the base prompt, with the same ablation in both.
Share removed = 1 - mean(effect under ablation) / mean(intact effect).
 Conditions: COUNT4 at every newline; COUNT4 at the last newline only; 10 random sets of 4 heads
 from layers 1-5 outside exp04's eight critical heads (L2H7, L2H11, L2H13, L4H10, L4H15, L5H6,
 L5H7, L6H11), drawn once with numpy seed 0, each at every newline.
H4 dissociation: natural text, width 60, the first 60 paragraphs (seed-100 order) with >= 5
 lines and <= 700 tokens; points with label 1 or <= 12 characters remaining. Separation = mean
 newline log-odds at breaks minus at non-breaks. Intact, COUNT4 at newlines only, COUNT4 at
 every position.
H6: natural text, widths 40, 60, 80, same paragraphs as conf01/c2_01 (first 60 per width).
 Points with <= 30 characters remaining. Regression of the newline log-odds on the spline in
 characters remaining (knots 2, 5, 8, 12, 16, 24, 32), tokens on the current line, the previous
 line's character length and the previous line's token count (newline excluded). Coefficient
 of the previous line's token count; intact, L5H7 at every newline, L5H11 at every newline.
 Paired paragraph bootstrap (2,000 resamples, seed 0, identical across conditions).
Item bootstrap for H4: 2,000 resamples, seed 0.
"""
import sys, json, random, numpy as np, torch
from common import *
from newtext import load_paras

SPLIT = sys.argv[1]; SMOKE = "--smoke" in sys.argv
assert SPLIT in ("train", "conf-pythia410m")
NITEMS = 8 if SMOKE else 150
NPARA = 5 if SMOKE else 60
NBOOT = 50 if SMOKE else 2000
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
LM = 6
COUNT4 = [(2, 7), (4, 15), (5, 6), (5, 7)]
EXP04 = {(2, 7), (2, 11), (2, 13), (4, 10), (4, 15), (5, 6), (5, 7), (6, 11)}
pool = [(l, h) for l in range(1, LM) for h in range(Hn) if (l, h) not in EXP04]
rs = np.random.default_rng(0)
RAND = [[pool[i] for i in rs.choice(len(pool), 4, replace=False)] for _ in range(10)]

CAP = {}; ST = dict(heads=[], mask=None, capture=False)
def hook(l):
    def f(m, a):
        z = a[0]
        if ST["capture"]: CAP[l] = z[0].detach().clone()
        hs = [h for (ll, h) in ST["heads"] if ll == l]
        if not hs or ST["mask"] is None: return None
        z = z.clone()
        for h in hs: z[:, ST["mask"], h * dh:(h + 1) * dh] = MEAN[l][h * dh:(h + 1) * dh]
        return (z,)
    return f
for l in range(LM): layers[l].attention.dense.register_forward_pre_hook(hook(l))
MEAN = {}
def run(ids, heads=(), where=None):
    t = torch.tensor([ids], device="cuda")
    if where is None or not heads: ST.update(heads=[], mask=None)
    else:
        m = torch.zeros(len(ids), dtype=torch.bool, device="cuda")
        if where == "all": m[:] = True
        elif where == "nl": m[t[0] == nl] = True
        elif where == "last": m[[i for i, x in enumerate(ids) if x == nl][-1]] = True
        ST.update(heads=list(heads), mask=m)
    with torch.no_grad():
        lp = torch.log_softmax(model(t).logits[0].float(), -1)[:, nl].cpu().numpy()
    ST.update(heads=[], mask=None)
    return lp

acc = {l: [] for l in range(LM)}
ST["capture"] = True
for q, p in enumerate(train_paragraphs(seed=6)[:30]):
    lines = wrap(p, (40, 60, 80)[q % 3])
    if lines is None: continue
    ids = tok("\n".join(lines), add_special_tokens=False)["input_ids"][:700]
    run(ids)
    pos = [i for i, x in enumerate(ids) if x == nl]
    for l in range(LM): acc[l].append(CAP[l][pos])
ST["capture"] = False
MEAN = {l: torch.cat(acc[l]).mean(0) for l in range(LM)}
pct = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
paras = load_paras(SPLIT, seed=100)
out = dict(split=SPLIT, smoke=SMOKE, random_sets=[[f"L{l}H{h}" for l, h in r] for r in RAND])

# ---------------- H4: width effect under newline-only ablation
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
CONDA = {"intact": ((), None), "COUNT4_nl": (COUNT4, "nl"), "COUNT4_last": (COUNT4, "last")}
for q, r in enumerate(RAND): CONDA[f"random{q}_nl"] = (r, "nl")
A = {c: np.array([float(run(s, hs, wh)[-1] - run(b, hs, wh)[-1]) for b, s in items]) for c, (hs, wh) in CONDA.items()}
n = len(items); br = np.random.default_rng(0); idx = [br.integers(0, n, n) for _ in range(NBOOT)]
E = A["intact"]
def share(c):
    F = A[c]; return dict(share=float(1 - F.mean() / E.mean()), ci95=pct([1 - F[i].mean() / E[i].mean() for i in idx]))
shares = {c: share(c) for c in CONDA if c != "intact"}
rand_mean = float(np.mean([shares[f"random{q}_nl"]["share"] for q in range(10)]))
eff_ci = pct([E[i].mean() for i in idx])
out["H4_width"] = dict(n=n, width_effect=float(E.mean()), width_effect_ci95=eff_ci, shares=shares, random_mean_share=rand_mean,
                       effects={c: float(v.mean()) for c, v in A.items()})
print("H4 width", json.dumps(out["H4_width"]), flush=True)

# ---------------- H4: dissociation on natural text
nat = []
for p in paras:
    if len(nat) >= NPARA: break
    lines = wrap(p, 60)
    if lines is None or len(lines) < 5: continue
    ids, recs = word_end_positions(tok, lines, 60)
    if len(ids) <= 700: nat.append((ids, recs))
B = {}
for c, (hs, wh) in {"intact": ((), None), "COUNT4_nl": (COUNT4, "nl"), "COUNT4_all": (COUNT4, "all")}.items():
    sc, lab = [], []
    for ids, recs in nat:
        lp = run(ids, hs, wh)
        for r in recs:
            if r["remaining"] <= 12 or r["label"] == 1:
                v = lp[r["i"]]; sc.append(v - np.log1p(-np.exp(min(v, -1e-6)))); lab.append(r["label"])
    sc, lab = np.array(sc), np.array(lab)
    B[c] = dict(n_points=len(sc), separation=float(sc[lab == 1].mean() - sc[lab == 0].mean()), auroc=auroc(sc, lab))
for c in ("COUNT4_nl", "COUNT4_all"): B[c]["kept"] = B[c]["separation"] / B["intact"]["separation"]
out["H4_dissociation"] = dict(n_paragraphs=len(nat), **B)
print("H4 dissociation", json.dumps(out["H4_dissociation"]), flush=True)

# ---------------- H6: previous line's tokens with L5H7 ablated at newlines only
def spline(rem): return np.c_[rem, *[np.maximum(rem - k, 0) for k in [2, 5, 8, 12, 16, 24, 32]]]
H6 = {}
for W in (40, 60, 80):
    its = []
    for p in paras:
        if len(its) >= NPARA: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        its.append((ids, recs, [i for i, t in enumerate(ids) if t == nl]))
    H6[W] = {}; bo = {}
    for c, (hs, wh) in {"intact": ((), None), "L5H7_nl": ([(5, 7)], "nl"), "L5H11_nl": ([(5, 11)], "nl")}.items():
        X, yy, g = [], [], []
        for k, (ids, recs, nls) in enumerate(its):
            lp = run(ids, hs, wh)
            for r in recs:
                if r["remaining"] > 30: continue
                li = r["line"]
                prev_ntok = nls[li - 1] - (nls[li - 2] + 1 if li >= 2 else 0)
                v = lp[r["i"]]; X.append((r["remaining"], r["ntok_line"], r["prevlen"], prev_ntok)); yy.append(v - np.log1p(-np.exp(min(v, -1e-6)))); g.append(k)
        X = np.array(X, float); yy = np.array(yy); g = np.array(g)
        M = np.c_[np.ones(len(yy)), spline(X[:, 0]), X[:, 1:]]
        w = np.linalg.lstsq(M, yy, rcond=None)[0]
        br = np.random.default_rng(0); bs = []
        groups = [np.where(g == k)[0] for k in range(len(its))]
        for _ in range(NBOOT):
            ii = np.concatenate([groups[j] for j in br.integers(0, len(its), len(its))])
            bs.append(np.linalg.lstsq(M[ii], yy[ii], rcond=None)[0][-1])
        bo[c] = np.array(bs)
        H6[W][c] = dict(n_points=len(yy), n_paragraphs=len(its), prev_ntok=float(w[-1]), ci95=pct(bs), prevlen=float(w[-2]), ntok_line=float(w[-3]))
        print("H6", W, c, json.dumps(H6[W][c]), flush=True)
    for c in ("L5H7_nl", "L5H11_nl"):
        H6[W][c + "_minus_intact"] = dict(diff=H6[W][c]["prev_ntok"] - H6[W]["intact"]["prev_ntok"], ci95=pct(bo[c] - bo["intact"]))
out["H6"] = H6

# ---------------- verdicts (PROTOCOL-2.md)
pos_ok = eff_ci[0] > 0
v = dict(positive_control=bool(pos_ok),
         H4a=bool(pos_ok and shares["COUNT4_nl"]["share"] >= 0.25 and shares["COUNT4_nl"]["ci95"][0] > 0),
         H4b=bool(pos_ok and shares["COUNT4_last"]["share"] >= 0.25 and shares["COUNT4_last"]["ci95"][0] > 0),
         H4c=bool(pos_ok and rand_mean < 0.15),
         H4d=bool(B["COUNT4_nl"]["kept"] >= 0.80 and B["COUNT4_all"]["kept"] < 0.30))
for W in (40, 60):
    d7 = H6[W]["L5H7_nl_minus_intact"]; d11 = H6[W]["L5H11_nl_minus_intact"]
    v[f"H6a_w{W}"] = bool(d7["diff"] <= -0.03 and d7["ci95"][1] < 0)
    v[f"H6b_w{W}"] = bool(abs(d11["diff"]) < 0.02)
out["verdicts"] = v
out["H4_pass"] = bool(all(v[k] for k in ("positive_control", "H4a", "H4b", "H4c", "H4d")))
out["H6_pass"] = bool(all(v[k] for k in v if k.startswith("H6")))
save_json(out, f"c3_03_newline_{SPLIT}{'_smoke' if SMOKE else ''}.json")
print(json.dumps(dict(verdicts=v, H4_pass=out["H4_pass"], H6_pass=out["H6_pass"]), indent=1))
