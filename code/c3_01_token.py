"""c3_01_token (CONFIRMATORY, PROTOCOL-3.md): round 3 on new text (Wikipedia articles created 2025-26,
part conf-pythia410m). A copy of the frozen code/c2_01_token.py; the only changes are this header, the
text loader (newtext.load_paras in place of split_paragraphs), the split name allowed and the
output file name, plus the round-3 residual verdict for width 40 (see PROTOCOL-3.md).
Usage: c3_01_token.py conf-pythia410m   (the frozen run);  c3_01_token.py train --smoke   (pipeline check only)

Original docstring:
c2_01 (CONFIRMATORY, PROTOCOL-2.md R1 and H3): replication of PROTOCOL.md H1 on the test split,
plus L4H14 as a second token corrector.

Usage: c2_01_token.py test              (the frozen run)
       c2_01_token.py train --smoke     (pipeline check only; 5 paragraphs per width)
The measure, sample rules, spline, ablation means, bootstrap and the H1 verdicts are those of the
frozen code/conf01_token.py, unchanged. The only additions: the ablation hook also covers layer 4
(mean of L4's attention-output input over positions, from the same 30 train paragraphs, seed 6),
and two more conditions, L4H14 ablated and L4H14 + L5H7 ablated. Each condition re-seeds its
bootstrap (numpy seed 0) over the same paragraphs, so the resamples are identical across
conditions and differences between conditions are paired.
Measure: coefficient of tokens-on-the-line in the newline log-odds at word-end decision points
(<= 30 characters remaining, line 3 on), given a linear spline in characters remaining with
knots 2, 5, 8, 12, 16, 24, 32. Paragraphs: the first 60 per width, in seed-100 order, that wrap
with >= 5 lines and tokenise to <= 700 tokens. 2,000 paragraph bootstrap resamples.
"""
import sys, json, numpy as np, torch
from common import *
from newtext import load_paras

SPLIT = sys.argv[1]; SMOKE = "--smoke" in sys.argv
assert SPLIT in ("train", "conf-pythia410m")
NPARA = 5 if SMOKE else 60
NBOOT = 50 if SMOKE else 2000
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
cap = {}
hs_ = [layers[l].attention.dense.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(l, a[0].detach())) for l in (4, 5)]
zs = {4: [], 5: []}
with torch.no_grad():
    for p in train_paragraphs(seed=6)[:30]:
        model.gpt_neox(torch.tensor([tok(p, add_special_tokens=False)["input_ids"][:500]], device="cuda"))
        for l in (4, 5): zs[l].append(cap[l][0].mean(0))
for h_ in hs_: h_.remove()
zmean = {l: torch.stack(zs[l]).mean(0) for l in (4, 5)}
ABL = {"heads": []}
def abl(l):
    def f(m, a):
        hs = [h for (ll, h) in ABL["heads"] if ll == l]
        if not hs: return None
        z = a[0].clone()
        for h in hs: z[..., h * dh:(h + 1) * dh] = zmean[l][h * dh:(h + 1) * dh]
        return (z,)
    return f
for l in (4, 5): layers[l].attention.dense.register_forward_pre_hook(abl(l))
CONDS = {"intact": [], "abl_L5H7": [(5, 7)], "abl_L5H3": [(5, 3)], "abl_L5H11": [(5, 11)], "abl_L5H14": [(5, 14)],
         "abl_L4H14": [(4, 14)], "abl_L4H14_L5H7": [(4, 14), (5, 7)]}
def spline(rem): return np.c_[rem, *[np.maximum(rem - k, 0) for k in [2, 5, 8, 12, 16, 24, 32]]]
pct = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

res, boots = {}, {}
paras = load_paras(SPLIT, seed=100)
for W in (40, 60, 80):
    items = []
    for p in paras:
        if len(items) >= NPARA: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        items.append((ids, recs))
    res[W], boots[W] = {}, {}
    for cname, heads in CONDS.items():
        ABL["heads"] = heads
        X, y, g = [], [], []
        for k, (ids, recs) in enumerate(items):
            with torch.no_grad():
                lp = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0].float(), -1)[:, nl].cpu().numpy()
            for r in recs:
                if r["remaining"] > 30: continue
                v = lp[r["i"]]; X.append((r["remaining"], r["ntok_line"])); y.append(v - np.log1p(-np.exp(min(v, -1e-6)))); g.append(k)
        X = np.array(X, float); y = np.array(y); g = np.array(g)
        M = np.c_[np.ones(len(y)), spline(X[:, 0]), X[:, 1]]
        w = np.linalg.lstsq(M, y, rcond=None)[0]
        br = np.random.default_rng(0); bs = []
        groups = [np.where(g == k)[0] for k in range(len(items))]
        for _ in range(NBOOT):
            ii = np.concatenate([groups[j] for j in br.integers(0, len(items), len(items))])
            bs.append(np.linalg.lstsq(M[ii], y[ii], rcond=None)[0][-1])
        boots[W][cname] = np.array(bs)
        res[W][cname] = dict(n_points=len(y), n_paragraphs=len(items), token_coef=float(w[-1]), ci95=pct(bs))
        print(W, cname, json.dumps(res[W][cname]), flush=True)
ABL["heads"] = []
v, paired = {}, {}
for W in res:
    i = res[W]["intact"]; a = res[W]["abl_L5H7"]; b = res[W]["abl_L4H14"]
    d14 = boots[W]["abl_L4H14"] - boots[W]["intact"]; dj = boots[W]["abl_L4H14_L5H7"] - boots[W]["abl_L5H7"]
    paired[W] = dict(L4H14_minus_intact=dict(diff=b["token_coef"] - i["token_coef"], ci95=pct(d14)),
                     joint_minus_L5H7=dict(diff=res[W]["abl_L4H14_L5H7"]["token_coef"] - a["token_coef"], ci95=pct(dj)))
    v[W] = dict(H1a=bool(i["ci95"][0] > -0.10 and i["ci95"][1] < 0.10),
                H1b=bool(a["token_coef"] >= 0.30 and a["ci95"][0] > 0.20),
                H1c=bool(all(abs(res[W][c]["token_coef"] - i["token_coef"]) < 0.10 for c in ("abl_L5H3", "abl_L5H11", "abl_L5H14"))),
                H3=bool(paired[W]["L4H14_minus_intact"]["diff"] >= 0.10 and paired[W]["L4H14_minus_intact"]["ci95"][0] > 0))
R1 = all(v[W]["H1a"] and v[W]["H1b"] and v[W]["H1c"] for W in v)
R1a_w40 = v[40]["H1a"]
R3_w40_residual = bool(res[40]["intact"]["ci95"][0] > 0)
H3 = all(v[W]["H3"] for W in v)
out = dict(split=SPLIT, smoke=SMOKE, results=res, paired=paired, verdicts=v, R1_pass=R1, R1a_width40_pass=R1a_w40, residual_w40_above_zero=R3_w40_residual, H3_pass=H3)
save_json(out, f"c3_01_token_{SPLIT}{'_smoke' if SMOKE else ''}.json")
print(json.dumps(dict(paired=paired, verdicts=v, R1_pass=R1, H3_pass=H3), indent=1))
