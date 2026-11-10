"""Builds the round-4 result tables and verdicts (PROTOCOL-4.md) from the saved outputs of p4_01..p4_04 on the
round-4 part, and checks that every table line appears verbatim in REPORT-4.md (exit 1 if not).
Bootstrap: 2,000 resamples, seed 0, over items or paragraphs, paired where stated.
Run: python3 report4_tables.py [--check] [--split train|conf4-pythia410m] [--suffix _smoke]"""
import sys, json, numpy as np, torch
from pathlib import Path
R = Path(__file__).resolve().parent.parent; O = R / "outputs"
SPLIT = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "conf4-pythia410m"
SUF = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ""
NB = 2000
f3 = lambda x: f"{x:.3f}"; ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
fci = lambda c: f"({c[0]:.3f}, {c[1]:.3f})"; pf = lambda b: "pass" if b else "fail"
L = []; V = {}

# ---------------- D
d1 = json.loads((O / f"p4_01_decision_{SPLIT}{SUF}.json").read_text()); S = d1["summary"]; rows = d1["rows"]
pt = torch.load(O / f"p4_01_decision_{SPLIT}{SUF}_means.pt")
setn = list(S["mediation_sets"]["C"].keys()); setn = [s for s in setn if s != "random_mlp3_mean"]
n = len(rows); rng = np.random.default_rng(0); IDX = [rng.integers(0, n, n) for _ in range(NB)]
E = {c: np.array([r[c] for r in rows]) for c in ("C", "Wd")}
F = {c: np.array([r[f"frozen_{c}"] for r in rows]) for c in ("C", "Wd")}
def share(c, s, idx): return 1 - F[c][idx, setn.index(s)].mean() / E[c][idx].mean()
def rshare(c, idx): return np.mean([share(c, f"random_mlp3_{q}", idx) for q in range(10)])
full = np.arange(n)
b14 = np.array([share("C", "MLP14_16_18", i) for i in IDX]); bdiff = np.array([share("C", "MLP14_16_18", i) - rshare("C", i) for i in IDX])
p14 = share("C", "MLP14_16_18", full); pr = rshare("C", full)
V["D1"] = p14 >= 0.11 and ci(b14)[0] > 0 and ci(bdiff)[0] > 0
D2 = {c: (share(c, "MLP21_23", full), ci(np.array([share(c, "MLP21_23", i) for i in IDX]))) for c in ("C", "Wd")}
V["D2"] = all(v[0] <= -0.10 and v[1][1] < 0 for v in D2.values())
dR = pt["dR"]  # cond -> [n, 19, D]; row k is after layer 5 + k
def cosL(idx, k): a = dR["C"][idx, k].mean(0); b = dR["Wd"][idx, k].mean(0); return float(torch.nn.functional.cosine_similarity(a, b, dim=0))
c6, c12 = cosL(full, 1), cosL(full, 7)
tidx = [torch.as_tensor(i) for i in IDX]
bc = np.array([cosL(i, 7) - cosL(i, 1) for i in tidx])
V["D3"] = (c12 - c6) >= 0.20 and ci(bc)[0] > 0 and c12 >= 0.65
names = d1["names"]; heads = [i for i, x in enumerate(names) if "H" in x]
dla = {c: np.array(d1["dla"][c]) for c in ("C", "Wd")}
top = {c: sorted(heads, key=lambda i: -dla[c][i])[:2] for c in dla}
V["D4"] = all(names[top[c][0]] == "L17H1" for c in dla)
V["D"] = all(V[k] for k in ("D1", "D2", "D3", "D4"))
L += [f"**D, the decision step.** Items: {n}. Effects on log P(newline): COUNT {f3(E['C'].mean())}, WIDTH {f3(E['Wd'].mean())}, both {f3(S['eff']['CW'])} (interaction {f3(S['interaction'])}).", "",
      "| Part | Measure | Value (95% CI) | Threshold | Verdict |", "|---|---|---|---|---|",
      f"| D1 | COUNT share lost, MLPs 14, 16, 18 frozen | {f3(p14)} {fci(ci(b14))} | >= 0.11, lower bound > 0 | {pf(p14 >= 0.11 and ci(b14)[0] > 0)} |",
      f"| D1 | minus the mean of 10 random MLP triples ({f3(pr)}) | {f3(p14 - pr)} {fci(ci(bdiff))} | lower bound > 0 | {pf(ci(bdiff)[0] > 0)} |",
      f"| D2 | COUNT share lost, MLPs 21, 23 frozen | {f3(D2['C'][0])} {fci(D2['C'][1])} | <= -0.10, upper bound < 0 | {pf(D2['C'][0] <= -0.10 and D2['C'][1][1] < 0)} |",
      f"| D2 | WIDTH share lost, MLPs 21, 23 frozen | {f3(D2['Wd'][0])} {fci(D2['Wd'][1])} | <= -0.10, upper bound < 0 | {pf(D2['Wd'][0] <= -0.10 and D2['Wd'][1][1] < 0)} |",
      f"| D3a | cosine after layer 12 minus after layer 6 ({f3(c12)} - {f3(c6)}) | {f3(c12 - c6)} {fci(ci(bc))} | >= 0.20, lower bound > 0 | {pf((c12 - c6) >= 0.20 and ci(bc)[0] > 0)} |",
      f"| D3b | cosine after layer 12 | {f3(c12)} | >= 0.65 | {pf(c12 >= 0.65)} |",
      f"| D4 | largest head direct effect, COUNT: {names[top['C'][0]]} {f3(dla['C'][top['C'][0]])} (next {names[top['C'][1]]} {f3(dla['C'][top['C'][1]])}) | | L17H1 | {pf(names[top['C'][0]] == 'L17H1')} |",
      f"| D4 | largest head direct effect, WIDTH: {names[top['Wd'][0]]} {f3(dla['Wd'][top['Wd'][0]])} (next {names[top['Wd'][1]]} {f3(dla['Wd'][top['Wd'][1]])}) | | L17H1 | {pf(names[top['Wd'][0]] == 'L17H1')} |",
      "", f"D overall: {pf(V['D'])}", ""]
cos_line = ", ".join(f"{k.replace('after ', '')} {v:.2f}" for k, v in S["cos_layer"].items())
L += [f"Cosine between the COUNT and WIDTH residual changes at the decision token, by layer: {cos_line}.", ""]
ms = S["mediation_sets"]
L += ["| Frozen at the decision token | COUNT share lost | WIDTH share lost |", "|---|---|---|"]
for s in ("layers6-11", "layers12-17", "layers18-23", "all_heads", "all_mlps", "readers3", "L17H1", "MLP14_16_18", "MLP21_23", "random_mlp3_mean"):
    L.append(f"| {s} | {f3(ms['C'][s])} | {f3(ms['Wd'][s])} |")
L.append("")

# ---------------- S
d2 = json.loads((O / f"p4_02_sufficiency_{SPLIT}{SUF}.json").read_text()); raw = d2["raw"]
NR = (len(raw["sep"]) - 1) // 3 - 2
idx_of = lambda scope, k: 1 + {"DEC": 0, "EARLY": 1, "BOTH": 2}[scope] * (2 + NR) + k  # k: 0 mean, 1 resample, 2.. random
def sep_of(e, pidx=None):
    br, nb = np.array(e["br"]), np.array(e["nb"])
    if pidx is None: return br.mean() - nb.mean()
    pb, pn = np.array(e["pbr"]), np.array(e["pnb"])
    wb = np.bincount(pb, minlength=npar)[pidx].sum(); wn = np.bincount(pn, minlength=npar)[pidx].sum()
    sb = sum(br[pb == q].sum() for q in pidx); sn = sum(nb[pn == q].sum() for q in pidx)
    return sb / wb - sn / wn
from statistics import mean
npar = max(max(e["psc"]) for e in raw["sep"]) + 1
def auc_of(e):
    sc = np.array(e["sc"]); lab = np.array(e["lab"]); pos = sc[lab == 1]; neg = sc[lab == 0]
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
# paragraph-level sums for a fast bootstrap
def psums(e):
    br, nb = np.array(e["br"]), np.array(e["nb"]); pb, pn = np.array(e["pbr"]), np.array(e["pnb"])
    return np.bincount(pb, br, npar), np.bincount(pb, minlength=npar), np.bincount(pn, nb, npar), np.bincount(pn, minlength=npar)
rngp = np.random.default_rng(0); PIDX = [rngp.integers(0, npar, npar) for _ in range(NB)]
def sep_boot(e):
    sb, cb, sn, cn = psums(e)
    return np.array([sb[i].sum() / cb[i].sum() - sn[i].sum() / cn[i].sum() for i in PIDX])
W0 = np.array(raw["width"][0]); ni = len(W0); rngi = np.random.default_rng(0); IIDX = [rngi.integers(0, ni, ni) for _ in range(NB)]
def kept_w(k): Wk = np.array(raw["width"][k]); return Wk.mean() / W0.mean(), np.array([Wk[i].mean() / W0[i].mean() for i in IIDX])
sep0 = sep_of(raw["sep"][0]); bsep0 = sep_boot(raw["sep"][0]); auc0 = auc_of(raw["sep"][0])
a_dec = auc_of(raw["sep"][idx_of("DEC", 0)])
kw, bkw = kept_w(idx_of("DEC", 0)); rmax_w = max(kept_w(idx_of("DEC", 2 + q))[0] for q in range(NR))
V["S1"] = a_dec >= auc0 - 0.02 and kw >= 0.33 and ci(bkw)[0] > rmax_w
ks = sep_of(raw["sep"][idx_of("BOTH", 0)]) / sep0; bks = sep_boot(raw["sep"][idx_of("BOTH", 0)]) / bsep0
rmax_s = max(sep_of(raw["sep"][idx_of("BOTH", 2 + q)]) / sep0 for q in range(NR))
V["S2"] = ks >= 0.15 and ci(bks)[0] > rmax_s
V["S"] = V["S1"] and V["S2"]
L += [f"**S, sufficiency.** Paragraphs: {npar}; width items: {ni}. Intact separation {f3(sep0)}, AUROC {f3(auc0)}, width effect {f3(W0.mean())}.", "",
      "| Part | Measure | Value (95% CI) | Threshold | Verdict |", "|---|---|---|---|---|",
      f"| S1a | AUROC, late circuit only at the decision token (mean ablation) | {f3(a_dec)} | >= {f3(auc0 - 0.02)} | {pf(a_dec >= auc0 - 0.02)} |",
      f"| S1b | width effect kept, same | {f3(kw)} {fci(ci(bkw))} | >= 0.33, lower bound > {f3(rmax_w)} (largest random circuit) | {pf(kw >= 0.33 and ci(bkw)[0] > rmax_w)} |",
      f"| S2 | separation kept, whole circuit (mean ablation) | {f3(ks)} {fci(ci(bks))} | >= 0.15, lower bound > {f3(rmax_s)} (largest random circuit) | {pf(ks >= 0.15 and ci(bks)[0] > rmax_s)} |",
      "", f"S overall: {pf(V['S'])}", "",
      "| Scope | Ablation | Separation kept | AUROC | Width effect kept |", "|---|---|---|---|---|"]
for scope in ("DEC", "EARLY", "BOTH"):
    for k, nm in ((0, "mean"), (1, "resample")):
        e = raw["sep"][idx_of(scope, k)]
        L.append(f"| {scope} | {nm} | {f3(sep_of(e) / sep0)} | {f3(auc_of(e))} | {f3(kept_w(idx_of(scope, k))[0])} |")
    rs_ = [sep_of(raw["sep"][idx_of(scope, 2 + q)]) / sep0 for q in range(NR)]; rw_ = [kept_w(idx_of(scope, 2 + q))[0] for q in range(NR)]
    L.append(f"| {scope} | {NR} random circuits, mean ablation | {f3(min(rs_))} to {f3(max(rs_))} | | {f3(min(rw_))} to {f3(max(rw_))} |")
L.append("")

# ---------------- G
g = json.loads((O / f"p4_03_geometry_{SPLIT}{SUF}.json").read_text())
G = {"G1": True, "G2": True, "G3": True}
L += ["**G, the shape of the count** (split-half; half B's class means in half A's top directions).", "",
      "| Width | After layer | Top direction: variance share | its linear R^2 | Second direction: linear / spline R^2 | its reliability | Overall linear R^2, labels shuffled |", "|---|---|---|---|---|---|---|"]
for W in ("40", "60"):
    for l in ("5", "6"):
        r = g["split_half"][W][l]["real"]; s = g["split_half"][W][l]["shuffled"]
        L.append(f"| {W} | {l} | {f3(r['var_share_B'][0])} | {f3(r['r2_linear_by_pc'][0])} | {f3(r['r2_linear_by_pc'][1])} / {f3(r['r2_spline_by_pc'][1])} | {f3(r['reliability_by_pc'][1])} | {f3(s['r2_linear'])} |")
    r5 = g["split_half"][W]["5"]; r6 = g["split_half"][W]["6"]["real"]
    G["G1"] &= r5["real"]["var_share_B"][0] >= 0.50 and r5["real"]["r2_linear_by_pc"][0] >= 0.85 and r5["shuffled"]["r2_linear"] < 0.20
    G["G2"] &= (r6["r2_spline_by_pc"][1] - r6["r2_linear_by_pc"][1]) >= 0.30 and r6["reliability_by_pc"][1] >= 0.70
L += ["", "| Width | Head | Peak at characters remaining | Monotone fit R^2 |", "|---|---|---|---|"]
for W in ("40", "60"):
    for h in ("L6H2", "L8H15", "L17H1", "L5H6", "L5H7", "L8H11"):
        a = g["attention"][W][h]
        L.append(f"| {W} | {h} | {a['peak_at_remaining']} | {f3(a['monotone_r2'])} |")
        if h in ("L6H2", "L8H15", "L17H1"): G["G3"] &= a["monotone_r2"] >= 0.90 and a["peak_at_remaining"] <= 5
V.update(G); V["G"] = all(G.values())
L += ["", f"G1 {pf(G['G1'])}, G2 {pf(G['G2'])}, G3 {pf(G['G3'])}; G overall: {pf(V['G'])}", ""]

# ---------------- X
xg = json.loads((O / f"p4_04_roles_gpt2_{SPLIT}{SUF}.json").read_text())["summary"]
xp = json.loads((O / f"p4_04_roles_pythia-410m_{SPLIT}{SUF}.json").read_text())["summary"]
def best(x, key): return max(v[0] for v in x[key])
def dil(x): return max([h["share"] for h in x["dilution_heads"] if h["r_used"] <= -0.5], default=0.0)
pg, pp = max(xg["probe_used"]), max(xp["probe_used"])
cg, cp = best(xg, "count_shift_by_layer"), best(xp, "count_shift_by_layer")
wg, wp = xg["width_behaviour"][0], xp["width_behaviour"][0]
dg, dp = dil(xg), dil(xp)
w8g = max(abs(v[0]) for v in xg["width_patch_by_layer"][8:12]); w8p = xp["width_patch_by_layer"][8][0]
V["X1"] = pg >= 0.70; V["X2"] = cg <= 0.5 * cp; V["X3"] = wg <= 0.5 * wp; V["X4"] = dg <= 0.5 * dp and dp >= 0.60; V["X5"] = w8g < 0.10 and w8p >= 0.25
V["X"] = all(V[k] for k in ("X1", "X2", "X3", "X4", "X5"))
L += ["**X, GPT-2 Small against Pythia-410M.**", "", "| Part | Measure | GPT-2 Small | Pythia-410M | Threshold | Verdict |", "|---|---|---|---|---|---|",
      f"| X1 | best-layer probe R^2, characters used | {f3(pg)} | {f3(pp)} | GPT-2 >= 0.70 | {pf(V['X1'])} |",
      f"| X2 | largest count-shift effect over layers | {f3(cg)} | {f3(cp)} | GPT-2 <= 0.5 x Pythia | {pf(V['X2'])} |",
      f"| X3 | behavioural width effect | {f3(wg)} | {f3(wp)} | GPT-2 <= 0.5 x Pythia | {pf(V['X3'])} |",
      f"| X4 | largest line-start share among counting heads | {f3(dg)} | {f3(dp)} | GPT-2 <= 0.5 x Pythia; Pythia >= 0.60 | {pf(V['X4'])} |",
      f"| X5 | width patch at the newline: GPT-2 after layers 8-11 (largest absolute), Pythia after layer 8 | {f3(w8g)} | {f3(w8p)} | GPT-2 < 0.10; Pythia >= 0.25 | {pf(V['X5'])} |",
      "", f"X overall: {pf(V['X'])}", ""]
for nm, x in (("GPT-2 Small", xg), ("Pythia-410M", xp)):
    L.append(f"{nm}: AUROC width 40 {f3(x['behaviour']['40']['auroc'])}, width 60 {f3(x['behaviour']['60']['auroc'])}; token coefficient {f3(x['behaviour']['40']['token_coef'])} / {f3(x['behaviour']['60']['token_coef'])}; width items {x['n_items']}.")
txt = "\n".join(L)
(O / f"report4_tables_{SPLIT}{SUF}.md").write_text(txt + "\n")
(O / f"report4_verdicts_{SPLIT}{SUF}.json").write_text(json.dumps({k: bool(v) for k, v in V.items()}, indent=1))
if "--check" in sys.argv:
    doc = (R / "REPORT-4.md").read_text(); missing = [l for l in L if l.strip() and l not in doc]
    print("missing lines:", missing if missing else "none"); sys.exit(1 if missing else 0)
print(txt); print(json.dumps({k: bool(v) for k, v in V.items()}))
