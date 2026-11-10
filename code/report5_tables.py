"""report5_tables (round 5): tables and verdicts for PROTOCOL-5.md from the saved outputs. Only formats and tests
saved outputs; thresholds are those frozen in PROTOCOL-5.md (constants below).
Usage: python report5_tables.py conf     -> outputs/report5_tables_conf.md and report5_verdicts_conf.json
       python report5_tables.py train    -> the same on the exploration/pilot outputs (reference values)
       --check : regenerate and compare with the saved table file (exit 1 on any difference)."""
import os, sys, json, numpy as np
from pathlib import Path
OUT = Path(os.environ.get("P5_OUT") or Path(__file__).resolve().parent.parent / "outputs")  # P5_OUT: a test folder (pipeline checks only)
MODE = sys.argv[1]; assert MODE in ("conf", "train"); CHECK = "--check" in sys.argv
NB = 2000
def pct(b): return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def f3(x): return f"{x:.3f}"
def ci(c): return f"({c[0]:.3f}, {c[1]:.3f})"
L = []; V = {}
def H(s): L.append(s)

# ------------------------------------------------------------------ T: the token bias
TOK = OUT / ("p5_02_tokenbias_conf.npz" if MODE == "conf" else "exp28_tokenbias.npz")
z = np.load(TOK); C = {c: i for i, c in enumerate(z["cols"])}
ENG = "reserve-pythia410m" if MODE == "conf" else "train"
CODE, DE = ("conf5-code", "conf5-de") if MODE == "conf" else ("explore-code", "explore-de")
def spl(x, ks): return np.c_[x, *[np.maximum(x - k, 0) for k in ks]]
KR = (2, 5, 8, 12, 16, 24, 32); KU = (5, 10, 15, 20, 30, 40, 60, 80, 100)
def coef(d, y, xcol, ks):
    M = np.c_[np.ones(len(d)), spl(d[:, C[xcol]], ks), d[:, C["ntok"]]]; return float(np.linalg.lstsq(M, y, rcond=None)[0][-1])
def groups(d): g = d[:, C["para"]].astype(int); u = np.unique(g); return u, {k: np.where(g == k)[0] for k in u}
def boot_stat(d, fn, seed=0):
    u, idx = groups(d); rs = np.random.default_rng(seed); out = []
    for _ in range(NB):
        ii = np.concatenate([idx[k] for k in rs.choice(u, len(u))]); out.append(fn(d[ii]))
    return np.array(out)
def beh(col): return lambda d: coef(d[d[:, C["remaining"]] <= 30], d[d[:, C["remaining"]] <= 30][:, C[col]], "remaining", KR)
def att6(d): return coef(d, d[:, C["att6"]], "used", KU)
conds = [(ENG, w) for w in (20, 30, 40, 60, 80, 100, 120)] + [(CODE, 40), (CODE, 60), (DE, 40), (DE, 60)]
TT = {}
for s_, w in conds:
    d = z[f"{s_}__{w}"]; s = d[d[:, C["remaining"]] <= 30]
    bi = boot_stat(d, beh("lo")); ba = boot_stat(d, beh("lo_abl7")); bt = boot_stat(d, att6)
    TT[(s_, w)] = dict(n=len(s), npara=len(np.unique(d[:, C["para"]])), ntok=float(s[:, C["ntok"]].mean()), cpt=float((s[:, C["used"]] / s[:, C["ntok"]]).mean()),
                      intact=beh("lo")(d), intact_b=bi, abl=beh("lo_abl7")(d), abl_b=ba, att6=att6(d), att6_b=bt)
NAME = {ENG: "English", CODE: "code", DE: "German"}
H("## T: the token bias\n")
H("| Text | Width | Points | Mean tokens on the line | Characters per token | Intact (95% CI) | L5H7 ablated (95% CI) | Rise when ablated (95% CI) | L5H6 line-start attention per token (95% CI) |")
H("|---|---|---|---|---|---|---|---|---|")
T5 = []
for k, t in TT.items():
    diff = t["abl_b"] - t["intact_b"]; rise = t["abl"] - t["intact"]
    T5.append(bool(rise >= 0.10 and pct(diff)[0] > 0))
    H(f"| {NAME[k[0]]} | {k[1]} | {t['n']} | {t['ntok']:.1f} | {t['cpt']:.2f} | {f3(t['intact'])} {ci(pct(t['intact_b']))} | {f3(t['abl'])} {ci(pct(t['abl_b']))} | {f3(rise)} {ci(pct(diff))} | {t['att6']:.4f} ({pct(t['att6_b'])[0]:.4f}, {pct(t['att6_b'])[1]:.4f}) |")
a, b = TT[(ENG, 20)], TT[(ENG, 60)]
d1 = a["intact"] - b["intact"]; d1b = a["intact_b"] - b["intact_b"]
V["T1"] = bool(d1 >= 0.04 and pct(d1b)[0] > 0 and pct(a["intact_b"])[0] > 0)
xs = [t["ntok"] for t in TT.values()]; ys = [t["intact"] for t in TT.values()]
def rank(v): return np.argsort(np.argsort(v))
rho = float(np.corrcoef(rank(xs), rank(ys))[0, 1]); V["T2"] = bool(rho <= -0.50)
a3, b3 = TT[(ENG, 20)], TT[(ENG, 100)]; d3 = a3["abl"] - b3["abl"]; d3b = a3["abl_b"] - b3["abl_b"]
V["T3"] = bool(d3 >= 0.15 and pct(d3b)[0] > 0)
a4, b4 = TT[(ENG, 20)], TT[(ENG, 80)]; d4 = a4["att6"] - b4["att6"]; d4b = a4["att6_b"] - b4["att6_b"]
V["T4"] = bool(d4 <= -0.003 and pct(d4b)[1] < 0)
V["T5"] = all(T5)
H("")
H("| Part | Measure | Value (95% CI) | Threshold | Verdict |")
H("|---|---|---|---|---|")
H(f"| T1 | intact coefficient, English width 20 minus width 60 | {f3(d1)} {ci(pct(d1b))} | >= 0.04, lower bound > 0; width-20 lower bound > 0 ({pct(a['intact_b'])[0]:.3f}) | {'pass' if V['T1'] else 'fail'} |")
H(f"| T2 | rank correlation of the intact coefficient with mean tokens on the line, 11 text-width conditions | {rho:.3f} | <= -0.50 | {'pass' if V['T2'] else 'fail'} |")
H(f"| T3 | L5H7-ablated coefficient, English width 20 minus width 100 | {f3(d3)} {ci(pct(d3b))} | >= 0.15, lower bound > 0 | {'pass' if V['T3'] else 'fail'} |")
H(f"| T4 | L5H6 attention per token, English width 20 minus width 80 | {d4:.4f} ({pct(d4b)[0]:.4f}, {pct(d4b)[1]:.4f}) | <= -0.003, upper bound < 0 | {'pass' if V['T4'] else 'fail'} |")
H(f"| T5 | rise when L5H7 is ablated, in every condition (smallest {min(t['abl'] - t['intact'] for t in TT.values()):.3f}) | {sum(T5)} of {len(T5)} | each >= 0.10, lower bound > 0 | {'pass' if V['T5'] else 'fail'} |")
V["T"] = all(V[k] for k in ("T1", "T2", "T3", "T4", "T5"))
H(f"\nT overall: {'pass' if V['T'] else 'fail'}\n")

# ------------------------------------------------------------------ M: the damping
MJ = json.load(open(OUT / ("p5_03_damping_reserve-pythia410m.json" if MODE == "conf" else "p5_03_damping_train.json")))
rows = MJ["rows"]; n = len(rows); rs = np.random.default_rng(0); IDX = [rs.integers(0, n, n) for _ in range(NB)]
def share_b(src, key, base="e"):
    E = np.array([r[src][base] for r in rows]); F = np.array([r[src][key] for r in rows])
    return float(1 - F.mean() / E.mean()), pct([1 - F[i].mean() / E[i].mean() for i in IDX])
H("## M: the late damping\n")
H(f"Items: {n}. Share of each source's effect lost when the named components are frozen at the decision token (negative: they work against the source).\n")
H("| Source | Effect | MLPs 21, 23 frozen | same, final LayerNorm frozen | MLP23.736 and MLP21.1410 only | 20 damping neurons |")
H("|---|---|---|---|---|---|")
M = {}
for src, nm in (("C", "COUNT"), ("Wd", "WIDTH")):
    m = share_b(src, "frozen_mlp"); mln = share_b(src, "frozen_both", "frozen_ln"); t2 = share_b(src, "frozen_top2"); t20 = share_b(src, "frozen_top20")
    E = float(np.mean([r[src]["e"] for r in rows])); M[src] = dict(mlp=m, mln=mln, top2=t2, top20=t20, E=E)
    H(f"| {nm} | {E:.3f} | {f3(m[0])} {ci(m[1])} | {f3(mln[0])} {ci(mln[1])} | {f3(t2[0])} {ci(t2[1])} | {f3(t20[0])} {ci(t20[1])} |")
V["M1"] = all(M[s]["mln"][0] <= -0.15 and M[s]["mln"][1][1] < 0 and abs(M[s]["mln"][0] - M[s]["mlp"][0]) <= 0.10 for s in M)
V["M2"] = all(M[s]["top2"][0] <= -0.15 and M[s]["top2"][1][1] < 0 and M[s]["top2"][0] / M[s]["mlp"][0] >= 0.5 for s in M)
NZ = np.load(OUT / ("p5_03_damping_reserve-pythia410m_natural.npz" if MODE == "conf" else "p5_03_damping_train_natural.npz"))
R = NZ["rec"]; conds_n = list(NZ["conds"]); RC = {c: i for i, c in enumerate(NZ["cols"])}
def nat(cond, isnl, col="dlogp"):
    s = R[(R[:, RC["cond"]] == conds_n.index(cond)) & (R[:, RC["is_newline"]] == isnl) & (R[:, RC["p_top"]] >= 0.2) & (R[:, RC["p_top"]] < 0.9)]
    g = s[:, RC["para"]].astype(int); u = np.unique(g); idx = {k: np.where(g == k)[0] for k in u}; rr = np.random.default_rng(0)
    bb = [s[np.concatenate([idx[k] for k in rr.choice(u, len(u))]), RC[col]].mean() for _ in range(NB)]
    return len(s), float(s[:, RC[col]].mean()), pct(bb)
H("\nNatural text, every position whose intact top token has probability 0.2 to 0.9: mean change in that token's log-probability (and in entropy) when the named components are mean-ablated at every position.\n")
H("| Ablated | Top token | Positions | Change in log-probability (95% CI) | Change in entropy (95% CI) |")
H("|---|---|---|---|---|")
NAT = {}
for cond, lab in (("top2_abl", "MLP23.736 and MLP21.1410"), ("neur_abl", "20 damping neurons"), ("mlp_abl", "MLPs 21 and 23")):
    for isnl in (1, 0):
        k, mlp_, c_ = nat(cond, isnl); _, me, ce = nat(cond, isnl, "dent"); NAT[(cond, isnl)] = (k, mlp_, c_, me, ce)
        H(f"| {lab} | {'newline' if isnl else 'other'} | {k} | {f3(mlp_)} {ci(c_)} | {f3(me)} {ci(ce)} |")
nl_, ot_ = NAT[("top2_abl", 1)], NAT[("top2_abl", 0)]
V["M3"] = bool(nl_[1] >= 0.15 and nl_[2][0] > 0 and abs(ot_[1]) <= 0.10 and ot_[2][0] >= -0.15 and ot_[2][1] <= 0.15)
cr = MJ["corr"]; V["M4"] = all(v >= 0.5 for v in cr.values())
H("")
H("| Part | Measure | Value | Threshold | Verdict |")
H("|---|---|---|---|---|")
H(f"| M1 | damping by MLPs 21 and 23 with the final LayerNorm frozen: COUNT {f3(M['C']['mln'][0])}, WIDTH {f3(M['Wd']['mln'][0])} (free: {f3(M['C']['mlp'][0])}, {f3(M['Wd']['mlp'][0])}) | | each <= -0.15, upper bound < 0, within 0.10 of the free value | {'pass' if V['M1'] else 'fail'} |")
H(f"| M2 | MLP23.736 and MLP21.1410 frozen alone: COUNT {f3(M['C']['top2'][0])}, WIDTH {f3(M['Wd']['top2'][0])}; as a share of the whole MLPs' damping {M['C']['top2'][0] / M['C']['mlp'][0]:.2f}, {M['Wd']['top2'][0] / M['Wd']['mlp'][0]:.2f} | | each <= -0.15, upper bound < 0, >= 0.5 of the whole | {'pass' if V['M2'] else 'fail'} |")
H(f"| M3 | the two neurons ablated, natural text: newline {f3(nl_[1])} {ci(nl_[2])}; other tokens {f3(ot_[1])} {ci(ot_[2])} | | newline >= 0.15, lower bound > 0; other within 0.10, interval within 0.15 | {'pass' if V['M3'] else 'fail'} |")
H(f"| M4 | correlation of activation with the newline log-odds at decision points ({MJ['n_corr_points']} points): " + ", ".join(f"{k} {v:.3f}" for k, v in cr.items()) + f" | | each >= 0.50 | {'pass' if V['M4'] else 'fail'} |")
V["M"] = all(V[k] for k in ("M1", "M2", "M3", "M4"))
H(f"\nM overall: {'pass' if V['M'] else 'fail'}\n")

# ------------------------------------------------------------------ S: sufficiency of a fuller circuit
SJ = json.load(open(OUT / ("p5_01_sufficiency_reserve-pythia410m.json" if MODE == "conf" else "p5_01_sufficiency_train.json")))
RAW = SJ["raw"]; RECS = SJ["recs"]; NPA = len(RECS)
LO0 = [np.array(x) for x in RAW["intact"]["lo"]]; W0 = np.array(RAW["intact"]["width"])
def s_stats(key, pi, wi):
    lo = [np.array(x) for x in RAW[key]["lo"]]; w = np.array(RAW[key]["width"])
    a = np.concatenate([lo[i] for i in pi]); b = np.concatenate([LO0[i] for i in pi])
    fidv = 1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum()
    def sep(L_):
        br = [v for i in pi for r, v in zip(RECS[i], L_[i]) if r["label"] == 1]
        nb = [v for i in pi for r, v in zip(RECS[i], L_[i]) if r["label"] == 0 and r["remaining"] <= 12]
        return np.mean(br) - np.mean(nb)
    return float(fidv), float(sep(lo) / sep(LO0)), float(w[wi].mean() / W0[wi].mean())
rsb = np.random.default_rng(0); PB = [rsb.integers(0, NPA, NPA) for _ in range(NB)]; WB = [rsb.integers(0, len(W0), len(W0)) for _ in range(NB)]
ALLP = np.arange(NPA); ALLW = np.arange(len(W0))
keys = [k for k in RAW if k != "intact"]
SS = {k: s_stats(k, ALLP, ALLW) for k in keys}
SB = {k: np.array([s_stats(k, PB[b], WB[b]) for b in range(NB)]) for k in ("plus24|mean", "circuit|mean", "plus24|resample", "circuit|resample")}
H("## S: is a fuller circuit enough?\n")
H(f"Paragraphs: {NPA}; width items: {len(W0)}. Intact separation {SJ['intact_sep']:.3f}, AUROC {SJ['intact_auroc']:.3f}, width effect {SJ['intact_width']:.3f}. Fidelity: 1 minus the circuit's squared error in the newline log-odds over all decision points, relative to the intact model's variance.\n")
H("| Set | Ablation | Fidelity | Separation kept | AUROC | Width effect kept |")
H("|---|---|---|---|---|---|")
LAB = dict(circuit="round-4 circuit", plus4="+ 4 components", plus8="+ 8", plus16="+ 16", plus24="+ 24 (fuller circuit)")
for nm in ("circuit", "plus4", "plus8", "plus16", "plus24"):
    for mode in ("mean", "resample"):
        k = f"{nm}|{mode}"
        if k in SS: H(f"| {LAB[nm]} | {mode} | {SS[k][0]:.3f} | {SS[k][1]:.3f} | {SJ[k]['auroc']:.3f} | {SS[k][2]:.3f} |")
rnd = {mode: [SS[k] for k in keys if k.startswith("random") and k.endswith(mode)] for mode in ("mean", "resample")}
for mode in ("mean", "resample"):
    r = np.array(rnd[mode]); H(f"| 5 random sets, + 24 | {mode} | {r[:, 0].min():.3f} to {r[:, 0].max():.3f} | {r[:, 1].min():.3f} to {r[:, 1].max():.3f} | | {r[:, 2].min():.3f} to {r[:, 2].max():.3f} |")
rm = np.array(rnd["mean"])
f24, c24 = SB["plus24|mean"], SB["circuit|mean"]; f24r, c24r = SB["plus24|resample"], SB["circuit|resample"]
V["S1"] = bool(SS["plus24|mean"][0] >= 0.50 and pct(f24[:, 0])[0] > rm[:, 0].max())
dsep = SS["plus24|mean"][1] - SS["circuit|mean"][1]; dsepb = f24[:, 1] - c24[:, 1]
V["S2"] = bool(dsep >= 0.20 and pct(dsepb)[0] > 0)
V["S3"] = bool(SS["plus24|mean"][2] >= 0.40 and pct(f24[:, 2])[0] > rm[:, 2].max())
V["S4"] = bool(SS["plus24|resample"][0] >= 0.15 and SS["plus24|resample"][1] >= 0.25 and pct(f24r[:, 0])[0] > SS["circuit|resample"][0] and pct(f24r[:, 1])[0] > SS["circuit|resample"][1])
H("")
H("| Part | Measure | Value (95% CI) | Threshold | Verdict |")
H("|---|---|---|---|---|")
H(f"| S1 | fidelity, fuller circuit, mean ablation | {SS['plus24|mean'][0]:.3f} {ci(pct(f24[:, 0]))} | >= 0.50, lower bound > largest random ({rm[:, 0].max():.3f}) | {'pass' if V['S1'] else 'fail'} |")
H(f"| S2 | separation kept, fuller minus round-4 circuit, mean ablation | {dsep:.3f} {ci(pct(dsepb))} | >= 0.20, lower bound > 0 | {'pass' if V['S2'] else 'fail'} |")
H(f"| S3 | width effect kept, fuller circuit, mean ablation | {SS['plus24|mean'][2]:.3f} {ci(pct(f24[:, 2]))} | >= 0.40, lower bound > largest random ({rm[:, 2].max():.3f}) | {'pass' if V['S3'] else 'fail'} |")
H(f"| S4 | resample ablation, fuller circuit: fidelity {SS['plus24|resample'][0]:.3f} {ci(pct(f24r[:, 0]))}, separation kept {SS['plus24|resample'][1]:.3f} {ci(pct(f24r[:, 1]))} | | fidelity >= 0.15 and separation >= 0.25, lower bounds above the round-4 circuit's ({SS['circuit|resample'][0]:.3f}, {SS['circuit|resample'][1]:.3f}) | {'pass' if V['S4'] else 'fail'} |")
V["S"] = all(V[k] for k in ("S1", "S2", "S3", "S4"))
H(f"\nS overall: {'pass' if V['S'] else 'fail'}\n")

# ------------------------------------------------------------------ G: other widths and text types
GC = [(ENG, 20), (ENG, 30), (ENG, 100), (ENG, 120), (CODE, 40), (CODE, 60), (DE, 40), (DE, 60)]
# predicted pattern for the width roles (exploration): R holds / S holds; "ns" = not specific (random sets also remove it)
PRED_R = {20: False, 30: True, 100: True, 120: True, (CODE, 40): False, (CODE, 60): False, (DE, 40): False, (DE, 60): True}
PRED_S = {20: False, 30: False, 100: False, 120: False, (CODE, 40): False, (CODE, 60): False, (DE, 40): False, (DE, 60): True}
def pk(s_, w): return w if s_ == ENG else (s_, w)
def holds(r, role):
    if role == "C": c = r["C"]; return bool(c["share"] >= 0.30 and c["share_ci95"][0] > 0 and c["random_mean"] < 0.15)
    if role == "T":
        t = list(r["T"]["widths"].values())[0]; return bool(t["diff"] >= 0.10 and t["diff_ci95"][0] > 0 and t["random_mean"] - t["intact"] < 0.10)
    if role == "S":
        x = r["S"]; return bool(x["width_effect_ci95"][0] > 0 and x["share"] >= 0.15 and x["share_ci95"][0] > 0 and x["random_mean"] < 0.10)
    if role == "R":
        x = r["R"]; return bool(x["patch_effect_ci95"][0] > 0 and x["share"] >= 0.25 and x["share_ci95"][0] > 0 and x["random_mean"] < 0.15)
H("## G: other widths and text types\n")
H("Role battery of the cross-model study (counters C, token corrector T, stored width S, readers R), at one width per run. Values: the role's share (C, S, R) or the rise in the token coefficient when L5H7 is ablated (T), with 95% intervals; random-set means in brackets.\n")
H("| Text | Width | AUROC | Separation | Counters C | Corrector T | Width effect | Stored width S | Readers R |")
H("|---|---|---|---|---|---|---|---|---|")
GV = {}; gate = []; cnt = []; wid = []
for s_, w in GC:
    r = json.load(open(OUT / f"p5_04_general_{s_}_w{w}.json")); t = list(r["T"]["widths"].values())[0]
    c, x, rr = r["C"], r["S"], r["R"]
    hv = {k: holds(r, k) for k in "CTSR"}; GV[f"{NAME[s_]} {w}"] = hv
    gate.append(r["G"]["auroc"] >= 0.80); cnt += [hv["C"], hv["T"]]
    wid += [hv["R"] == PRED_R[pk(s_, w)], hv["S"] == PRED_S[pk(s_, w)]]
    mark = lambda b, pred=None: ("holds" if b else "does not") + ("" if pred is None else (" (as predicted)" if b == pred else " (NOT as predicted)"))
    H(f"| {NAME[s_]} | {w} | {r['G']['auroc']:.3f} | {r['G']['separation']:.3f} | {c['share']:.3f} {ci(c['share_ci95'])} [{c['random_mean']:.3f}]; {mark(hv['C'], True)} | "
      f"{t['diff']:.3f} {ci(t['diff_ci95'])} [{t['random_mean'] - t['intact']:.3f}]; {mark(hv['T'], True)} | {x['width_effect']:.3f} {ci(x['width_effect_ci95'])} | "
      f"{x['share']:.3f} {ci(x['share_ci95'])} [{x['random_mean']:.3f}]; {mark(hv['S'], PRED_S[pk(s_, w)])} | {rr['share']:.3f} {ci(rr['share_ci95'])} [{rr['random_mean']:.3f}]; {mark(hv['R'], PRED_R[pk(s_, w)])} |")
V["G1"] = all(gate); V["G2"] = all(cnt); V["G3"] = bool(sum(wid) >= 13)
H("")
H("| Part | Prediction | Result | Threshold | Verdict |")
H("|---|---|---|---|---|")
H(f"| G1 | line-breaking works in every condition | AUROC {min(json.load(open(OUT / f'p5_04_general_{s_}_w{w}.json'))['G']['auroc'] for s_, w in GC):.3f} at the lowest | AUROC >= 0.80 in all 8 | {'pass' if V['G1'] else 'fail'} |")
H(f"| G2 | the counters and the token corrector hold in every condition | {sum(cnt)} of {len(cnt)} | all 16 hold | {'pass' if V['G2'] else 'fail'} |")
H(f"| G3 | the width roles follow the predicted pattern (readers hold at English widths 30, 100 and 120 and German 60, not at English 20, code 40 and 60 or German 40; the stored width at the newline holds only in German at 60) | {sum(wid)} of {len(wid)} cells as predicted | >= 13 of 16 | {'pass' if V['G3'] else 'fail'} |")
V["G"] = all(V[k] for k in ("G1", "G2", "G3"))
H(f"\nG overall: {'pass' if V['G'] else 'fail'}\n")

text = "\n".join(L) + "\n"
tf = OUT / f"report5_tables_{MODE}.md"; vf = OUT / f"report5_verdicts_{MODE}.json"
if CHECK:
    ok = tf.read_text() == text and json.loads(vf.read_text()) == V
    print("check:", "identical" if ok else "DIFFERENT"); sys.exit(0 if ok else 1)
tf.write_text(text); vf.write_text(json.dumps(V, indent=1)); print(text); print(json.dumps(V))
