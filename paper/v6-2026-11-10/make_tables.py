"""Writes the paper's tables (tables/<name>-rows.tex) from the saved outputs, through values.py.
Verdicts are read from the frozen outputs where the scripts saved them, never recomputed here.
Run inside the memory cage: python make_tables.py"""
import json, os
import values

HERE = os.path.dirname(os.path.abspath(__file__)); TAB = os.path.join(HERE, "tables"); os.makedirs(TAB, exist_ok=True)
O = values.O
V = values.load()
J = lambda f: json.load(open(os.path.join(O, f)))


DEF = [3]  # decimals: 3 in the prediction tables, 2 in the descriptive ones
def f(x, d=None):
    d = DEF[0] if d is None else d
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0: s = s[1:]
    return s.replace("-", "$-$")
def fc(t, d=None): return f"{f(t[0], d)} [{f(t[1], d)}, {f(t[2], d)}]"
def pf(b): return "pass" if b else "fail"
def write(name, rows):
    with open(os.path.join(TAB, f"{name}-rows.tex"), "w") as fh:
        for r in rows:
            fh.write(r if isinstance(r, str) else " & ".join(r) + " \\\\")
            fh.write("\n")


# ---------------- Table: predictions of rounds 1-3
v1 = J("conf01_token_validation.json")["verdicts"]; w1 = J("conf02_width_validation.json")["verdicts"]
v2 = J("c2_01_token_test.json")["verdicts"]; v3 = J("c3_01_token_conf-pythia410m.json")["verdicts"]
w2 = J("c2_02_width_test.json")["verdicts"]; w3 = J("c3_02_width_conf-pythia410m.json")["verdicts"]
n2 = J("c2_03_newline_test.json")["verdicts"]; n3 = J("c3_03_newline_conf-pythia410m.json")["verdicts"]
p2 = J("c2_04_paths_test.json")["verdicts"]; p3 = J("c3_04_paths_conf-pythia410m.json")["verdicts"]
r3res = J("c3_01_token_conf-pythia410m.json")["residual_w40_above_zero"]
TV = {"r1": v1, "r2": v2, "r3": v3}
rows = []
for W in ("40", "60", "80"):
    rows.append([f"H1a: intact token weight, width {W}", "within $\\pm$0.10"] +
                [f"{fc(V[f'{r}.tok.{W}.intact'])} {pf(TV[r][W]['H1a'])}" for r in ("r1", "r2", "r3")])
rows.append(["Width-40 residual (round 3 only)", "lower bound $>$ 0", "", "", f"{fc(V['r3.tok.40.intact'])} {pf(r3res)}"])
lo = lambda r: min(V[f"{r}.tok.{W}.abl_L5H7"][0] for W in ("40", "60", "80")); hi = lambda r: max(V[f"{r}.tok.{W}.abl_L5H7"][0] for W in ("40", "60", "80"))
lb = lambda r: min(V[f"{r}.tok.{W}.abl_L5H7"][1] for W in ("40", "60", "80"))
rows.append(["H1b: L5H7 ablated (range over widths; lowest lower bound)", "$\\geq$ 0.30, lower bound $>$ 0.20"] +
            [f"{f(lo(r))}--{f(hi(r))} ({f(lb(r))}) {pf(all(TV[r][W]['H1b'] for W in ('40', '60', '80')))}" for r in ("r1", "r2", "r3")])
rows.append(["H1c: largest change from a control head", "$<$ 0.10 from intact"] +
            [f"{f(max(V[f'{r}.tok.{W}.ctlmax'] for W in ('40', '60', '80')))} {pf(all(TV[r][W]['H1c'] for W in ('40', '60', '80')))}" for r in ("r1", "r2", "r3")])
d14 = lambda r, i: [V[f"{r}.tok.{W}.L4H14_minus_intact"][i] for W in ("40", "60", "80")]
rows.append(["H3: L4H14 ablated minus intact (range; lowest lower bound)", "$\\geq$ 0.10, lower bound $>$ 0", "--"] +
            [f"{f(min(d14(r, 0)))}--{f(max(d14(r, 0)))} ({f(min(d14(r, 1)))}) {pf(all(TV[r][W]['H3'] for W in ('40', '60', '80')))}" for r in ("r2", "r3")])
rows.append("\\midrule")
WV = {"r1": w1, "r2": w2, "r3": w3}
rows.append(["Width effect (positive control)", "interval $>$ 0"] + [f"{fc(V[f'{r}.h2.width'])} {pf(WV[r]['positive_control'])}" for r in ("r1", "r2", "r3")])
rows.append(["H2: share carried by L8H15", "$\\geq$ 0.25"] + [f"{fc(V[f'{r}.h2.L8H15'])} {pf(WV[r]['H2_L8H15'])}" for r in ("r1", "r2", "r3")])
rows.append(["H2: share carried by L8H15, L17H1, L6H2", "$\\geq$ 0.50"] + [f"{fc(V[f'{r}.h2.readers3'])} {pf(WV[r]['H2_readers3'])}" for r in ("r1", "r2", "r3")])
rows.append(["H2: ten random triples, mean share", "$<$ 0.15"] + [f"{f(V[f'{r}.h2.random'])} {pf(WV[r]['H2_random'])}" for r in ("r1", "r2", "r3")])
rows.append("\\midrule")
NV = {"r2": n2, "r3": n3}
rows.append(["H4a: COUNT4 at every newline, share of width effect removed", "$\\geq$ 0.25, lower bound $>$ 0", "--"] + [f"{fc(V[f'{r}.h4.COUNT4_nl'])} {pf(NV[r]['H4a'])}" for r in ("r2", "r3")])
rows.append(["H4b: COUNT4 at the last newline only", "$\\geq$ 0.25, lower bound $>$ 0", "--"] + [f"{fc(V[f'{r}.h4.COUNT4_last'])} {pf(NV[r]['H4b'])}" for r in ("r2", "r3")])
rows.append(["H4c: ten random sets of four heads, mean", "$<$ 0.15", "--"] + [f"{f(V[f'{r}.h4.random'])} {pf(NV[r]['H4c'])}" for r in ("r2", "r3")])
rows.append(["H4d: separation kept, newline-only / every position", "$\\geq$ 0.80 / $<$ 0.30", "--"] +
            [f"{f(V[f'{r}.h4.kept_COUNT4_nl'])} / {f(V[f'{r}.h4.kept_COUNT4_all'])} {pf(NV[r]['H4d'])}" for r in ("r2", "r3")])
PV = {"r2": p2, "r3": p3}
rows.append(["H5a: L8H15, value-only minus key-only", "interval $>$ 0", "--"] + [f"{fc(V[f'{r}.h5.L8H15_VmK'])} {pf(PV[r]['H5a'])}" for r in ("r2", "r3")])
rows.append(["H5b: L17H1, key-only minus value-only", "interval $>$ 0", "--"] + [f"{fc(V[f'{r}.h5.L17H1_KmV'])} {pf(PV[r]['H5b'])}" for r in ("r2", "r3")])
for W in ("40", "60"):
    rows.append([f"H6a: L5H7 at newlines, change in previous-line token weight, width {W}", "$\\leq$ $-$0.03, interval $<$ 0", "--"] +
                [f"{fc(V[f'{r}.h6.{W}.L5H7'])} {pf(NV[r][f'H6a_w{W}'])}" for r in ("r2", "r3")])
rows.append(["H6b: L5H11 at newlines, largest change (widths 40, 60)", "$<$ 0.02", "--"] +
            [f"{f(max(abs(V[f'{r}.h6.{W}.L5H11'][0]) for W in ('40', '60')), 3)} {pf(NV[r]['H6b_w40'] and NV[r]['H6b_w60'])}" for r in ("r2", "r3")])
write("rounds123", rows)

# ---------------- Table: round 4
R4 = J("report4_verdicts_conf4-pythia410m.json")
rows = [
    ["D1", "MLPs 14, 16, 18 frozen: share of COUNT effect lost", fc(V["r4.d1"]), "$\\geq$ 0.11, lower bound $>$ 0", ""],
    ["", f"minus mean of ten random MLP triples ({f(V['r4.d1.rand'])})", fc(V["r4.d1.diff"]), "lower bound $>$ 0", pf(R4["D1"])],
    ["D2", "MLPs 21, 23 frozen: share of COUNT effect lost", fc(V["r4.d2.C"]), "$\\leq$ $-$0.10, upper bound $<$ 0", ""],
    ["", "MLPs 21, 23 frozen: share of WIDTH effect lost", fc(V["r4.d2.Wd"]), "$\\leq$ $-$0.10, upper bound $<$ 0", pf(R4["D2"])],
    ["D3", f"cosine of COUNT and WIDTH changes, after layer 12 minus after layer 6", fc(V["r4.d3a"]), "$\\geq$ 0.20, lower bound $>$ 0", ""],
    ["", "cosine after layer 12", f(V["r4.cos.12"]), "$\\geq$ 0.65", pf(R4["D3"])],
    ["D4", f"largest head direct effect: COUNT {V['r4.d4.C.top']} {f(V['r4.d4.C.topv'])} (next {f(V['r4.d4.C.nextv'])}); WIDTH {V['r4.d4.Wd.top']} {f(V['r4.d4.Wd.topv'])} (next {f(V['r4.d4.Wd.nextv'])})", "", "L17H1 for both", pf(R4["D4"])],
    "\\midrule",
    ["S1", "late circuit only, decision token: AUROC (intact " + f(V["r4.s.auc0"], 3) + ")", f(V["r4.s.DEC.mean.auc"], 3), "$\\geq$ intact $-$ 0.02", ""],
    ["", "same: width effect kept", fc(V["r4.s1b"]), f"$\\geq$ 0.33, lower bound $>$ {f(V['r4.s.DEC.rand.widthmax'])}", pf(R4["S1"])],
    ["S2", "whole circuit: separation kept", fc(V["r4.s2"]), f"$\\geq$ 0.15, lower bound $>$ {f(V['r4.s.BOTH.rand.sepmax'])}", pf(R4["S2"])],
    "\\midrule",
    ["G1", "top direction after layer 5: variance share / linear $R^2$ (widths 40; 60)",
     f"{f(V['r4.g.40.5.var1'])} / {f(V['r4.g.40.5.lin1'])}; {f(V['r4.g.60.5.var1'])} / {f(V['r4.g.60.5.lin1'])}", "$\\geq$ 0.50 / $\\geq$ 0.85", pf(R4["G1"])],
    ["G2", "second direction after layer 6: spline minus linear $R^2$ (widths 40; 60)",
     f"{f(V['r4.g.40.6.spl2'] - V['r4.g.40.6.lin2'])}; {f(V['r4.g.60.6.spl2'] - V['r4.g.60.6.lin2'])}", "$\\geq$ 0.30", pf(R4["G2"])],
    ["G3", "readers' attention to the line start: monotone $R^2$ (lowest); peak (largest characters remaining)",
     f"{f(min(V[f'r4.g.{W}.{h}.mono'] for W in ('40', '60') for h in ('L6H2', 'L8H15', 'L17H1')))}; {max(V[f'r4.g.{W}.{h}.peak'] for W in ('40', '60') for h in ('L6H2', 'L8H15', 'L17H1'))}",
     "$\\geq$ 0.90; $\\leq$ 5", pf(R4["G3"])],
    "\\midrule",
    ["X1", "GPT-2 Small: best-layer probe $R^2$ for characters used", f(V["r4.x.gpt2.probe"]), "$\\geq$ 0.70", pf(R4["X1"])],
    ["X2", "largest count-shift effect over layers, GPT-2 / Pythia-410M", f"{f(V['r4.x.gpt2.count'])} / {f(V['r4.x.p410.count'])}", "ratio $\\leq$ 0.5", pf(R4["X2"])],
    ["X3", "whole-prompt width effect, GPT-2 / Pythia-410M", f"{f(V['r4.x.gpt2.width'])} / {f(V['r4.x.p410.width'])}", "ratio $\\leq$ 0.5", pf(R4["X3"])],
    ["X4", "largest line-start share of a counting head, GPT-2 / Pythia-410M", f"{f(V['r4.x.gpt2.dil'])} / {f(V['r4.x.p410.dil'])}", "ratio $\\leq$ 0.5; Pythia $\\geq$ 0.60", pf(R4["X4"])],
    ["X5", "width patch at the newline: GPT-2 after layers 8--11 (largest) / Pythia-410M after layer 8", f"{f(V['r4.x.gpt2.patch8max'])} / {f(V['r4.x.p410.patch8'])}", "$<$ 0.10 / $\\geq$ 0.25", pf(R4["X5"])],
]
write("round4", rows)

# ---------------- Table: sufficiency scopes (round 4)
rows = []
for scope, lab in (("DEC", "Late circuit only, decision token"), ("EARLY", "Early circuit heads only, every position"), ("BOTH", "Both")):
    for nm in ("mean", "resample"):
        rows.append([lab if nm == "mean" else "", nm, f(V[f"r4.s.{scope}.{nm}.sep"]), f(V[f"r4.s.{scope}.{nm}.auc"], 3), f(V[f"r4.s.{scope}.{nm}.width"])])
    rows.append(["", "five random circuits", f"{f(V[f'r4.s.{scope}.rand.sepmin'])} to {f(V[f'r4.s.{scope}.rand.sepmax'])}", "", f"{f(V[f'r4.s.{scope}.rand.widthmin'])} to {f(V[f'r4.s.{scope}.rand.widthmax'])}"])
write("sufficiency", rows)

# ---------------- Table: decision mediation by set (round 4)
rows = []
for s, lab in (("layers6-11", "All components, layers 6--11"), ("layers12-17", "All components, layers 12--17"), ("layers18-23", "All components, layers 18--23"),
               ("all_heads", "All heads, layers 6--23"), ("all_mlps", "All MLPs, layers 6--23"), ("readers3", "L6H2, L8H15, L17H1"), ("L17H1", "L17H1"),
               ("MLP14_16_18", "MLPs 14, 16, 18"), ("MLP21_23", "MLPs 21, 23"), ("random_mlp3_mean", "Ten random MLP triples, mean")):
    rows.append([lab, f(V[f"r4.med.C.{s}"]), f(V[f"r4.med.Wd.{s}"])])
write("mediation", rows)

DEF[0] = 2

# ---------------- Round 5 (tables use 3 decimals)
DEF[0] = 3
R5 = V["r5.verdicts"]
def r5(k): return V[k]
rows = []
rows.append(["T1", "Leftover token weight, width 20 minus width 60 (English)", fc(r5("r5.t1")), "$\\geq$ 0.04, lower bound $>$ 0", pf(R5["T1"])])
rows.append(["T2", "Rank correlation of the leftover weight with tokens on the line (11 conditions)", f(r5("r5.t2")), "$\\leq$ $-$0.50", pf(R5["T2"])])
rows.append(["T3", "Bias with L5H7 ablated, width 20 minus width 100 (English)", fc(r5("r5.t3")), "$\\geq$ 0.15, lower bound $>$ 0", pf(R5["T3"])])
rows.append(["T4", "L5H6 line-start attention per token, width 20 minus width 80", fc(r5("r5.t4"), 4), "$\\leq$ $-$0.003, upper bound $<$ 0", pf(R5["T4"])])
rows.append(["T5", "Rise when L5H7 is ablated, smallest of 11 conditions", f(min(r5(k)[0] for k in V if k.startswith("r5.t.") and k.endswith(".rise"))), "each $\\geq$ 0.10, lower bound $>$ 0", pf(R5["T5"])])
rows.append("\\midrule")
rows.append(["M1", "Damping by MLPs 21, 23 with the final LayerNorm frozen: COUNT; WIDTH", f"{fc(r5('r5.m.C.mln'))}; {fc(r5('r5.m.Wd.mln'))}", "$\\leq$ $-$0.15 and within 0.10 of the free value", pf(R5["M1"])])
rows.append(["M2", "Damping by MLP23.736 and MLP21.1410 alone: COUNT; WIDTH", f"{fc(r5('r5.m.C.top2'))}; {fc(r5('r5.m.Wd.top2'))}", "$\\leq$ $-$0.15 and $\\geq$ half the MLPs' damping", pf(R5["M2"])])
rows.append(["M3", "The two neurons ablated, natural text: newline; other likely tokens", f"{fc(r5('r5.m.nat.top2_abl.nl.dlogp'))}; {fc(r5('r5.m.nat.top2_abl.other.dlogp'))}", "newline $\\geq$ 0.15; other within $\\pm$0.10", pf(R5["M3"])])
rows.append(["M4", "Correlation of each neuron's activation with the newline log-odds", f"{f(r5('r5.m.corr.MLP23.736'))}; {f(r5('r5.m.corr.MLP21.1410'))}", "each $\\geq$ 0.50", pf(R5["M4"])])
rows.append("\\midrule")
rows.append(["S1", "Fidelity of the fuller circuit, mean ablation", f"{f(r5('r5.s.plus24.mean.fid'))} [{f(r5('r5.s.plus24.mean.fid.ci')[0])}, {f(r5('r5.s.plus24.mean.fid.ci')[1])}]", f"$\\geq$ 0.50, lower bound $>$ random ({f(r5('r5.s.rand.mean.fid')[1])})", pf(R5["S1"])])
rows.append(["S2", "Separation kept, fuller minus round-4 circuit", fc(r5("r5.s.dsep")), "$\\geq$ 0.20, lower bound $>$ 0", pf(R5["S2"])])
rows.append(["S3", "Width effect kept by the fuller circuit", f"{f(r5('r5.s.plus24.mean.width'))} [{f(r5('r5.s.plus24.mean.width.ci')[0])}, {f(r5('r5.s.plus24.mean.width.ci')[1])}]", f"$\\geq$ 0.40, lower bound $>$ random ({f(r5('r5.s.rand.mean.width')[1])})", pf(R5["S3"])])
rows.append(["S4", "Resample ablation: fidelity; separation kept", f"{f(r5('r5.s.plus24.resample.fid'))} [{f(r5('r5.s.plus24.resample.fid.ci')[0])}, {f(r5('r5.s.plus24.resample.fid.ci')[1])}]; {f(r5('r5.s.plus24.resample.sep'))}", f"$\\geq$ 0.15; $\\geq$ 0.25; above the round-4 circuit", pf(R5["S4"])])
rows.append("\\midrule")
gmin = min(V[k] for k in V if k.startswith("r5.g.") and k.endswith(".auroc"))
rows.append(["G1", "AUROC, lowest of 8 conditions", f(gmin), "$\\geq$ 0.80 in all", pf(R5["G1"])])
rows.append(["G2", "Counters and token corrector meet the frozen criteria (cells)", f"{V['r5.g.cnt']} of 16", "all 16", pf(R5["G2"])])
rows.append(["G3", "Width roles as predicted (cells)", f"{V['r5.g.match']} of 16", "$\\geq$ 13", pf(R5["G3"])])
write("round5", rows)

rows = []
for tx, lab in (("en", "English"), ("code", "code"), ("de", "German")):
    for w in ((20, 30, 40, 60, 80, 100, 120) if tx == "en" else (40, 60)):
        k = f"r5.t.{tx}.{w}"
        rows.append([lab, str(w), f(V[k + ".ntok"], 1), f(V[k + ".cpt"], 2), fc(V[k + ".intact"]), fc(V[k + ".abl"]), fc(V[k + ".att6"], 4)])
write("tokbias", rows)

rows = []
LABS = (("circuit", "Round-4 circuit"), ("plus4", "+ 4"), ("plus8", "+ 8"), ("plus16", "+ 16"), ("plus24", "+ 24 (fuller circuit)"))
for nm_, lab in LABS:
    for mode in ("mean", "resample"):
        k = f"r5.s.{nm_}.{mode}"
        rows.append([lab if mode == "mean" else "", mode, f(V[k + ".fid"]), f(V[k + ".sep"]), f(V[k + ".auroc"]), f(V[k + ".width"])])
for mode in ("mean", "resample"):
    rows.append(["5 random sets, + 24" if mode == "mean" else "", mode] + [f"{f(V[f'r5.s.rand.{mode}.{q}'][0])} to {f(V[f'r5.s.rand.{mode}.{q}'][1])}" for q in ("fid", "sep")] + ["", f"{f(V[f'r5.s.rand.{mode}.width'][0])} to {f(V[f'r5.s.rand.{mode}.width'][1])}"])
write("fuller", rows)

rows = []
def hold(k, role, pred=None):
    h = V[k + ".holds"][role]; s_ = "yes" if h else "no"
    return s_ if pred is None or pred == h else s_ + "$^\\dagger$"
for tx, lab, ws in (("en", "English", (20, 30, 100, 120)), ("code", "code", (40, 60)), ("de", "German", (40, 60))):
    for w in ws:
        k = f"r5.g.{tx}.{w}"; pkey = w if tx == "en" else ({"code": "conf5-code", "de": "conf5-de"}[tx], w)
        pr, ps = V["r5.g.pred_R"][pkey], V["r5.g.pred_S"][pkey]
        rows.append([lab, str(w), f(V[k + ".auroc"]), f"{f(V[k + '.C'][0], 2)} ({V[k + '.C.n']}) {hold(k, 'C')}", f"{f(V[k + '.T'][0], 2)} {hold(k, 'T')}",
                     f(V[k + ".W"][0], 2), f"{f(V[k + '.S'][0], 2)} ({V[k + '.S.n']}) {hold(k, 'S', ps)}", f"{f(V[k + '.R'][0], 2)} {hold(k, 'R', pr)}"])
write("general", rows)
DEF[0] = 2

# ---------------- Table: cross-model
rows = []
for m, lab in (("pythia-1b", "Pythia-1B"), ("qwen05b", "Qwen2.5-0.5B"), ("olmo1b", "OLMo-2-1B")):
    vd = V[f"x.{m}.verdicts"]
    T = [V[f"x.{m}.T.{W}"][0] for W in ("40", "60", "80")]
    rows.append([lab, f(V[f"x.{m}.auroc"], 3), f"{fc(V[f'x.{m}.C'])}", f"{f(min(T))}--{f(max(T))}",
                 f"{fc(V[f'x.{m}.S'])}; {f(V[f'x.{m}.S.kept'][0])}", f"{fc(V[f'x.{m}.R'])}",
                 f"{f(max(V[f'x.{m}.{t}.rand'] for t in ('C', 'S', 'R')))}", "pass" if all(vd.values()) else "fail"])
rows.append("\\midrule")
for m, lab in (("qwen3-17b", "Qwen3-1.7B"), ("qwen3-06b", "Qwen3-0.6B$^*$")):
    vd = V[f"x.{m}.verdicts"]
    T = [V[f"x.{m}.T.{W}"][0] for W in ("40", "60", "80")]
    rows.append([lab, f(V[f"x.{m}.auroc"], 3), f"{fc(V[f'x.{m}.C'])}", f"{f(min(T))}--{f(max(T))}",
                 f"{fc(V[f'x.{m}.X'])}; {f(V[f'x.{m}.X.kept'][0])}$^\\dagger$", f"{fc(V[f'x.{m}.R'])}",
                 f"{f(max(V[f'x.{m}.{t}.rand'] for t in ('C', 'X', 'R')))}", "pass" if all(vd.values()) else "fail"])
rows.append("\\midrule")
for m, lab in (("gemma4-e2b", "Gemma-4-E2B"), ("smollm2-17b", "SmolLM2-1.7B")):
    vd = V[f"x.{m}.verdicts"]; assert V[f"x.{m}.sx"] == "S"
    T = [V[f"x.{m}.T.{W}"][0] for W in ("40", "60", "80")]
    rows.append([lab, f(V[f"x.{m}.auroc"], 3), f"{fc(V[f'x.{m}.C'])}", f"{f(min(T))}--{f(max(T))}",
                 f"{fc(V[f'x.{m}.S'])}; {f(V[f'x.{m}.S.kept'][0])}", f"{fc(V[f'x.{m}.R'])}",
                 f"{f(max(V[f'x.{m}.{t}.rand'] for t in ('C', 'S', 'R')))}", "pass" if all(vd.values()) else "fail"])
write("xmodel", rows)
rows = []
for m, lab in (("pythia-1b", "Pythia-1B"), ("qwen05b", "Qwen2.5-0.5B"), ("olmo1b", "OLMo-2-1B")):
    for t, tl in (("C", "Counters"), ("T", "Token correctors"), ("S", "Stored length"), ("X", "Stored length, second set"), ("R", "Readers")):
        k = f"x.{m}.{t}.set"
        if k in V: rows.append([lab if t == "C" else "", tl, V[k].replace(",", ", ")])
    rows.append("\\midrule")
for t, tl in (("C", "Counters"), ("T", "Token corrector"), ("X", "Newline writers"), ("R", "Readers")):
    rows.append(["Qwen3-1.7B and 0.6B" if t == "C" else "", tl, V[f"x.qwen3-17b.{t}.set"].replace(",", ", ")])
    assert V[f"x.qwen3-17b.{t}.set"] == V[f"x.qwen3-06b.{t}.set"]
for m, lab in (("gemma4-e2b", "Gemma-4-E2B"), ("smollm2-17b", "SmolLM2-1.7B")):
    rows.append("\\midrule")
    for t, tl in (("C", "Counters"), ("T", "Token correctors"), ("S", "Stored length"), ("R", "Readers")):
        rows.append([lab if t == "C" else "", tl, V[f"x.{m}.{t}.set"].replace(",", ", ")])
write("xmodel-sets", rows)

# ---------------- Table: training (exploratory)
rows = []
for s, lab in (("1000", "1,000"), ("2000", "2,000"), ("4000", "4,000"), ("8000", "8,000"), ("16000", "16,000"), ("48000", "48,000"), ("final", "143,000")):
    rows.append([lab, f(V[f"ck.{s}.auroc"], 3), f(V[f"ck.{s}.sep"]), f(V[f"ck.{s}.L5H6_share"]), f(V[f"ck.{s}.counters_sep_kept"]),
                 f(V[f"ck.{s}.corrector_token_coef_change"]), f(V[f"ck.{s}.width_share_removed_COUNT4_nl"]),
                 f(V[f"ck.{s}.width_share_removed_readers3"]), f(V[f"ck.{s}.width_share_removed_MLP14_16_18"]), f(V[f"ck.{s}.width_share_removed_MLP21_23"])])
write("training", rows)

# ---------------- Table: behaviour (exploratory)
rows = []
for W in ("20", "40", "60", "80", "120", "160"):
    r = [W]
    for m in ("pythia-410m", "pythia-1b", "gpt2"):
        r += [f(V[f"exp01.{m}.{W}.break_acc"]), f(V[f"exp01.{m}.{W}.false_break"]), f(V[f"exp01.{m}.{W}.auroc_model"], 3)]
    rows.append(r)
write("behaviour", rows)

# ---------------- Table: post hoc Holm (appendix)
# the saved records label round 2's repeats of H1 and H2 "2 (R1)" and "2 (R2)" and write criteria in ASCII; print plain labels
HOLM_PART = {"H2 L8H15 >= 0.25": "H2: L8H15, share $\\geq$ 0.25", "H2 readers3 >= 0.50": "H2: three readers, share $\\geq$ 0.50",
             "H2 random mean < 0.15": "H2: random triples, mean share $<$ 0.15", "H2 positive control": "H2: positive control",
             "H4 positive control": "H4: positive control", "H5 positive control": "H5: positive control",
             "H4d, newline-only kept >= 0.80": "H4d: separation kept, newline-only, $\\geq$ 0.80",
             "H4d, all-position kept < 0.30": "H4d: separation kept, every position, $<$ 0.30"}
rows = []
for t in V["ph.tests"]:
    rnd = t["round"].split(" ")[0]; part = HOLM_PART.get(t["part"], t["part"])
    assert rnd in ("1", "2") and t["round"] in ("1", "2", "2 (R1)", "2 (R2)") and not any(c in part.replace("$<$", "") for c in "<>=") and "readers3" not in part, t
    rows.append([rnd, part, f(t["p"], 4), f(t["holm"], 4), "yes" if t["holm"] < V["ph.alpha"] else "no"])
write("holm", rows)
print("tables written:", sorted(os.listdir(TAB)))
