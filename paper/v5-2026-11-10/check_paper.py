"""Check every number in the D1 paper (version 5) against the saved study outputs.

1. Tables: make_tables.py is re-run into a scratch folder and must reproduce tables/ exactly, so the tables are
   current with the outputs. Their verdicts come from the frozen outputs.
2. Cross-check: the values behind the tables (values.py) must match, at the printed precision, the frozen round
   reports, which were written by the frozen table scripts (REPORT.md, REPORT-2.md, REPORT-3.md, the round-4 table
   file, and the cross-model reports). Round-5 values come from running the frozen code/report5_tables.py itself,
   whose regenerated table must equal the saved outputs/report5_tables_conf.md and appear in REPORT-5.md. values.py and those scripts read the same outputs by separate code.
3. Text: every number quoted in main.tex is asserted by a claim below, computed from values.py, from the frozen
   protocols and scripts, or from the material manifests.
4. Coverage: any number left in main.tex that no claim covers (and that is not part of a model or head name, a
   prediction label, a layer or MLP index, or another listed non-result) is reported, and fails.
Exits non-zero on any failure. Run inside the memory cage: python check_paper.py [--tex COPY_OF_MAIN_TEX]
"""
import json, os, re, subprocess, sys, tempfile, shutil, filecmp
import numpy as np
import values

HERE = os.path.dirname(os.path.abspath(__file__)); D1 = values.D1; O = values.O
bad, n = [], [0]
def check(ok, what):
    n[0] += 1
    if not ok: bad.append(what)

# ---------------------------------------------------------------- 1. tables current
tmp = tempfile.mkdtemp()
shutil.copytree(HERE, os.path.join(tmp, "p"), ignore=shutil.ignore_patterns("tables", "*.pdf", "*.aux", "*.log", "refcheck"))
os.symlink(os.path.join(D1), os.path.join(tmp, "D1"))
# run make_tables from a copy whose values.py points at the real outputs
src = open(os.path.join(tmp, "p", "values.py")).read().replace('D1 = os.path.normpath(os.path.join(HERE, "..", ".."))', f'D1 = {D1!r}')
open(os.path.join(tmp, "p", "values.py"), "w").write(src)
subprocess.run([sys.executable, "make_tables.py"], cwd=os.path.join(tmp, "p"), check=True, capture_output=True)
for f in sorted(os.listdir(os.path.join(HERE, "tables"))):
    check(filecmp.cmp(os.path.join(HERE, "tables", f), os.path.join(tmp, "p", "tables", f), shallow=False), f"table {f} is not current")
shutil.rmtree(tmp)

V = values.load()
f2 = lambda x: (lambda s: "0.00" if s == "-0.00" else s)(f"{x:.2f}")
f3 = lambda x: (lambda s: "0.000" if s == "-0.000" else s)(f"{x:.3f}")
def pc(x): return int(np.floor(100 * x + 0.5))
t3 = lambda t: f"{f3(t[0])} ({f3(t[1])}, {f3(t[2])})"

# ---------------------------------------------------------------- 2. cross-check against the frozen reports
R1, R2, R3 = (open(os.path.join(D1, f)).read() for f in ("REPORT.md", "REPORT-2.md", "REPORT-3.md"))
for rnd, doc in (("r1", R1), ("r2", R2), ("r3", R3)):
    for W in ("40", "60", "80"):
        for c in ("intact", "abl_L5H7", "abl_L5H3", "abl_L5H11", "abl_L5H14"):
            check(t3(V[f"{rnd}.tok.{W}.{c}"]) in doc, f"cross-check {rnd} token {W} {c}")
    check(t3(V[f"{rnd}.h2.width"]) in doc, f"cross-check {rnd} width effect")
    for k in ("L8H15", "readers3"):
        check(t3(V[f"{rnd}.h2.{k}"]) in doc, f"cross-check {rnd} {k}")
for rnd, doc in (("r2", R2), ("r3", R3)):
    for W in ("40", "60", "80"):
        check(t3(V[f"{rnd}.tok.{W}.L4H14_minus_intact"]) in doc, f"cross-check {rnd} H3 {W}")
        check(t3(V[f"{rnd}.h6.{W}.L5H7"]) in doc, f"cross-check {rnd} H6 {W}")
    for k in ("COUNT4_nl", "COUNT4_last"):
        x = V[f"{rnd}.h4.{k}"]; check(f"{100*x[0]:.1f}% ({100*x[1]:.1f}%, {100*x[2]:.1f}%)" in doc, f"cross-check {rnd} {k}")
    check(t3(V[f"{rnd}.h5.L8H15_VmK"]) in doc and t3(V[f"{rnd}.h5.L17H1_KmV"]) in doc, f"cross-check {rnd} H5")
T4 = open(os.path.join(O, "report4_tables_conf4-pythia410m.md")).read()
fc3 = lambda t: f"{f3(t[0])} ({f3(t[1])}, {f3(t[2])})"
for k in ("r4.d1", "r4.d1.diff", "r4.d2.C", "r4.d2.Wd", "r4.d3a", "r4.s1b", "r4.s2"):
    check(fc3(V[k]) in T4, f"cross-check round 4 {k}")
for k, v in (("r4.x.gpt2.probe", None), ("r4.x.gpt2.count", None), ("r4.x.p410.count", None), ("r4.x.gpt2.width", None), ("r4.x.p410.width", None),
             ("r4.x.gpt2.dil", None), ("r4.x.p410.dil", None), ("r4.x.gpt2.patch8max", None), ("r4.x.p410.patch8", None)):
    check(f"| {f3(V[k])} |" in T4, f"cross-check round 4 {k}")
XR = {m: open(os.path.join(D1, "xmodel", f"REPORT-{m}.md")).read() for m in ("pythia-1b", "qwen05b", "olmo1b")}
for m, doc in XR.items():
    for t in ("C", "S", "R"):
        x = V[f"x.{m}.{t}"]; check(f"{100*x[0]:.1f}% ({100*x[1]:.1f}%, {100*x[2]:.1f}%)" in doc, f"cross-check {m} {t}")
    check(f"{V[f'x.{m}.auroc']:.3f}" in doc, f"cross-check {m} AUROC")
XR3 = {m: open(os.path.join(D1, "xmodel", f"REPORT-{m}.md")).read() for m in ("qwen3-17b", "qwen3-06b")}
for m, doc in XR3.items():
    for t in ("C", "X", "R"):
        x = V[f"x.{m}.{t}"]; check(f"{100*x[0]:.1f}% ({100*x[1]:.1f}%, {100*x[2]:.1f}%)" in doc, f"cross-check {m} {t}")
    x = V[f"x.{m}.X.kept"]; check(f"{100*x[0]:.1f}% ({100*x[1]:.1f}%, {100*x[2]:.1f}%)" in doc, f"cross-check {m} X kept")
    for W in ("40", "60", "80"):
        check(t3(V[f"x.{m}.T.{W}"]) in doc, f"cross-check {m} T {W}")
    check(f"{V[f'x.{m}.auroc']:.3f}" in doc, f"cross-check {m} AUROC")
XR3b = {m: open(os.path.join(D1, "xmodel", f"REPORT-{m}.md")).read() for m in ("gemma4-e2b", "smollm2-17b")}
for m, doc in XR3b.items():
    check(V[f"x.{m}.sx"] == "S", f"{m} uses the S form")
    for t in ("C", "S", "R", "S.kept"):
        x = V[f"x.{m}.{t}"]; check(f"{100*x[0]:.1f}% ({100*x[1]:.1f}%, {100*x[2]:.1f}%)" in doc, f"cross-check {m} {t}")
    for W in ("40", "60", "80"):
        check(t3(V[f"x.{m}.T.{W}"]) in doc, f"cross-check {m} T {W}")
    check(f"{V[f'x.{m}.auroc']:.3f}" in doc, f"cross-check {m} AUROC")
XC = open(os.path.join(D1, "xmodel", "REPORT-CROSSMODEL.md")).read()
OV = V["xa.ov"]; nm7 = len(OV); cnt = [sum(1 for o in OV.values() if o[k]) for k in ("CT", "CS", "CR", "CTS")]
check(f"| Models with at least one such head | {cnt[0]} of {nm7} | {cnt[1]} of {nm7} | {cnt[2]} of {nm7} | {cnt[3]} of {nm7} |" in XC, "cross-check role overlaps")
ndil = sum(c < -0.3 for _, _, c in V["xa.dil"])
check(f"| All | {len(V['xa.dil'])} heads | | {ndil} of {len(V['xa.dil'])} below −0.30 |" in XC, "cross-check dilution count")
check(f"{V['xa.npass']} of {V['xa.ntest']} role tests, and all {V['xa.ngate']} gates** (AUROC {f3(V['xa.auroc'][0])}–{f3(V['xa.auroc'][1])}" in XC, "cross-check test counts")
check(f"**Counters carry {pc(V['xa.C'][0])}–{pc(V['xa.C'][1])}%**" in XC and f"**Readers carry {pc(V['xa.R'][0])}–{pc(V['xa.R'][1])}%**" in XC, "cross-check C and R ranges")
check(f"at most {pc(V['xa.randmax'])}% on any single set" in XC, "cross-check largest random set")

T5S = open(os.path.join(O, "report5_tables_conf.md")).read()
check(values.R5_TEXT.get("table") == T5S, "round-5 tables regenerated by the frozen script equal the saved table file")
R5 = open(os.path.join(D1, "REPORT-5.md")).read()
for line in T5S.splitlines():
    if line.startswith("|"): check(line in R5, f"round-5 table line in REPORT-5.md: {line[:60]}")
check(json.load(open(os.path.join(O, "report5_verdicts_conf.json"))) == V["r5.verdicts"], "round-5 verdicts equal the saved verdict file")

# ---------------------------------------------------------------- 3. the paper text
TEXFILE = sys.argv[sys.argv.index("--tex") + 1] if "--tex" in sys.argv else os.path.join(HERE, "main.tex")  # --tex: planted-error test on a copy
tex = open(TEXFILE).read()
body = tex[tex.index(r"\begin{document}"):]
body = re.sub(r"(?<!\\)%.*", "", body)
covered = np.zeros(len(body), bool)
def claim(text, ok=True, what=None):
    hits = [m.start() for m in re.finditer(re.escape(text), body)]
    check(bool(hits), f"text not found: {what or text!r}")
    check(ok, f"claim false: {what or text!r}")
    for h in hits: covered[h:h + len(text)] = True
def note(text, why): claim(text, True, f"{text} ({why})")
def rng(a, b, unit="\\%"): return f"{a}--{b}{unit}"
def prange(xs, unit="\\%"): return rng(min(pc(x) for x in xs), max(pc(x) for x in xs), unit)
def drange(xs, sep=" to "): return f"{f2(min(xs))}{sep}{f2(max(xs))}"

# abstract and introduction
check(0.2 <= V["r4.x.gpt2.count"] / V["r4.x.p410.count"] <= 0.3, "GPT-2 uses the count a quarter as strongly")
# the named decision step: readers from layer 6, writers up to MLP 18, dampers in MLPs 21 and 23 (set names of the frozen round-4 script)
check(all(k in V for k in ("r4.med.C.readers3", "r4.med.C.MLP14_16_18", "r4.med.C.MLP21_23")) and V["r4.d4.C.top"] == "L17H1", "named decision parts: L6H2/L8H15/L17H1, MLPs 14, 16, 18, MLPs 21, 23")
claim("layers 6 to 18"); claim("layers 6 to 23"); claim("layers 6 to 14", max(range(5, 24), key=lambda l: V[f"r4.cos.{l}"]) == 14)
note("January 2025 and September 2026", "selection rule, data/newtext/MANIFEST.md")
claim("within the first 1,000 steps"); claim("step 1,000")
for s in ("2,000", "16,000 and 48,000"): note(s, "checkpoint steps")
# design
for t, why in (("600 to 1,800 characters", "paragraph filter, common.py"), ("knots at 2, 5, 8, 12, 16, 24 and 32", "spline, frozen scripts"),
               ("at most 12 characters left", "separation rule"), ("at most 30 characters left", "regression points"),
               ("4 to 14 characters from the edge", "width items"), ("at least 10 characters", "width items"), ("Widths were 40, 60 and 80", "design"),
               ("95\\%", "interval level"), ("2,000 resamples", "bootstrap"), ("third line", "design"), ("264 paragraphs", "data/newtext/fetch_summary.json"),
               ("208 paragraphs", "data/newtext2/fetch_summary.json"), ("146 articles", "fetch summary"), ("108 articles", "fetch summary"),
               ("2,520 articles", "fetch summary"), ("630 different articles", "fetch summary"), ("6,000 bytes", "selection rule"),
               ("6 November 2026", "fetch date"), ("CC BY-SA 4.0", "licence"), ("10,000 resamples", "post hoc"), ("0.025", "post hoc alpha")):
    note(t, why)
fs1 = json.load(open(os.path.join(D1, "data", "newtext", "fetch_summary.json"))); fs2 = json.load(open(os.path.join(D1, "data", "newtext2", "fetch_summary.json")))
check(fs1["parts"]["conf-pythia410m"] == {"paragraphs": 264, "articles": 146} and fs1["n_articles"] == 2520, "first batch counts")
check(fs2["parts"]["conf4-pythia410m"] == {"paragraphs": 208, "articles": 108} and fs2["n_articles"] == 630, "second batch counts")
# behaviour
claim(f"newline is its top prediction at {pc(V['exp01.pythia-410m.60.break_acc'])}\\% of due breaks, and AUROC is {f3(V['exp01.pythia-410m.60.auroc_model'])}")
claim(f"GPT-2 Small manages {pc(V['exp01.gpt2.60.break_acc'])}\\% at the same width, with AUROC {f3(V['exp01.gpt2.60.auroc_model'])}")
c = V["exp02"]["60"]["r2"]
claim(f"($R^2$ {f2(c['chars_remaining(bins)'])} at width 60)"); claim(f"($R^2$ {f2(c['tokens_on_line(bins)'])})")
check(c["chars+tokens"] <= c["chars_remaining(bins)"] + 0.005, "adding tokens adds nothing")
# section 6.1
claim(f"raises the token weight by {drange([V[f'{r}.tok.{W}.L4H14_minus_intact'][0] for r in ('r2', 'r3') for W in ('40', '60', '80')])} (H3)")
claim(f"a weight of {drange([V[f'{r}.tok.{W}.abl_L5H7'][0] for r in ('r1', 'r2', 'r3') for W in ('40', '60', '80')])} units of log-odds at widths 40, 60 and 80 in rounds 1 to 3 (H1b)")
claim("within $\\pm$0.10")
for r in ("r1", "r2", "r3"):
    x = V[f"{r}.tok.40.intact"]; claim(f"{f3(x[0])} [{f3(x[1])}, {f3(x[2])}]", x[2] > 0.10)
check(V["r3.tok.60.intact"][1] > 0, "width-60 interval above zero on new text")
ratio = [V[f"{r}.tok.40.intact"][0] / V[f"{r}.tok.40.abl_L5H7"][0] for r in ("r1", "r2", "r3")]
check(all(0.08 <= x <= 0.16 for x in ratio), "between a tenth and a seventh of the weight without L5H7")
claim(f"($-${f2(-V['r2.h6.40.L5H7'][0])} and $-${f2(-V['r2.h6.60.L5H7'][0])} on the test split, $-${f2(-V['r3.h6.40.L5H7'][0])} and $-${f2(-V['r3.h6.60.L5H7'][0])} on new text")
# 6.2
claim(f"carried {prange([V[f'{r}.h2.readers3'][0] for r in ('r1', 'r2', 'r3')])} of the patched width effect in rounds 1 to 3, and L8H15 alone {prange([V[f'{r}.h2.L8H15'][0] for r in ('r1', 'r2', 'r3')])}")
claim(f"removed {pc(V['r2.h4.COUNT4_nl'][0])}\\% of the whole-prompt width effect", pc(V["r2.h4.COUNT4_nl"][0]) == pc(V["r3.h4.COUNT4_nl"][0]))
claim(f"and {prange([V[f'{r}.h4.COUNT4_last'][0] for r in ('r2', 'r3')])} when only")
claim(f"kept {prange([V[f'{r}.h4.kept_COUNT4_nl'] for r in ('r2', 'r3')])} of the break/no-break separation")
claim(f"kept {prange([V[f'{r}.h4.kept_COUNT4_all'] for r in ('r2', 'r3')])} (H4d)")
claim(f"carries {drange([V[f'{r}.h5.L8H15.V'][0] for r in ('r2', 'r3')])} log-probability units")
claim(f"against {drange([V[f'{r}.h5.L8H15.K'][0] for r in ('r2', 'r3')])} for its key alone")
claim(f"({drange([V[f'{r}.h5.L17H1.K'][0] for r in ('r2', 'r3')])}, against {f2(V['r2.h5.L17H1.V'][0])} for the value)", f2(V["r2.h5.L17H1.V"][0]) == f2(V["r3.h5.L17H1.V"][0]))
# 6.3
claim(f"by {f2(V['r4.d.C'])}, the patched width by {f2(V['r4.d.W'])}, and both together by {f2(V['r4.d.CW'])}", V["r4.d.CW"] < V["r4.d.C"] + V["r4.d.W"])
claim(f"(cosine {f2(V['r4.cos.6'])})"); claim(f"to {f2(V['r4.cos.12'])} after layer 12 and {f2(V['r4.cos.14'])} after layer 14")
m = lambda c, s: pc(V[f"r4.med.{c}.{s}"])
claim(f"layers 6 to 11 carry {m('C', 'layers6-11')}\\% of the count's effect and {m('Wd', 'layers6-11')}\\% of the width's, and layers 12 to 17 carry {m('C', 'layers12-17')}\\% and {m('Wd', 'layers12-17')}\\%")
check(pc(V["r4.med.Wd.all_heads"]) == 100, "all heads remove the whole width effect")
note("288 heads", "16 heads x 18 layers"); claim("all 288 heads in layers 6 to 23", V["r4.d4.C.top"] == "L17H1" and V["r4.d4.Wd.top"] == "L17H1")
claim(f"carry {pc(V['r4.d1'][0])}\\% of the count's effect, against {pc(V['r4.d1.rand'])}\\% for random MLP triples")
claim(f"makes the count's effect {-pc(V['r4.d2.C'][0])}\\% larger and the width's {-pc(V['r4.d2.Wd'][0])}\\% larger")
# 6.4
claim("7 of 288 heads and 7 of 18 MLPs"); claim(f"AUROC was unchanged ({f3(V['r4.s.DEC.mean.auc'])} against {f3(V['r4.s.auc0'])})")
claim(f"and {pc(V['r4.s1b'][0])}\\% of the width effect remained, against at most {pc(V['r4.s.DEC.rand.widthmax'])}\\%")
check(0.45 <= V["r4.s.DEC.mean.sep"] <= 0.55 and V["r4.s.DEC.rand.sepmax"] >= 0.45, "about half the separation, a level random circuits also reach")
claim(f"kept {pc(V['r4.s.EARLY.mean.sep'])}\\% of the separation, against about zero", abs(V["r4.s.EARLY.rand.sepmin"]) < 0.05 and abs(V["r4.s.EARLY.rand.sepmax"]) < 0.05)
note("80 heads of layers 1 to 5", "16 x 5"); note("eight early heads", "circuit"); note("seven later heads", "circuit")
claim(f"With the whole circuit, {pc(V['r4.s2'][0])}\\% of the separation remained, and AUROC {f3(V['r4.s.BOTH.mean.auc'])}, against at most {pc(V['r4.s.BOTH.rand.sepmax'])}\\%")
check(V["r4.s.EARLY.resample.sep"] < V["r4.s.EARLY.mean.sep"] and V["r4.s.BOTH.resample.sep"] < V["r4.s.BOTH.mean.sep"], "resample lower for the early scopes")
# 6.5
claim(f"carries {prange([V['r4.g.40.5.var1'], V['r4.g.60.5.var1']])} of the reliable between-class variance")
claim(f"explains {prange([V['r4.g.40.5.lin1'], V['r4.g.60.5.lin1']])} of it (G1)")
claim(f"a spline explains {prange([V['r4.g.40.6.spl2'], V['r4.g.60.6.spl2']])} of it and a straight line {prange([V['r4.g.40.6.lin2'], V['r4.g.60.6.lin2']])}")
pk = [V[f"r4.g.{W}.{h}.peak"] for W in ("40", "60") for h in ("L6H2", "L8H15", "L17H1")]
claim(f"peaks at {min(pk)} to {max(pk)} characters remaining")
# 7.1
claim(f"AUROC of {f3(V['r4.x.gpt2.auc60'])} at width 60 on the new text, against {f3(V['r4.x.p410.auc60'])}")
claim(f"$R^2$ {f2(V['r4.x.gpt2.probe'])}, against {f2(V['r4.x.p410.probe'])}")
claim(f"raises its log-probability of the newline by {f2(V['r4.x.gpt2.count'])}, against {f2(V['r4.x.p410.count'])} in Pythia-410M")
claim(f"raises it by {f2(V['r4.x.gpt2.width'])}, against {f2(V['r4.x.p410.width'])} (the whole-prompt width effect)")
# the two measures of Section 7 and Section 9.1 on the same items: the count shift after layer 5 and the newline patch after layer 5
claim(f"larger than the {f2(V['r4.d.C'])} and {f2(V['r4.d.W'])} of Section~\\ref{{sec:decision}}, which come from the same {V['r4.d.n']} items",
      abs(V['r4.x.p410.count_by_layer'][5] - V['r4.d.C']) < 0.005 and abs(V['r4.x.p410.patch'][5] - V['r4.d.W']) < 0.005 and V['r4.d.n'] == V['r4.x.p410.nitems'] == V['r4.x.gpt2.nitems']
      and V['r4.x.p410.count'] > V['r4.d.C'] and V['r4.x.p410.width'] > V['r4.d.W'] and int(np.argmax(V['r4.x.p410.count_by_layer'])) > 5)
claim(f"({V['r4.x.gpt2.dilhead']}) puts {pc(V['r4.x.gpt2.dil'])}\\% of its attention on the line start, where Pythia's L5H6 puts {pc(V['r4.x.p410.dil'])}\\%")
claim("by less than 0.05 after layer 8", V["r4.x.gpt2.patch8max"] < 0.05)
claim(f"still moves it by {f2(V['r4.x.p410.patch8'])}")
# 7.2
X = ("pythia-1b", "qwen05b", "olmo1b")
claim(f"Counters carry {prange([V[f'x.{m}.C'][0] for m in X])} of a count change")
claim(f"a weight of {drange([V[f'x.{m}.T.{W}'][0] for m in X for W in ('40', '60', '80')])} per token")
claim(f"removes {prange([V[f'x.{m}.S'][0] for m in X])} of the width effect while the decision keeps {prange([V[f'x.{m}.S.kept'][0] for m in X])} of its separation")
claim(f"readers carry {prange([V[f'x.{m}.R'][0] for m in X])} of the stored length")
check(all(all(V[f"x.{m}.verdicts"].values()) for m in X), "every cross-model test passed")
claim(f"({pc(V['x.olmo1b.R'][0])}\\%)"); claim(f"overshoot real ({pc(V['x.ov.share'][0])}\\%"); claim(f"brought the share to {pc(V['x.ov.withL5H11'][0])}\\%")
check(all(V["x.ov.verdicts"].values()), "all six follow-up predictions held")
# 8
claim(f"(AUROC {f3(V['ck.1000.auroc'])}), L5H6 puts {pc(V['ck.1000.L5H6_share'])}\\%")
check(1000 / 143000 < 0.01, "step 1,000 is under 1% of training"); claim("under 1\\% of training")
claim(f"carrying {pc(V['ck.1000.width_share_removed_readers3'])}\\% of the width effect at step 1,000 and {pc(V['ck.final.width_share_removed_readers3'])}\\% at the end")
claim(f"grows from nothing at step 2,000 to {f2(V['ck.final.corrector_token_coef_change'])} at the end", abs(V["ck.2000.corrector_token_coef_change"]) < 0.03)
check(all(abs(V[f"ck.{s}.token_coef"]) < 0.1 for s in ("1000", "2000", "4000", "8000", "16000", "48000", "final")), "intact token weight near zero throughout")
check(V["ck.16000.width_share_removed_MLP21_23"] > -0.05 and V["ck.48000.width_share_removed_MLP21_23"] < -0.25, "damping appears between 16k and 48k")
# discussion, limits, appendices
note("widths of 40 to 80 characters", "design")
claim(f"keeps every part except H1a at width 40", V["ph.ntests"] - V["ph.hold"] == 2 and all(t["holm"] >= V["ph.alpha"] for t in V["ph.tests"] if t["part"] == "H1a, width 40"))
claim(f"Of the {V['ph.ntests']} parts, {V['ph.hold']} hold after correction")
note("26 numbered experiments", "exp01-exp26 in NOTEBOOK.md")
check(len({f[:5] for f in os.listdir(os.path.join(D1, "code")) if re.match(r"exp(0[1-9]|1\d|2[0-6])_", f)}) == 26, "26 exploration scripts before round 4")
claim(f"removed {pc(V['x.olmo1b.X'][0])}\\% of the width effect and the decision kept only {pc(V['x.olmo1b.X.kept'][0])}\\%")
claim(f"tested on {V['x.ov.n']} width items")
x = V["x.ov.share"]; claim(f"removed {pc(x[0])}\\% [{pc(x[1])}\\%, {pc(x[2])}\\%]")
claim(f"gave {pc(V['x.ov.mean'][0])}\\% and {pc(V['x.ov.resample'][0])}\\%")
x = V["x.ov.L5H11"]; claim(f"by {f2(x[0])} of the patched effect [{f2(x[1])}, {f2(x[2])}], against {f2(V['x.ov.L5H11.randmean'])} on average")
claim(f"readers' share to {pc(V['x.ov.withL5H11'][0])}\\%"); claim(f"({f2(V['x.ov.inject'][0])} of the effect)")
claim(f"value ({f2(V['x.ov.value'][0])} of the effect), not its key ({f2(V['x.ov.key'][0])})")

# Qwen3 (PROTOCOL-X3)
Q = ("qwen3-17b", "qwen3-06b")
check(all(all(V[f"x.{m}.verdicts"].values()) for m in Q), "every Qwen3 test passed")
check(all(V[f"x.qwen3-17b.{t}.set"] == V[f"x.qwen3-06b.{t}.set"] for t in ("C", "T", "X", "R")), "same head sets in both Qwen3 models")
cfg = {}
for m, rp in (("qwen3-17b", "Qwen3-1.7B-Base"), ("qwen3-06b", "Qwen3-0.6B-Base")):
    import glob
    cfg[m] = json.load(open(os.path.join(D1, "models", "configs", f"Qwen--{rp}", "config.json")))  # release: shipped copy
claim("hidden sizes of 2,048 and 1,024 over the same 28 layers of 16 query heads",
      cfg["qwen3-17b"]["hidden_size"] == 2048 and cfg["qwen3-06b"]["hidden_size"] == 1024 and all(c["num_hidden_layers"] == 28 and c["num_attention_heads"] == 16 for c in cfg.values()))
cset = V["x.qwen3-17b.C.set"].split(","); cl = [int(h[1:].split("H")[0]) for h in cset]
claim("Five heads in layers 4 to 7", len(cset) == 5 and min(cl) == 4 and max(cl) == 7)
claim(f"carry {pc(V['x.qwen3-17b.C'][0])}\\% of a count change in Qwen3-1.7B and {pc(V['x.qwen3-06b.C'][0])}\\% in Qwen3-0.6B")
tr = lambda m: [V[f"x.{m}.T.{W}"][0] for W in ("40", "60", "80")]
claim(f"{f2(min(tr('qwen3-17b')))} to {f2(max(tr('qwen3-17b')))} units of log-odds in the larger model and {f2(min(tr('qwen3-06b')))} to {f2(max(tr('qwen3-06b')))} in the smaller",
      V["x.qwen3-17b.T.set"] == "L4H1")
claim(f"they remove {pc(V['x.qwen3-17b.X'][0])}\\% and {pc(V['x.qwen3-06b.X'][0])}\\% of the width effect")
claim(f"which keeps {pc(V['x.qwen3-17b.X.kept'][0])}\\% and {pc(V['x.qwen3-06b.X.kept'][0])}\\% of its separation", all(V[f"x.{m}.X.kept"][0] <= 0.75 for m in Q))
claim(f"carry {pc(V['x.qwen3-17b.R'][0])}\\% of the stored length's effect in both", pc(V["x.qwen3-17b.R"][0]) == pc(V["x.qwen3-06b.R"][0]))
check(all(abs(V[f"x.{m}.{t}.rand"]) < 0.05 for m in Q for t in ("C", "X", "R")) and all(abs(V[f"x.{m}.T.{W}.rand"] - V[f"x.{m}.T.{W}.intact"][0]) < 0.05 for m in Q for W in ("40", "60", "80")), "random heads near zero")
note("at least a quarter of its separation", "PROTOCOL-X3 X criterion, kept <= 0.75")
P3 = V["x3.parts"]
claim("a third batch of 2,520 English Wikipedia articles", V["x3.n_articles"] == 2520)
claim("186 paragraphs from 111 articles (Qwen3-1.7B)", P3["conf-qwen3-17b"] == {"paragraphs": 186, "articles": 111})
claim("194 paragraphs from 102 articles (Qwen3-0.6B)", P3["conf-qwen3-06b"] == {"paragraphs": 194, "articles": 102})
check(len(P3) == 8, "split into parts for four models")

# Gemma-4-E2B and SmolLM2-1.7B (PROTOCOL-X3b and its SmolLM2 addendum), and all seven models
GS = ("gemma4-e2b", "smollm2-17b")
check(all(all(V[f"x.{m}.verdicts"].values()) for m in GS), "every Gemma and SmolLM2 test passed")
cg = {}
for m, rp in (("gemma4-e2b", "models--google--gemma-4-E2B"), ("smollm2-17b", "models--HuggingFaceTB--SmolLM2-1.7B")):
    c_ = json.load(open(os.path.join(D1, "models", "configs", rp[len("models--"):], "config.json"))); cg[m] = c_.get("text_config", c_)  # release: shipped copy
g = cg["gemma4-e2b"]; lt = g["layer_types"]
claim("each of its 35 layers has 8 query heads sharing one key and value head",
      g["num_hidden_layers"] == 35 and g["num_attention_heads"] == 8 and g["num_key_value_heads"] == 1)
claim("four sliding-window layers precede each global layer", lt == (["sliding_attention"] * 4 + ["full_attention"]) * 7)
claim("the last 20 layers reuse keys and values", g["num_kv_shared_layers"] == 20)
sm = cg["smollm2-17b"]
claim("SmolLM2-1.7B has 24 layers of 32 heads", sm["num_hidden_layers"] == 24 and sm["num_attention_heads"] == 32 and sm["num_key_value_heads"] == 32)
claim("one of 32 heads in its layer", sm["num_attention_heads"] == 32 and V["x.smollm2-17b.R.set"] == "L11H21")
nC = {m: o["nC"] for m, o in OV.items()}
claim("where the other models used three to six", nC["gemma4-e2b"] == 2 and min(v for k, v in nC.items() if k != "gemma4-e2b") == 3 and max(nC.values()) == 6)
claim(f"carry {pc(V['x.gemma4-e2b.C'][0])}\\% of a count change, where", V["x.gemma4-e2b.C.set"] == "L6H3,L12H3")
tg = [V[f"x.gemma4-e2b.T.{W}"][0] for W in ("40", "60", "80")]; ts = [V[f"x.smollm2-17b.T.{W}"][0] for W in ("40", "60", "80")]
claim(f"{f2(min(tg))} to {f2(max(tg))} units of log-odds. It is also", V["x.gemma4-e2b.T.set"] == "L6H3")
claim(f"which remove {pc(V['x.gemma4-e2b.S'][0])}\\% of the width effect while the decision keeps {pc(V['x.gemma4-e2b.S.kept'][0])}\\%", V["x.gemma4-e2b.S.set"] == "L6H1,L6H3,L7H6")
claim(f"which together carry {pc(V['x.gemma4-e2b.R'][0])}\\% of the stored length", V["x.gemma4-e2b.R.set"] == "L9H5,L12H3")
claim(f"carry {pc(V['x.smollm2-17b.C'][0])}\\% of a count change. Two correctors", V["x.smollm2-17b.C.set"] == "L5H21,L7H22,L9H7")
claim(f"give {f2(min(ts))} to {f2(max(ts))} per token", V["x.smollm2-17b.T.set"] == "L5H21,L3H2")
claim(f"Four newline writers remove {pc(V['x.smollm2-17b.S'][0])}\\% of the width effect while the decision keeps {pc(V['x.smollm2-17b.S.kept'][0])}\\%",
      len(V["x.smollm2-17b.S.set"].split(",")) == 4)
claim(f"carries {pc(V['x.smollm2-17b.R'][0])}\\% of the stored length's effect. Here too")
claim("Here too one head, L5H21, counts, corrects and writes", OV["smollm2-17b"]["CTS"] == {"L5H21"})
gsmax = max(max(J_[t]["random_share"]) for m, part in (("gemma4-e2b", "conf-gemma4-e2b"), ("smollm2-17b", "conf-smollm2-17b"))
            for J_ in [values.J(values.XO, f"xm_conf_{m}_{part}.json")] for t in ("C", "S", "R"))
claim(f"from the same layers carried more than {pc(gsmax)}\\% of any effect")
claim(f"every frozen test passed: {V['xa.ntest']} role tests and all {V['xa.ngate']} behaviour gates (AUROC {f3(V['xa.auroc'][0])} to {f3(V['xa.auroc'][1])} at width 60)",
      V["xa.npass"] == V["xa.ntest"] and V["xa.ngate"] == V["xa.nmodel"] == 7)
claim(f"Counters carry {pc(V['xa.C'][0])}--{pc(V['xa.C'][1])}\\% of a count change and readers {pc(V['xa.R'][0])}--{pc(V['xa.R'][1])}\\%")
claim(f"carried more than {pc(V['xa.randmax'])}\\% of any effect. The roles")
claim(f"all {V['xa.ntest']} frozen role tests passed", V["xa.npass"] == V["xa.ntest"])
note("Seven other open models", "seven cross-model runs")
check(V["xa.nmodel"] == 7, "seven other models")
oth = lambda k: sorted(m for m, o in OV.items() if not o[k])
claim(f"In {['', '', '', '', '', '', 'six'][cnt[0]]} of the seven models at least one counter is also a token corrector, and in {['', '', '', '', '', '', 'six'][cnt[1]]} at least one counter also writes",
      oth("CT") == oth("CS") == ["olmo1b"])
claim("In four models (Pythia-1B, Qwen2.5-0.5B, Gemma-4-E2B and SmolLM2-1.7B) a single head plays all three roles",
      sorted(m for m, o in OV.items() if o["CTS"]) == sorted(["pythia-1b", "qwen05b", "gemma4-e2b", "smollm2-17b"]))
check(cnt[3] == 4, "four models with one head that counts, corrects and writes")
claim("In two, OLMo-2-1B and Gemma-4-E2B, a counter is also a reader", sorted(m for m, o in OV.items() if o["CR"]) == ["gemma4-e2b", "olmo1b"])
dl = V["xa.dil"]; ex = [(m, h, c) for m, h, c in dl if c >= -0.3]
claim(f"Of the {len(dl)} token-corrector heads named across the seven models, {ndil} attend less", len(dl) == 15)
claim("below $-0.30$")
claim(f"rises with the count ($+{f2(ex[0][2])}$)", len(ex) == 1 and ex[0][:2] == ("smollm2-17b", "L5H21"))
XP = open(os.path.join(D1, "xmodel", "PRECISION-smollm2-17b-posthoc.md")).read(); xp = lambda k: V[f"xp.smol.{k}"]
check(f"| AUROC at width 60 | {xp('auroc')[0]:.4f} | {xp('auroc')[1]:.4f} |" in XP and f"| Separation | {xp('sep')[0]:.3f} | {xp('sep')[1]:.3f} |" in XP
      and f"r = {xp('r'):.4f}" in XP and f"| Decision points | {xp('n')} | {xp('n')} |" in XP and "POST HOC" in XP, "cross-check SmolLM2 float32 note")
claim(f"compared the newline log-odds at {xp('n'):,} decision points of WikiText-2 training text wrapped at width 60")
claim(f"the AUROC was {xp('auroc')[0]:.4f} in float32 and {xp('auroc')[1]:.4f} in bfloat16, the separation {xp('sep')[0]:.3f} in both",
      f"{xp('sep')[0]:.3f}" == f"{xp('sep')[1]:.3f}")
claim(f"log-odds correlated at $r = {xp('r'):.4f}$")
claim("A float32 check of SmolLM2-1.7B, run post hoc after its confirmation", "POST HOC" in XP)
claim("Gemma-4-E2B had no float32 check", "Gemma-4-E2B still has no float32 check" in XP)
claim("Gemma-4-E2B ran in bfloat16 only, with no float32 check, for lack of GPU memory; a post hoc float32 check of SmolLM2-1.7B agreed")
check("SmolLM2-1.7B had no float32" not in tex and "Gemma-4-E2B and SmolLM2-1.7B had no float32" not in tex
      and "Gemma-4-E2B and SmolLM2-1.7B ran in bfloat16 only" not in tex, "no stale SmolLM2 precision limit")
claim("193 paragraphs from 110 articles and 158 paragraphs from 88 articles",
      P3["conf-gemma4-e2b"] == {"paragraphs": 193, "articles": 110} and P3["conf-smollm2-17b"] == {"paragraphs": 158, "articles": 88})

# round 5
r5v = V["r5.verdicts"]; parts = [k for k in r5v if len(k) == 2]
claim("All 16 predictions held", len(parts) == 16 and all(r5v[k] for k in parts))
en = lambda w, q: V[f"r5.t.en.{w}.{q}"]
claim(f"a token weight of {f2(en(20, 'intact')[0])} remains at width 20", all(abs(en(w, "intact")[0]) <= 0.04 for w in (60, 80, 100, 120)) and en(20, "intact")[0] == max(en(w, "intact")[0] for w in (20, 30, 40, 60, 80, 100, 120)))
P5 = re.search(r"PATH = \[([^\]]*)\]", open(os.path.join(D1, "code", "p5_01_sufficiency.py")).read()).group(1)
PATH5 = [x.strip().strip('"') for x in P5.split(",")]
claim("24 further components", len(PATH5) == 24); claim("24 components took fidelity", len(PATH5) == 24)
claim(f"keeps {f2(V['r5.s.plus24.mean.fid'])} of the intact model's variation under mean ablation but only {f2(V['r5.s.plus24.resample.fid'])} under resample ablation")
claim("added 24 components", len(PATH5) == 24)
claim(f"bring the circuit's fidelity to {f3(V['r5.s.plus24.mean.fid'])} under mean ablation")
note("widths of 20 to 120 characters", "round-5 widths"); note("widths 20, 30, 100 and 120", "round-5 widths")
note("widths of 40 to 80 characters", "rounds 1-4 widths"); note("every width from 20 to 120 characters", "round-5 widths")
note("1 for a perfect copy, 0 for a constant", "definition of fidelity")
fs2r = fs2["parts"]["reserve-pythia410m"]
claim("unread part (200 paragraphs)", fs2r["paragraphs"] == 200); claim("(200 paragraphs from 116 articles)", fs2r == {"paragraphs": 200, "articles": 116})
claim(f"Without L5H7 the token weight is {f2(en(20, 'abl')[0])} at width 20, {f2(en(100, 'abl')[0])} at width 100 and {f2(en(120, 'abl')[0])} at width 120; the frozen test T3 was the difference between widths 20 and 100 ({f2(V['r5.t3'][0])})",
      abs(V['r5.t3'][0] - (en(20, 'abl')[0] - en(100, 'abl')[0])) < 1e-9)
r5abl = lambda keys: [V[f'r5.t.{t}.{w}.abl'][0] for t, w in keys]
claim(f"in round 5 it was {drange(r5abl([('en', w) for w in (20, 30, 40, 60, 80)]))} in English at widths 20 to 80, and {drange(r5abl([('en', 100), ('en', 120), ('code', 40), ('code', 60), ('de', 40), ('de', 60)]))} at widths 100 and 120, in code and in German")
claim(f"falls by {f3(-en(20, 'att6')[0])} per extra token at width 20 and by {f3(-en(80, 'att6')[0])} at width 80")
rm = lambda w: 1 - en(w, "intact")[0] / en(w, "abl")[0]
claim(f"{pc(rm(20))}\\% at width 20, {pc(rm(30))}\\% at width 30, and {pc(min(rm(w) for w in (40, 60, 80, 100, 120)))}\\% or more at widths of 40 and above")
claim(f"(rank correlation $-${f2(-V['r5.t2'])} over eleven conditions")
claim(f"{f2(en(20, 'intact')[0])} at width 20, {f2(en(30, 'intact')[0])} at width 30, {f2(en(40, 'intact')[0])} at width 40, and within 0.04 of zero at widths of 60 to 120",
      all(abs(en(w, "intact")[0]) <= 0.04 for w in (60, 80, 100, 120)))
check(all(V[f"r5.t.{t}.{w}.cpt"] < min(en(x, "cpt") for x in (20, 30, 40, 60, 80, 100, 120)) for t in ("code", "de") for w in (40, 60)), "code and German have fewer characters per token")
check(all(V[f"r5.t.de.{w}.intact"][0] < en(w, "intact")[0] and V[f"r5.t.de.{w}.ntok"] > en(w, "ntok") for w in (40, 60)), "German keeps less, holds more tokens")
cw = lambda w, q: V[f"r5.t.code.{w}.{q}"]
claim(f"({f3(cw(40, 'intact')[0])} and {f3(cw(60, 'intact')[0])} against {f3(en(40, 'intact')[0])} and {f3(en(60, 'intact')[0])})", all(cw(w, "intact")[0] > en(w, "intact")[0] for w in (40, 60)))
check(all(0.4 <= cw(w, "abl")[0] / en(w, "abl")[0] <= 0.6 for w in (40, 60)), "code's bias without L5H7 about half as large")
claim(f"({pc(1 - cw(40, 'intact')[0] / cw(40, 'abl')[0])}\\% and {pc(1 - cw(60, 'intact')[0] / cw(60, 'abl')[0])}\\%)", all(1 - cw(w, "intact")[0] / cw(w, "abl")[0] < rm(w) for w in (40, 60)))
check(abs(en(120, "att6")[0]) > 2 * abs(en(80, "att6")[0]), "steep again at width 120")
M_ = lambda src, k: V[f"r5.m.{src}.{k}"][0]
claim(f"makes the count's effect {pc(-M_('C', 'mlp'))}\\% larger and the width's {pc(-M_('Wd', 'mlp'))}\\% larger on this text")
claim(f"({pc(-M_('C', 'mln'))}\\% and {pc(-M_('Wd', 'mln'))}\\%; M1)")
claim("among the largest 0.1\\% of their layers", all(V[f"e29.{k}.norm_pct"] >= 0.999 for k in ("MLP23.736", "MLP21.1410")))
claim(f"by {f2(-V['e29b.MLP23.736.nl_logit'])[:-1]} and {f2(-V['e29b.MLP21.1410.nl_logit'])[:-1]} logit units per unit of activation, against a spread of {f2(min(V['e29b.MLP23.736.sd'], V['e29b.MLP21.1410.sd']))} to {f2(max(V['e29b.MLP23.736.sd'], V['e29b.MLP21.1410.sd']))}",
      all(V[f"e29b.{k}.rank"] == 1 for k in ("MLP23.736", "MLP21.1410")) and f"{-V['e29b.MLP23.736.nl_logit']:.1f}" == "3.3" and f"{-V['e29b.MLP21.1410.nl_logit']:.1f}" == "1.1")
claim(f"reproduces {pc(M_('C', 'top2') / M_('C', 'mlp'))}\\% and {pc(M_('Wd', 'top2') / M_('Wd', 'mlp'))}\\% of the damping (M2)")
claim(f"(correlations {f2(V['r5.m.corr.MLP23.736'])} and {f2(V['r5.m.corr.MLP21.1410'])}; M4)")
claim(f"raises the newline's log-probability by {f2(V['r5.m.nat.top2_abl.nl.dlogp'][0])} where the newline is the likely next token, while other likely tokens change by $-${f2(-V['r5.m.nat.top2_abl.other.dlogp'][0])}")
claim(f"lowers other tokens by {f2(-V['r5.m.nat.mlp_abl.other.dlogp'][0])}")
S_ = lambda k: V[f"r5.s.{k}"]
claim(f"(fidelity $-${f2(-S_('circuit.mean.fid'))}; Table")
e27f = V["e27.final"]; claim(f"to {f2(e27f['sep_kept'])} times its separation after ten additions, while AUROC fell to {f2(e27f['auroc'])}", e27f["late_added"] == 10)
lastfid = json.load(open(os.path.join(O, "exp27b_fidelity.json")))["path"][-1]["search_fid"]
claim(f"24 components took fidelity from $-${f2(-V['e27b.circuit'])} to {f2(lastfid)}")
roles27 = json.load(open(os.path.join(O, "exp27_suff_search.json")))["roles"]
early5 = [x for x in PATH5 if x.startswith("L") and int(x[1:].split("H")[0]) <= 5]; late5 = [x for x in PATH5 if x not in early5]
lab5 = [roles27[x]["label"] for x in early5]
claim(f"{len(early5)} heads of layers 1 to 4, of which {lab5.count('earlier')} attend mainly to earlier lines and {lab5.count('prev')} to the previous token",
      max(int(x[1:].split("H")[0]) for x in early5) == 4 and min(int(x[1:].split("H")[0]) for x in early5) == 1 and lab5.count("earlier") + lab5.count("prev") == len(early5))
check(sorted(x for x in late5 if x.startswith("MLP")) == ["MLP13", "MLP19", "MLP9"] and len([x for x in late5 if not x.startswith("MLP")]) == 8, "MLPs 9, 13, 19 and eight later heads")
note("eight later heads", "path composition, checked above")
claim(f"fidelity {f3(S_('plus24.mean.fid'))}, with {pc(S_('plus24.mean.sep'))}\\% of the separation and {pc(S_('plus24.mean.width'))}\\% of the width effect kept")
claim(f"The round-4 circuit keeps {pc(S_('circuit.mean.sep'))}\\% and {pc(S_('circuit.mean.width'))}\\%, and random additions of the same composition reach at most {f2(S_('rand.mean.fid')[1])}, {pc(S_('rand.mean.sep')[1])}\\% and {pc(S_('rand.mean.width')[1])}\\%")
claim(f"fidelity {f2(S_('plus24.resample.fid'))} and {pc(S_('plus24.resample.sep'))}\\% of the separation, more than the round-4 circuit ({pc(S_('circuit.resample.sep'))}\\%)",
      S_("plus24.resample.sep") > S_("rand.resample.sep")[1])
check(0.75 <= S_("plus24.mean.fid") <= 0.85 and 0.15 <= S_("plus24.resample.fid") <= 0.25, "about four-fifths and about a fifth")
GK = [("en", 20), ("en", 30), ("en", 100), ("en", 120), ("code", 40), ("code", 60), ("de", 40), ("de", 60)]
G_ = lambda t, w, q: V[f"r5.g.{t}.{w}.{q}"]
claim(f"(AUROC {f3(min(G_(t, w, 'auroc') for t, w in GK))} to {f3(max(G_(t, w, 'auroc') for t, w in GK))} on held-out text; G1)")
check(all(G_(t, w, "holds")["C"] and G_(t, w, "holds")["T"] for t, w in GK), "counting roles hold in all eight")
check(G_("en", 20, "C.n") == 1 and G_("en", 20, "S.n") == 8 and G_("en", 20, "R.n") == 8, "width 20: one and eight items")
check([ (t, w) for t, w in GK if G_(t, w, "holds")["R"]] == [("en", 30), ("en", 100), ("en", 120), ("code", 60), ("de", 60)], "readers hold where stated")
nR5 = sum(G_(t, w, "holds")["R"] for t, w in GK); nS5 = sum(G_(t, w, "holds")["S"] for t, w in GK)
claim("in eight further conditions of width and text type, the readers held in five and the stored length was detectable in one", len(GK) == 8 and nR5 == 5 and nS5 == 1)
claim("in only one of the eight conditions", len(GK) == 8 and nS5 == 1)
check([(t, w) for t, w in GK if not G_(t, w, "holds")["R"]] == [("en", 20), ("code", 40), ("de", 40)], "discussion: readers failed at English 20, code 40, German 40")
claim("at English widths 30, 100 and 120 and in German at 60", [(t, w) for t, w in GK if G_(t, w, "holds")["R"] and t != "code"] == [("en", 30), ("en", 100), ("en", 120), ("de", 60)])
note("widths 20 to 120", "round-5 widths"); note("bars at 24", "the fuller circuit has 24 additions")
check([ (t, w) for t, w in GK if G_(t, w, "holds")["S"]] == [("de", 60)], "stored width only German 60")
check(V["r5.g.match"] == 15, "fifteen of sixteen as predicted")
claim(f"the readers carried {pc(G_('code', 60, 'R')[0])}\\% of the width effect")
fail_ = [G_(t, w, "W")[0] for t, w in GK if not G_(t, w, "holds")["R"]]; hold_ = [G_(t, w, "W")[0] for t, w in GK if G_(t, w, "holds")["R"]]
claim(f"least (whole-prompt width effect {f2(min(fail_))} to {f2(max(fail_))}) and held in the other five ({f2(min(hold_))} to {f2(max(hold_))})", max(fail_) < min(hold_) and len(fail_) == 3)
check(G_("en", 100, "S.rand") > 0.15 and G_("en", 120, "S.rand") > 0.15, "random heads remove much at 100, 120")
# material, round 5
bs = json.load(open(os.path.join(D1, "data", "code", "build_summary.json"))); fde = json.load(open(os.path.join(D1, "data", "de", "fetch_summary.json")))
claim("(374 paragraphs from 154 files)", bs["parts"]["conf5-code"] == {"paragraphs": 374, "files": 154})
claim("(252 articles, fetched on 6 November 2026)", fde["n_articles"] == 252 and fde["fetched_utc"].startswith("2026-11-06"))
claim("(114 paragraphs from 60 articles)", fde["parts"]["conf5-de"] == {"paragraphs": 114, "articles": 60})
for t, why in (("Python 3.14", "build_code.py"), ("at most 1,800 characters", "build rule"), ("no word over 38 characters", "build rule"), ("2025 and 2026", "creation period")):
    note(t, why)

# non-result numbers: labels, design values, names
for t, why in (("2025--26", "creation period"), ("a4paper", "class option")):
    if t in body: note(t, why)

# ---------------------------------------------------------------- 4. coverage
skip = [r"\\(ref|label|cite[pt]?|citeauthor|input|inputrows|includegraphics|section|subsection|paragraph|bibliography(style)?|usetikzlibrary|resizebox)\*?(\[[^\]]*\])?\{[^}]*\}",
        r"\\(begin|end)\{[^}]*\}(\{[^}]*\})?(\[[^\]]*\])?", r"\[[a-z!]+\]", r"P\{\d\.\dcm\}", r"\d?\.?\d+\\(linewidth|textwidth|columnwidth)",
        r"\\node\[[^\]]*\] \([a-z]+\) at \([^)]*\)", r"(fill|[a-z]+)=[a-z]*!?\d+", r"length=\d+mm", r"inner sep=\d+pt", r"minimum height=\d\.\dcm", r"bend right=\d+",
        r"rounded corners=\dpt", r"GPT-2", r"Pythia-[\d.]+[MB]", r"Qwen2\.5-0\.5B", r"Qwen3(-\d\.\dB)?", r"Gemma-4-E2B", r"SmolLM2(-1\.7B)?", r"OLMo-2-1B", r"Claude~3\.5", r"\bL\d+H\d+\b", r"COUNT4",
        r"\b[HRDSGXTM]\d[a-d]?\b", r"S\d--S\d", r"MLP\d+\.\d+", r"[Rr]ounds? \d to \d", r"Round 5", r"\bH\d[a-d]?--[cd]\b", r"H\d--H\d", r"X\d--X\d", r"D\d--D\d", r"G\d--G\d", r"MLPs? ?\d+(, \d+)*(,? (and|or) \d+)?",
        r"MLPs \d+--\d+", r"\blayers? \d+( (to|and) \d+)?\b", r"layers \d+--\d+", r"\d+--\d+\}", r"rounds? \d( (to|and) \d)?\b", r"Round \d",
        r"\bwidths? \d+( and \d+)?\b", r"Width \d+", r"[Rr]ounds 1--3", r"Table~\\ref", r"Figure~\\ref", r"Appendix~\\ref", r"\$R\^2\$", r"\(1a\)|\(1b\)",
        r"\\scriptsize|\\footnotesize|\\small", r"\{2\.\d\}", r"0\.\d+\\textwidth", r"sm_\d+", r"step \d+(,\d{3})?", r"steps \d+(,\d{3})?( and \d+(,\d{3})?)?",
        r"pdfoutput=1", r"\d+pt", r"\\setcounter\{\w+\}\{\d\}", r"\\renewcommand\{\\\w+\}\{0\.\d+\}", r"widths 40, 60 and 80", r"40, 60 and 80",
        r"seven parts", r"width-\d+", r"round-\d", r"\\multicolumn\{\d\}", r"(?i:b?float)\d+", r"1B models", r"\d+ \(\d+ paragraphs", r"WikiText-2", r"first \d,\d{3}", r"head \d+ of layer \d+"]
for pat in skip:
    for mm in re.finditer(pat, body): covered[mm.start():mm.end()] = True
for mm in re.finditer(r"\d+(?:,\d{3})*(?:\.\d+)?", body):
    if not covered[mm.start():mm.end()].all():
        ctx = body[max(0, mm.start() - 50):mm.end() + 30].replace("\n", " ")
        bad.append(f"unchecked number {mm.group()!r}: ...{ctx}...")

print(f"{n[0]} checks, {len(bad)} failed")
for b in bad: print("FAIL", b)
sys.exit(1 if bad else 0)
