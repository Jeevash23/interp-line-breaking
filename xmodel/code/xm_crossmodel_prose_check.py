"""Check every number quoted in the prose of REPORT-CROSSMODEL.md against the saved outputs.
Each claim is a (text that must appear in the report, value recomputed from the outputs) pair, as in
xm3_report_prose_check.py; structural claims (counts, ranges, layer bounds) are recomputed and asserted.
Usage: xm_crossmodel_prose_check.py"""
import json, sys
from pathlib import Path
X = Path(__file__).resolve().parent.parent
O = X / "outputs"
M = [("Pythia-1B", "pythia-1b", "pythia1b"), ("Qwen2.5-0.5B", "qwen05b", "qwen05b"), ("OLMo-2-1B", "olmo1b", "olmo1b"),
     ("Qwen3-1.7B", "qwen3-17b", "qwen3-17b"), ("Qwen3-0.6B", "qwen3-06b", "qwen3-06b"), ("Gemma-4-E2B", "gemma4-e2b", "gemma4-e2b"),
     ("SmolLM2-1.7B", "smollm2-17b", "smollm2-17b")]
D = {n: json.load(open(O / f"xm_conf_{m}_conf-{k}.json")) for n, m, k in M}
rep = (X / "REPORT-CROSSMODEL.md").read_text()
P = lambda x: f"{100 * x:.0f}%"; Lo = lambda x: f"{100 * x:.0f}–"; P1 = lambda x: f"{100 * x:.1f}%"; D2 = lambda x: f"{x:.2f}"; D3 = lambda x: f"{x:.3f}"
bad = 0
def check(text, val):
    global bad
    ok = text in rep and val in text
    if not ok: bad += 1
    print("ok " if ok else "BAD", repr(val), "in", repr(text[:95]))
def claim(desc, ok):
    global bad
    if not ok: bad += 1
    print("ok " if ok else "BAD", desc)

roles = [b for d in D.values() for k, b in d["verdicts"].items() if k in ("C", "T", "S", "X", "R")]
gates = [d["G"]["gate_pass"] for d in D.values()]
t = "Every frozen test passed in all seven models: 29 of 29 role tests, and all 7 gates"
check(t, f"{sum(roles)} of {len(roles)}"); check(t, f"all {sum(gates)} gates"); claim("all role tests and gates pass", all(roles) and all(gates))
g = [d["G"]["auroc"] for d in D.values()]; check("(AUROC 0.984–0.988 at width 60)", D3(min(g))); check("(AUROC 0.984–0.988 at width 60)", D3(max(g)))
c = [d["C"]["share"] for d in D.values()]; check("Counters carry 59–76%", Lo(min(c))); check("Counters carry 59–76%", P(max(c)))
r = {n: d["R"]["share"] for n, d in D.items()}; check("Readers carry 48–113%", Lo(min(r.values()))); check("Readers carry 48–113%", P(max(r.values())))
rs = max(max(d[k]["random_share"]) for d in D.values() for k in ("C", "S", "X", "R") if k in d)
rm = max(d[k]["random_mean"] for d in D.values() for k in ("C", "S", "X", "R") if k in d)
rt = max(max(d["T"]["widths"][w]["random"]) for d in D.values() for w in ("40", "60", "80"))
t = "at most 9% on any single set, at most 2.3% averaged over a test's ten sets, and a token weight of at most 0.13"
check(t, P(rs)); check(t, P1(rm)); check(t, D2(rt))

def sets(d):
    s = d["spec"]["sets"]; st = lambda k: set(s.get(k, "").split(",")) - {""}
    return st("C"), st("T"), st("S") | st("X"), st("R")
ov = {n: sets(d) for n, d in D.items()}
nCT = sum(1 for C, T, SX, R in ov.values() if C & T); nCS = sum(1 for C, T, SX, R in ov.values() if C & SX)
nCR = sum(1 for C, T, SX, R in ov.values() if C & R); nCTS = sum(1 for C, T, SX, R in ov.values() if C & T & SX)
t = "In 6 of 7 models a counter head is also a token corrector, and in 6 of 7 a counter also writes the stored length. In 2 of 7 a counter is also a reader."
check(t, f"In {nCT} of 7 models a counter head is also a token corrector"); check(t, f"in {nCS} of 7 a counter also writes"); check(t, f"In {nCR} of 7 a counter is also a reader")
claim("OLMo is the model without C∩T and without C∩S/X", not (ov["OLMo-2-1B"][0] & ov["OLMo-2-1B"][1]) and not (ov["OLMo-2-1B"][0] & ov["OLMo-2-1B"][2])
      and [n for n, v in ov.items() if not (v[0] & v[1])] == ["OLMo-2-1B"] and [n for n, v in ov.items() if not (v[0] & v[2])] == ["OLMo-2-1B"])
claim("C∩R models are OLMo and Gemma", sorted(n for n, v in ov.items() if v[0] & v[3]) == ["Gemma-4-E2B", "OLMo-2-1B"])
claim("Gemma C∩R is L12H3", ov["Gemma-4-E2B"][0] & ov["Gemma-4-E2B"][3] == {"L12H3"})
claim("OLMo C∩R is L6H2, L7H10, L11H4", ov["OLMo-2-1B"][0] & ov["OLMo-2-1B"][3] == {"L6H2", "L7H10", "L11H4"})
three = {n: v[0] & v[1] & v[2] for n, v in ov.items() if v[0] & v[1] & v[2]}
check("**Four models have one head in three roles**", "Four" if len(three) == 4 else "XX")
claim("three-role heads as listed", three == {"Pythia-1B": {"L1H1"}, "Qwen2.5-0.5B": {"L2H1", "L3H1"}, "Gemma-4-E2B": {"L6H3"}, "SmolLM2-1.7B": {"L5H21"}})
claim("L4H1 in C and T in both Qwen3; L6H3 in Gemma; L5H21 in SmolLM2",
      "L4H1" in ov["Qwen3-1.7B"][0] & ov["Qwen3-1.7B"][1] and "L4H1" in ov["Qwen3-0.6B"][0] & ov["Qwen3-0.6B"][1]
      and "L6H3" in ov["Gemma-4-E2B"][0] & ov["Gemma-4-E2B"][1] and "L5H21" in ov["SmolLM2-1.7B"][0] & ov["SmolLM2-1.7B"][1])
claim("OLMo T set is L1H10, L0H5, L3H15", D["OLMo-2-1B"]["spec"]["sets"]["T"] == "L1H10,L0H5,L3H15")
e5 = json.load(open(O / "xm_e05_smollm2-17b_train_a.json"))["heads_share"]; check("(36%), but it was kept out", P(e5["L11H21"]))
claim("L11H21 not in SmolLM2 C set", "L11H21" not in ov["SmolLM2-1.7B"][0])

# dilution heads
E01 = {"Pythia-1B": "pythia-1b", "Qwen2.5-0.5B": "qwen05b", "OLMo-2-1B": "olmo1b", "Qwen3-1.7B": "qwen3-17b", "Qwen3-0.6B": "qwen3-06b", "Gemma-4-E2B": "gemma4-e2b", "SmolLM2-1.7B": "smollm2-17b"}
cr = []
for n, d in D.items():
    e = json.load(open(O / f"xm_e01_{E01[n]}_train.json"))["attn"]["corr_used"]
    for h in d["spec"]["sets"]["T"].split(","):
        l, hh = int(h[1:].split("H")[0]), int(h.split("H")[1]); cr.append((n, h, e[l][hh]))
neg = [z for z in cr if z[2] < -0.3]; pos = [z for z in cr if z[2] >= -0.3]
t = "In exploration, 14 of the 15 named token-corrector heads attend less"
check(t, f"{len(neg)} of the {len(cr)}"); check("In 14 of the 15 heads, attention to the newline falls", f"{len(neg)} of the {len(cr)}")
claim("the one exception is SmolLM2 L5H21 with positive correlation", [(z[0], z[1]) for z in pos] == [("SmolLM2-1.7B", "L5H21")] and pos[0][2] > 0)

# stored length
k = lambda n, key: D[n][key]["nat_kept"]
grp = [k("Pythia-1B", "S"), k("Qwen2.5-0.5B", "S"), k("SmolLM2-1.7B", "S")]
check("left the decision 85–93% of its separation", Lo(min(grp))); check("left the decision 85–93% of its separation", P(max(grp)))
grp = [k("Gemma-4-E2B", "S"), k("OLMo-2-1B", "S")]; check("It left 70–73% in Gemma-4-E2B", Lo(min(grp))); check("It left 70–73% in Gemma-4-E2B", P(max(grp)))
check("In OLMo-2-1B's L5H0 (57%)", P(k("OLMo-2-1B", "X")))
grp = [k("Qwen3-1.7B", "X"), k("Qwen3-0.6B", "X")]; check("the Qwen3 writers (42–52%)", Lo(min(grp))); check("the Qwen3 writers (42–52%)", P(max(grp)))
# readers
half = [r["Pythia-1B"], r["Gemma-4-E2B"], r["SmolLM2-1.7B"]]
check("(48–50%), 78% in both Qwen3 models, 96% in Qwen2.5-0.5B, and more than all of it in OLMo-2-1B (113%)", Lo(min(half)))
check("(48–50%), 78% in both Qwen3 models, 96% in Qwen2.5-0.5B, and more than all of it in OLMo-2-1B (113%)", P(max(half)))
claim("both Qwen3 readers 78%", P(r["Qwen3-1.7B"]) == "78%" and P(r["Qwen3-0.6B"]) == "78%")
check("96% in Qwen2.5-0.5B", P(r["Qwen2.5-0.5B"])); check("in OLMo-2-1B (113%)", P(r["OLMo-2-1B"]))
# head counts and layers
n = {nm: {kk: len(v.split(",")) for kk, v in d["spec"]["sets"].items()} for nm, d in D.items()}
check("Gemma-4-E2B carries its count with two heads (68%)", P(D["Gemma-4-E2B"]["C"]["share"])); claim("Gemma C has 2 heads", n["Gemma-4-E2B"]["C"] == 2)
others = [v["C"] for nm, v in n.items() if nm != "Gemma-4-E2B"]; check("against three to six in the others", {3: "three"}[min(others)] + " to " + {6: "six"}[max(others)])
allsz = [v for d in n.values() for v in d.values()]; claim("set sizes 1 to 6", min(allsz) == 1 and max(allsz) == 6)
claim("one-head sets: SmolLM2 R, Qwen3 T, Gemma T", sorted((nm, kk) for nm, d in n.items() for kk, v in d.items() if v == 1 and kk != "X") ==
      sorted([("SmolLM2-1.7B", "R"), ("Qwen3-1.7B", "T"), ("Qwen3-0.6B", "T"), ("Gemma-4-E2B", "T")]))
claim("six-head sets: Pythia C and S", sorted((nm, kk) for nm, d in n.items() for kk, v in d.items() if v == 6) == [("Pythia-1B", "C"), ("Pythia-1B", "S")])
heads = [(nm, h) for nm, d in D.items() for v in d["spec"]["sets"].values() for h in v.split(",")]
late = sorted({h for nm, h in heads if int(h[1:].split("H")[0]) > 14})
claim(f"all heads in layers 0–14 except {late}", late == ["L15H3", "L17H6", "L22H3"])
check("except three Qwen3 readers (L15H3, L17H6, L22H3)", "three")
# PROTOCOL-X2
x2 = json.load(open(O / "xm_c2_overshoot_olmo1b_validation.json")); claim("X2: all six predictions passed", all(x2["verdicts"].values()) and len(x2["verdicts"]) == 6)
print("\nPROSE CHECK:", "all claims match" if not bad else f"{bad} problems"); sys.exit(1 if bad else 0)
