"""D1-X cross-model summary tables for REPORT-CROSSMODEL.md, generated from the saved confirmation outputs
of every model (PROTOCOL-X1, PROTOCOL-X3, PROTOCOL-X3b). Usage: xm_crossmodel_tables.py [--check]
Writes outputs/xm_crossmodel_tables.md. With --check, every table line must appear verbatim in
REPORT-CROSSMODEL.md. Models without a confirmation output yet are listed as not run."""
import json, sys
from pathlib import Path
X = Path(__file__).resolve().parent.parent
# (name in tables, output file, protocol)
MODELS = [("Pythia-1B", "xm_conf_pythia-1b_conf-pythia1b.json", "X1"), ("Qwen2.5-0.5B", "xm_conf_qwen05b_conf-qwen05b.json", "X1"),
          ("OLMo-2-1B", "xm_conf_olmo1b_conf-olmo1b.json", "X1"), ("Qwen3-1.7B", "xm_conf_qwen3-17b_conf-qwen3-17b.json", "X3"),
          ("Qwen3-0.6B", "xm_conf_qwen3-06b_conf-qwen3-06b.json", "X3"), ("Gemma-4-E2B", "xm_conf_gemma4-e2b_conf-gemma4-e2b.json", "X3b"),
          ("SmolLM2-1.7B", "xm_conf_smollm2-17b_conf-smollm2-17b.json", "X3b")]
pc = lambda x: f"{100 * x:.0f}%"; f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"
PF = lambda b: "pass" if b else "fail"
D = {}
for name, f, prot in MODELS:
    p = X / "outputs" / f
    D[name] = (json.load(open(p)), prot) if p.exists() else (None, prot)

L = ["| Model | Protocol | Gate AUROC | C, counters | T, token correctors (per token, widths 40/60/80) | S, stored length (decision kept) | X, newline writers (decision kept) | R, readers | Verdicts |",
     "|---|---|---|---|---|---|---|---|---|"]
for name, _, _ in MODELS:
    d, prot = D[name]
    if d is None: L.append(f"| {name} | {prot} | not run | | | | | | |"); continue
    v = d["verdicts"]
    C = pc(d["C"]["share"]) if "C" in d else "–"
    T = " / ".join(f2(d["T"]["widths"][w]["ablated"]) for w in ("40", "60", "80")) if "T" in d else "–"
    S = f"{pc(d['S']['share'])} ({pc(d['S']['nat_kept'])})" if "S" in d else "–"
    Xs = f"{pc(d['X']['share'])} ({pc(d['X']['nat_kept'])})" if "X" in d else "–"
    R = pc(d["R"]["share"]) if "R" in d else "–"
    vs = ", ".join(f"{k} {PF(b)}" for k, b in v.items() if not k.startswith("T_w"))
    L.append(f"| {name} | {prot} | {f3(d['G']['auroc'])} | {C} | {T} | {S} | {Xs} | {R} | {vs} |")
L.append("")
L += ["| Model | Counters | Token correctors | Stored length / newline writers | Readers |", "|---|---|---|---|---|"]
for name, _, _ in MODELS:
    d, _ = D[name]
    if d is None: continue
    s = d["spec"]["sets"]; sp = lambda k: s.get(k, "").replace(",", ", ") or "–"
    store = "; ".join(x for x in ([f"S: {sp('S')}"] if "S" in s else []) + ([f"X: {sp('X')}"] if "X" in s else []))
    L.append(f"| {name} | {sp('C')} | {sp('T')} | {store} | {sp('R')} |")
L.append("")
L += ["| Model | Largest random-set share, C / S or X / R | Largest random token weight (any width) |", "|---|---|---|"]
for name, _, _ in MODELS:
    d, _ = D[name]
    if d is None: continue
    sx = max(max(d[k]["random_share"]) for k in ("S", "X") if k in d)
    tmax = max(max(d["T"]["widths"][w]["random"]) for w in ("40", "60", "80"))
    L.append(f"| {name} | {pc(max(d['C']['random_share']))} / {pc(sx)} / {pc(max(d['R']['random_share']))} | {f2(tmax)} |")

# Heads that appear in more than one role (from the frozen spec sets)
E01 = {"Pythia-1B": "pythia-1b", "Qwen2.5-0.5B": "qwen05b", "OLMo-2-1B": "olmo1b", "Qwen3-1.7B": "qwen3-17b", "Qwen3-0.6B": "qwen3-06b",
       "Gemma-4-E2B": "gemma4-e2b", "SmolLM2-1.7B": "smollm2-17b"}
L.append("")
L += ["| Model | Heads in C and T | Heads in C and S or X | Heads in C and R | Heads in C, T and S or X |", "|---|---|---|---|---|"]
OV = {}
for name, _, _ in MODELS:
    d, _ = D[name]
    if d is None: continue
    s = d["spec"]["sets"]; st = lambda k: set(s.get(k, "").split(",")) - {""}
    C, T, SX, R = st("C"), st("T"), st("S") | st("X"), st("R")
    srt = lambda z: ", ".join(sorted(z, key=lambda h: (int(h[1:].split("H")[0]), int(h.split("H")[1])))) or "none"
    OV[name] = (C & T, C & SX, C & R, C & T & SX)
    L.append(f"| {name} | {srt(C & T)} | {srt(C & SX)} | {srt(C & R)} | {srt(C & T & SX)} |")
n = len(OV); cnt = [sum(1 for v in OV.values() if v[i]) for i in range(4)]
L.append(f"| Models with at least one such head | {cnt[0]} of {n} | {cnt[1]} of {n} | {cnt[2]} of {n} | {cnt[3]} of {n} |")
# Token correctors' attention to the line-start newline (exploratory, WikiText-2 train, width 60)
L.append("")
L += ["| Model | Token corrector | Newline attention share | Correlation with characters used |", "|---|---|---|---|"]
ndil = 0; nT = 0
for name, _, _ in MODELS:
    d, _ = D[name]
    if d is None: continue
    e = json.load(open(X / "outputs" / f"xm_e01_{E01[name]}_train.json"))
    sh, cr = e["attn"]["share"], e["attn"]["corr_used"]
    for h in d["spec"]["sets"]["T"].split(","):
        l, hh = int(h[1:].split("H")[0]), int(h.split("H")[1]); nT += 1; ndil += cr[l][hh] < -0.3
        L.append(f"| {name} | {h} | {sh[l][hh]:.2f} | {cr[l][hh]:+.2f} |")
L.append(f"| All | {nT} heads | | {ndil} of {nT} below −0.30 |")
out = X / "outputs" / "xm_crossmodel_tables.md"; out.write_text("\n".join(L) + "\n"); print("\n".join(L))
if "--check" in sys.argv:
    rep = (X / "REPORT-CROSSMODEL.md").read_text(); miss = [l for l in L if l.strip() and l not in rep]
    print("\nCHECK:", "all table lines present" if not miss else f"{len(miss)} missing:\n" + "\n".join(miss)); sys.exit(1 if miss else 0)
