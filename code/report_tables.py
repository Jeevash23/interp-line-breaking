"""Builds the confirmation tables for REPORT.md from the saved validation outputs, and checks
that every table line appears verbatim in REPORT.md (exit 1 if not). Run: python3 report_tables.py [--check]"""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parent.parent; O = R / "outputs"
a = json.loads((O / "conf01_token_validation.json").read_text())
b = json.loads((O / "conf02_width_validation.json").read_text())
f = lambda x: f"{x:.3f}"; ci = lambda c: f"({c[0]:.3f}, {c[1]:.3f})"
L = ["| Width | Points | Intact | L5H7 ablated | L5H3 ablated | L5H11 ablated | L5H14 ablated | H1a | H1b | H1c |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for W, r in a["results"].items():
    v = a["verdicts"][W]; pf = lambda x: "pass" if x else "fail"
    L.append(f"| {W} | {r['intact']['n_points']} | " + " | ".join(f"{f(r[c]['token_coef'])} {ci(r[c]['ci95'])}" for c in ("intact", "abl_L5H7", "abl_L5H3", "abl_L5H11", "abl_L5H14"))
             + f" | {pf(v['H1a'])} | {pf(v['H1b'])} | {pf(v['H1c'])} |")
L += ["", f"H1 overall: {'pass' if a['H1_pass'] else 'fail'}", ""]
L += [f"Width effect (n = {b['n']}): {f(b['width_effect'])} {ci(b['width_effect_ci95'])}", "",
      "| Frozen at the decision token | Mediation share (95% CI) |", "|---|---|",
      f"| L8H15 | {f(b['shares']['L8H15']['share'])} {ci(b['shares']['L8H15']['ci95'])} |",
      f"| L8H15, L17H1, L6H2 | {f(b['shares']['readers3']['share'])} {ci(b['shares']['readers3']['ci95'])} |",
      f"| 10 random triples, mean | {f(b['random_mean_share'])} |"]
L += ["", f"H2 overall: {'pass' if b['H2_pass'] else 'fail'}"]
txt = "\n".join(L)
(O / "report_tables.md").write_text(txt + "\n")
if "--check" in sys.argv:
    rep = (R / "REPORT.md").read_text()
    missing = [l for l in L if l.strip() and l not in rep]
    print("missing lines:", missing if missing else "none"); sys.exit(1 if missing else 0)
print(txt)
