"""Builds the result tables for REPORT-3.md from the saved round-3 (new text) outputs, and checks that every
table line appears verbatim in REPORT-3.md (exit 1 if not). Run: python3 report3_tables.py [--check]"""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parent.parent; O = R / "outputs"
a = json.loads((O / "c3_01_token_conf-pythia410m.json").read_text())
b = json.loads((O / "c3_02_width_conf-pythia410m.json").read_text())
c = json.loads((O / "c3_03_newline_conf-pythia410m.json").read_text())
d = json.loads((O / "c3_04_paths_conf-pythia410m.json").read_text())
f = lambda x: f"{x:.3f}"; ci = lambda c_: f"({c_[0]:.3f}, {c_[1]:.3f})"; pf = lambda x: "pass" if x else "fail"
pc = lambda x: f"{100 * x:.1f}%"; pci = lambda c_: f"({100 * c_[0]:.1f}%, {100 * c_[1]:.1f}%)"
L = ["**R1 and H3.** Token coefficient (95% paragraph-bootstrap interval).", "",
     "| Width | Points | Intact | L5H7 ablated | L5H3 ablated | L5H11 ablated | L5H14 ablated | R1a | R1b | R1c |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for W, r in a["results"].items():
    v = a["verdicts"][W]
    L.append(f"| {W} | {r['intact']['n_points']} | " + " | ".join(f"{f(r[k]['token_coef'])} {ci(r[k]['ci95'])}" for k in ("intact", "abl_L5H7", "abl_L5H3", "abl_L5H11", "abl_L5H14"))
             + f" | {pf(v['H1a'])} | {pf(v['H1b'])} | {pf(v['H1c'])} |")
L += ["", f"Original H1 (all parts, all widths, frozen 0.10 limit): {pf(a['R1_pass'])}; H1a at width 40 under the 0.10 limit: {pf(a['R1a_width40_pass'])}",
      f"Round-3 prediction for width 40, intact lower bound above 0: {pf(a['residual_w40_above_zero'])}",
      "T1 (round-3 token-correction claim: H1b and H1c at all widths, H1a at widths 60 and 80, width-40 residual above 0): "
      + pf(all(a['verdicts'][W]['H1b'] and a['verdicts'][W]['H1c'] for W in a['verdicts']) and a['verdicts']['60']['H1a'] and a['verdicts']['80']['H1a'] and a['residual_w40_above_zero']), "",
      "| Width | L4H14 ablated | L4H14 minus intact (paired) | H3 | L4H14 + L5H7 ablated | Joint minus L5H7 alone (paired) |", "|---|---|---|---|---|---|"]
for W, r in a["results"].items():
    p = a["paired"][W]
    L.append(f"| {W} | {f(r['abl_L4H14']['token_coef'])} {ci(r['abl_L4H14']['ci95'])} | {f(p['L4H14_minus_intact']['diff'])} {ci(p['L4H14_minus_intact']['ci95'])} | {pf(a['verdicts'][W]['H3'])} | "
             f"{f(r['abl_L4H14_L5H7']['token_coef'])} {ci(r['abl_L4H14_L5H7']['ci95'])} | {f(p['joint_minus_L5H7']['diff'])} {ci(p['joint_minus_L5H7']['ci95'])} |")
L += ["", f"H3 overall: {pf(a['H3_pass'])}", "",
      f"**R2.** Width effect (n = {b['n']}): {f(b['width_effect'])} {ci(b['width_effect_ci95'])}", "",
      "| Frozen at the decision token | Mediation share (95% CI) |", "|---|---|",
      f"| L8H15 | {f(b['shares']['L8H15']['share'])} {ci(b['shares']['L8H15']['ci95'])} |",
      f"| L8H15, L17H1, L6H2 | {f(b['shares']['readers3']['share'])} {ci(b['shares']['readers3']['ci95'])} |",
      f"| 10 random triples, mean | {f(b['random_mean_share'])} |", "", f"R2 overall: {pf(b['H2_pass'])}", ""]
h4 = c["H4_width"]; dd = c["H4_dissociation"]; v = c["verdicts"]
L += [f"**H4.** Width effect, intact (n = {h4['n']}): {f(h4['width_effect'])} {ci(h4['width_effect_ci95'])}", "",
      "| Newline-only ablation | Share of the width effect removed (95% CI) |", "|---|---|",
      f"| COUNT4, every newline | {pc(h4['shares']['COUNT4_nl']['share'])} {pci(h4['shares']['COUNT4_nl']['ci95'])} |",
      f"| COUNT4, last newline only | {pc(h4['shares']['COUNT4_last']['share'])} {pci(h4['shares']['COUNT4_last']['ci95'])} |",
      f"| 10 random sets of 4 heads, mean | {pc(h4['random_mean_share'])} |", "",
      f"| Natural text, width 60 ({dd['n_paragraphs']} paragraphs) | Separation | AUROC | Kept |", "|---|---|---|---|",
      f"| Intact | {f(dd['intact']['separation'])} | {f(dd['intact']['auroc'])} | |",
      f"| COUNT4 at newlines only | {f(dd['COUNT4_nl']['separation'])} | {f(dd['COUNT4_nl']['auroc'])} | {pc(dd['COUNT4_nl']['kept'])} |",
      f"| COUNT4 at every position | {f(dd['COUNT4_all']['separation'])} | {f(dd['COUNT4_all']['auroc'])} | {pc(dd['COUNT4_all']['kept'])} |", "",
      f"H4a {pf(v['H4a'])}, H4b {pf(v['H4b'])}, H4c {pf(v['H4c'])}, H4d {pf(v['H4d'])}; H4 overall: {pf(c['H4_pass'])}", ""]
s = d["summary"]; vd = d["verdicts"]
L += [f"**H5.** Width effect, eager attention (n = {d['n']}): {f(s['width_effect']['mean'])} {ci(s['width_effect']['ci95'])}", "",
      "| Head | Key only | Value only |", "|---|---|---|"]
for h in ("L8H15", "L17H1", "L8H11"):
    L.append(f"| {h} | {f(s[h]['K']['mean'])} {ci(s[h]['K']['ci95'])} | {f(s[h]['V']['mean'])} {ci(s[h]['V']['ci95'])} |")
L += ["", f"L8H15 value minus key: {f(s['L8H15_V_minus_K']['mean'])} {ci(s['L8H15_V_minus_K']['ci95'])}; L17H1 key minus value: {f(s['L17H1_K_minus_V']['mean'])} {ci(s['L17H1_K_minus_V']['ci95'])}",
      "", f"H5a {pf(vd['H5a'])}, H5b {pf(vd['H5b'])}; H5 overall: {pf(d['H5_pass'])}", "",
      "**H6.** Coefficient of the previous line's token count (95% paragraph-bootstrap interval).", "",
      "| Width | Points | Intact | L5H7 at newlines | L5H11 at newlines | L5H7 minus intact (paired) | L5H11 minus intact | H6a | H6b |", "|---|---|---|---|---|---|---|---|---|"]
for W, r in c["H6"].items():
    va = pf(v[f"H6a_w{W}"]) if f"H6a_w{W}" in v else "not assessed"; vb = pf(v[f"H6b_w{W}"]) if f"H6b_w{W}" in v else "not assessed"
    L.append(f"| {W} | {r['intact']['n_points']} | {f(r['intact']['prev_ntok'])} {ci(r['intact']['ci95'])} | {f(r['L5H7_nl']['prev_ntok'])} {ci(r['L5H7_nl']['ci95'])} | "
             f"{f(r['L5H11_nl']['prev_ntok'])} {ci(r['L5H11_nl']['ci95'])} | {f(r['L5H7_nl_minus_intact']['diff'])} {ci(r['L5H7_nl_minus_intact']['ci95'])} | {f(r['L5H11_nl_minus_intact']['diff'])} | {va} | {vb} |")
L += ["", f"H6 overall: {pf(c['H6_pass'])}"]
txt = "\n".join(L)
(O / "report3_tables.md").write_text(txt + "\n")
if "--check" in sys.argv:
    rep = (R / "REPORT-3.md").read_text()
    missing = [l for l in L if l.strip() and l not in rep]
    print("missing lines:", missing if missing else "none"); sys.exit(1 if missing else 0)
print(txt)
