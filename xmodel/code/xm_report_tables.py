"""Tables for xmodel/REPORT-<model>.md, generated from the saved confirmation output.
Usage: xm_report_tables.py <model> [--check]
Writes outputs/xm_report_tables_<model>.md. With --check, verifies that every table line appears
verbatim in REPORT-<model>.md and lists any that do not."""
import json, sys
from pathlib import Path
X = Path(__file__).resolve().parent.parent
MODEL = sys.argv[1]
KEY = {"pythia-1b": "pythia1b", "qwen05b": "qwen05b", "olmo1b": "olmo1b"}[MODEL]
TEST = "--file" in sys.argv  # pipeline test on another output; writes to the scratchpad, never to outputs/
d = json.load(open(sys.argv[sys.argv.index("--file") + 1] if TEST else X / "outputs" / f"xm_conf_{MODEL}_conf-{KEY}.json"))
v = d["verdicts"]; TH = d["spec"]["thresholds"]
f3 = lambda x: f"{x:.3f}"; ci = lambda c: f"({c[0]:.3f}, {c[1]:.3f})"; pc = lambda x: f"{100 * x:.1f}%"; pci = lambda c: f"({100 * c[0]:.1f}%, {100 * c[1]:.1f}%)"
PF = lambda b: "pass" if b else "fail"
L = []
G = d["G"]
L += ["| Behaviour, width 60 | Value |", "|---|---|", f"| Paragraphs | {G['n_paragraphs']} |", f"| Decision points | {G['n_points']} |",
      f"| AUROC | {f3(G['auroc'])} |", f"| Separation | {f3(G['separation'])} |", f"| Gate (AUROC ≥ 0.90) | {PF(G['gate_pass'])} |", ""]
rows = ["| Test | Set | n | Effect (95% CI) | Share (95% CI) | Random sets, mean (max) | Verdict |", "|---|---|---|---|---|---|---|"]
if "C" in d:
    c = d["C"]; rows.append(f"| C, counters | {c['set']} | {c['n']} | {f3(c['total'])} {ci(c['total_ci95'])} | {pc(c['share'])} {pci(c['share_ci95'])} | {pc(c['random_mean'])} ({pc(max(c['random_share']))}) | {PF(v['C'])} |")
for k, lab in (("S", "S, stored length"), ("X", "X, stored length (second set)")):
    if k in d:
        s = d[k]; rows.append(f"| {lab} | {s['set']} | {s['n']} | {f3(s['width_effect'])} {ci(s['width_effect_ci95'])} | {pc(s['share'])} {pci(s['share_ci95'])} | {pc(s['random_mean'])} ({pc(max(s['random_share']))}) | {PF(v[k])} |")
if "R" in d:
    r = d["R"]; rows.append(f"| R, readers (Lw {r['Lw']}) | {r['set']} | {r['n']} | {f3(r['patch_effect'])} {ci(r['patch_effect_ci95'])} | {pc(r['share'])} {pci(r['share_ci95'])} | {pc(r['random_mean'])} ({pc(max(r['random_share']))}) | {PF(v['R'])} |")
L += rows + [""]
kept = ["| Newline-only ablation | Natural separation kept (95% CI) | AUROC |", "|---|---|---|"]
for k in ("S", "X"):
    if k in d: s = d[k]; kept.append(f"| {k}: {s['set']} | {pc(s['nat_kept'])} {pci(s['nat_kept_ci95'])} | {f3(s['nat_auroc'])} |")
if len(kept) > 2: L += kept + [""]
if "T" in d:
    L += [f"T, token correctors: {d['T']['set']} ablated at every position.", "",
          "| Width | Points | Intact (95% CI) | Ablated (95% CI) | Ablated minus intact (95% CI) | Random sets, mean (max) | Verdict |", "|---|---|---|---|---|---|---|"]
    for W, t in d["T"]["widths"].items():
        L.append(f"| {W} | {t['n_points']} | {f3(t['intact'])} {ci(t['intact_ci95'])} | {f3(t['ablated'])} {ci(t['ablated_ci95'])} | {f3(t['diff'])} {ci(t['diff_ci95'])} | {f3(t['random_mean'])} ({f3(max(t['random']))}) | {PF(v[f'T_w{W}'])} |")
    L.append("")
L.append("Verdicts: " + ", ".join(f"{k} {PF(b)}" for k, b in v.items() if not k.startswith("T_w")))
out = Path("/tmp/tables_test.md") if TEST else X / "outputs" / f"xm_report_tables_{MODEL}.md"
out.write_text("\n".join(L) + "\n")
print("\n".join(L))
if "--check" in sys.argv:
    rep = (X / f"REPORT-{MODEL}.md").read_text()
    miss = [l for l in L if l.strip() and l not in rep]
    print("\nCHECK:", "all table lines present" if not miss else f"{len(miss)} missing:\n" + "\n".join(miss))
    sys.exit(1 if miss else 0)
