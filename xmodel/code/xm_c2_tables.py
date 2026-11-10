"""Table for the PROTOCOL-X2 result, from outputs/xm_c2_overshoot_olmo1b_validation.json.
Usage: xm_c2_tables.py [--check]   (--check: every line must appear verbatim in NOTEBOOK.md)"""
import json, sys
from pathlib import Path
X = Path(__file__).resolve().parent.parent
d = json.load(open(X / "outputs" / "xm_c2_overshoot_olmo1b_validation.json")); v = d["verdicts"]
pc = lambda x: f"{100 * x:.1f}%"; pci = lambda c: f"({100 * c[0]:.1f}%, {100 * c[1]:.1f}%)"; PF = lambda b: "pass" if b else "fail"
p2 = d["P2"]
L = [f"Width items: {d['n']}; patched effect {d['Pw']:.3f} ({d['Pw_ci95'][0]:.3f}, {d['Pw_ci95'][1]:.3f}); frozen-reader effect {d['Fr']:.3f} ({d['Fr_ci95'][0]:.3f}, {d['Fr_ci95'][1]:.3f}).", "",
     "| Prediction | Measure | Value (95% CI) | Verdict |", "|---|---|---|---|",
     f"| P1 | Reader mediation share | {pc(d['A_share']['share'])} {pci(d['A_share']['ci95'])} | {PF(v['P1'])} |",
     f"| P2 | Rise in frozen effect when L5H11 is also frozen | {pc(p2['dFr_L5H11'])} {pci(p2['ci95'])}; rank {p2['rank_L5H11']}; 10 random heads mean {pc(p2['random_mean'])} | {PF(v['P2'])} |",
     f"| P3 | Mediation share of readers plus L5H11 | {pc(d['P3']['share'])} {pci(d['P3']['ci95'])} | {PF(v['P3'])} |",
     f"| P4 | L5H11 injected alone | {pc(d['P4']['inject_L5H11'])} {pci(d['P4']['ci95'])} | {PF(v['P4'])} |",
     f"| P5 | Reader share, mean-ablation / resample-ablation | {pc(d['D_mean_ablation']['share'])} {pci(d['D_mean_ablation']['ci95'])} / {pc(d['D_resample_ablation']['share'])} {pci(d['D_resample_ablation']['ci95'])} | {PF(v['P5'])} |",
     f"| P6 | L6H2 value only / key only at the marker | {pc(d['F']['value']['frac'])} {pci(d['F']['value']['ci95'])} / {pc(d['F']['key']['frac'])} {pci(d['F']['key']['ci95'])} | {PF(v['P6'])} |"]
print("\n".join(L))
if "--check" in sys.argv:
    nb = (X / "NOTEBOOK.md").read_text(); miss = [l for l in L if l.strip() and l not in nb]
    print("\nCHECK:", "all table lines present" if not miss else f"{len(miss)} missing"); sys.exit(1 if miss else 0)
