"""Builds the training-dynamics table (exploratory, exp26) for REPORT-4.md from outputs/exp26_checkpoint_<step>.json
and checks it verbatim against REPORT-4.md with --check. Not part of the frozen protocol."""
import sys, json
from pathlib import Path
R = Path(__file__).resolve().parent.parent; O = R / "outputs"
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"
L = ["| Step | AUROC | Separation | L5H6 share to line start | Separation kept, COUNT4 ablated | L5H7 correction | Width share removed: COUNT4 at newlines | readers | MLPs 14, 16, 18 | MLPs 21, 23 |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for s, lab in (("1000", "1k"), ("2000", "2k"), ("4000", "4k"), ("8000", "8k"), ("16000", "16k"), ("48000", "48k"), ("final", "143k (final)")):
    d = json.loads((O / f"exp26_checkpoint_{s}.json").read_text()); i = d["intact"]
    L.append(f"| {lab} | {f3(i['auroc'])} | {f2(i['sep'])} | {f2(i['L5H6_share'])} | {f2(d['counters_sep_kept'])} | {f3(d['corrector_token_coef_change'])} | "
             f"{f2(d['width_share_removed_COUNT4_nl'])} | {f2(d['width_share_removed_readers3'])} | {f2(d['width_share_removed_MLP14_16_18'])} | {f2(d['width_share_removed_MLP21_23'])} |")
if "--check" in sys.argv:
    doc = (R / "REPORT-4.md").read_text(); miss = [l for l in L if l not in doc]
    print("missing lines:", miss if miss else "none"); sys.exit(1 if miss else 0)
print("\n".join(L))
