"""D1-X Amendment 3: compare the precision spot-check outputs (xm_e00_<model>_<label>.json) of two runs.
Usage: xm3_e00_compare.py <model> <label_a> <label_b>. Prints and saves AUROC, separation, the
point-by-point correlation and the largest absolute difference of the newline log-odds."""
import sys, json, numpy as np
from pathlib import Path
O = Path(__file__).resolve().parent.parent / "outputs"
m, a, b = sys.argv[1:4]
A, B = (json.load(open(O / f"xm_e00_{m}_{x}.json")) for x in (a, b))
assert A["labels"] == B["labels"], "different decision points"
x, y = np.array(A["logodds"]), np.array(B["logodds"])
r = dict(model=m, a=a, b=b, n=len(x), auroc=[A["auroc"], B["auroc"]], separation=[A["separation"], B["separation"]],
         r=float(np.corrcoef(x, y)[0, 1]), max_abs_diff=float(np.abs(x - y).max()), mean_abs_diff=float(np.abs(x - y).mean()))
(O / f"xm3_e00_compare_{m}.json").write_text(json.dumps(r, indent=1)); print(r)
