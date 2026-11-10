"""Post hoc on exp16's saved rows: head votes, residual patches and behaviour against the
number of extra tokens a re-spelling adds (640 item-by-family pairs). Exploratory."""
import json
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent.parent / "outputs"
R = json.loads((OUT / "exp16_respelling_families.json").read_text())["rows"]
X = np.array([[v["ntok_extra"], v["votes"]["L5H7"], v["votes"]["L5H6"], v["votes"]["L4H14"], v["behaviour"], v["resid"]["4"], v["resid"]["5"]]
              for r in R for f, v in r.items() if f != "rem"])
names = ["L5H7_vote", "L5H6_vote", "L4H14_vote", "behaviour", "resid_after_L4", "resid_after_L5"]
res = {"n": len(X)}
for q, nm in enumerate(names, start=1):
    res[nm] = dict(corr_with_extra_tokens=float(np.corrcoef(X[:, 0], X[:, q])[0, 1]), slope_per_token=float(np.polyfit(X[:, 0], X[:, q], 1)[0]))
(OUT / "posthoc_exp16_tokens.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
