"""D1-X exp e00 (EXPLORATORY; AMENDMENTS.md Amendment 1): precision spot check.
Usage: [XM_DTYPE=float32 XM_DEVICE=cpu] xm_e00_precision.py <model> <label>
Newline log-odds at every letter-ending decision point of the first 12 WikiText-2 train
paragraphs (seed 0) wrapped at width 60, plus AUROC and separation (as e01). Saved per point so
two precisions can be compared point by point (xm_e00_compare.py)."""
import sys, numpy as np, torch
from xm_common import *
MODEL, LABEL = sys.argv[1], sys.argv[2]
torch.set_num_threads(8)
m = M(MODEL); tok = m.tok
sc, lab, rem = [], [], []
for ids, recs, _ in wrapped_items(load_paras("train", seed=0), 60, 12, tok=tok):
    lp = m.lp(ids)
    for r in recs: sc.append(float(logodds(lp[r["i"]]))); lab.append(r["label"]); rem.append(r["remaining"])
sc, lab, rem = np.array(sc), np.array(lab), np.array(rem)
sep = float(sc[lab == 1].mean() - sc[(lab == 0) & (rem <= 12)].mean())
save_json(dict(model=MODEL, label=LABEL, dtype=str(next(m.model.parameters()).dtype), n=len(sc), auroc=auroc(sc, lab), separation=sep,
               logodds=sc.tolist(), labels=lab.tolist()), f"xm_e00_{MODEL}_{LABEL}.json")
print(MODEL, LABEL, len(sc), auroc(sc, lab), sep)
