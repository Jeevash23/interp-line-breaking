"""D1-X Amendment 3, exp e05 (EXPLORATORY, CHARTER.md): which heads carry the current line's count to the decision?
Count interchange. Usage: xm_e05_count.py <model> <part> [--sets "A,B;C,D"] [--tag T] [--smoke]
Items: width 60, one per paragraph (random choices seed 300): a line k >= 2 and a letter-ending
decision word with 4..14 characters left (as e02). The short prompt drops leading words of that
line until >= 10 characters are gone (at least 2 words kept); earlier lines are identical, so
the width information is the same. Total effect = log P(newline) at the decision, short minus base
(negative: fewer characters used).
 A residual at the decision token after layer l patched from the short run: share of the effect.
 B each head's decision-token output patched from the short run (single heads): share.
 C each candidate set patched together, against 10 layer-matched random sets (no repeats).
Share = mean(patched effect) / mean(total effect). Item bootstrap, 1,000 resamples, seed 0.
"""
import sys, json, random, numpy as np, torch
from xm3_common import *

MODEL, PART = sys.argv[1], sys.argv[2]; SMOKE = "--smoke" in sys.argv
def arg(k, d=None): return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
SETS = [[tuple(int(x) for x in t[1:].split("H")) for t in s.split(",")] for s in arg("--sets", "").split(";") if s]
TAG = arg("--tag", "a")
SETS_ONLY = "--sets-only" in sys.argv  # Amendment 3: for set runs on large models; parts A and C unchanged
m = M(MODEL); tok = m.tok
paras = load_paras(PART, seed=0)
N, NB = (6, 50) if SMOKE else (100, 1000)
W = 60; rng = random.Random(300); items = []
for p in paras:
    if len(items) >= N: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 5: continue
    k = rng.randrange(2, len(lines) - 1); words = lines[k].split(" ")
    js = [j for j in range(3, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    cur = words[: js[0] + 1]; d = 0
    while len(cur) > 2 and d < 10: d += len(cur[0]) + 1; cur = cur[1:]
    if d < 10: continue
    ib = tok("\n".join(lines[:k]) + "\n" + " ".join(words[: js[0] + 1]), add_special_tokens=False)["input_ids"]
    is_ = tok("\n".join(lines[:k]) + "\n" + " ".join(cur), add_special_tokens=False)["input_ids"]
    if ib[-1] != is_[-1]: continue
    items.append((ib, is_))
pct = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
HEADS = m.heads(); CH = 8 if MODEL in ("gemma4-e2b", "smollm2-17b") else 32

# Amendment 3: Gemma's and SmolLM2's batched calls run in chunks of 8 rows (full batches of long prompts overran the GPU); other models unchanged.
CB = 8 if MODEL in ("gemma4-e2b", "smollm2-17b") else 10**6
def resid_rows(ib, T, rs_, lb):
    """log P(newline) at the last position with each layer's residual at position T patched (one batch row per layer), minus lb."""
    out = []
    for a in range(0, m.L, CB):
        ls = list(range(a, min(a + CB, m.L))); m.rpatch = {l: [(T, rs_[l], l - a)] for l in ls}
        out.append(m.lp_last_batch(ib, len(ls))); m.rpatch = {}
    return np.concatenate(out) - lb
tot, A, B, cache = [], [], [], []
for ib, is_ in items:
    m.patch, m.rpatch = {}, {}
    lb = m.lp(ib)[-1]
    ls = m.lp(is_, capture=True)[-1]; zs = {l: m.cap[l][-1].clone() for l in range(m.L)}; rs_ = {l: m.rcap[l][-1].clone() for l in range(m.L)}
    tot.append(ls - lb); T = len(ib) - 1
    A.append(resid_rows(ib, T, rs_, lb))
    row = np.zeros(len(HEADS))
    for s in (range(0, len(HEADS), CH) if not SETS_ONLY else []):  # --sets-only: skip the single-head patches (B), as already saved in run a
        part = HEADS[s:s + CH]; m.patch = {}
        for r, (l, h) in enumerate(part): m.patch.setdefault(l, []).append((h, T, zs[l][m.hs(l, h)], r))
        row[s:s + len(part)] = m.lp_last_batch(ib, len(part)) - lb
    m.patch = {}; B.append(row); cache.append((ib, lb, zs))
tot, A, B = np.array(tot), np.array(A), np.array(B)
n = len(tot); br = np.random.default_rng(0); idx = [br.integers(0, n, n) for _ in range(NB)]
names = [f"L{l}H{h}" for l, h in HEADS]; Bs = B.mean(0) / tot.mean()
out = dict(model=MODEL, part=PART, smoke=SMOKE, tag=TAG, n=n, total=float(tot.mean()), total_ci95=pct([tot[i].mean() for i in idx]),
           resid_share_by_layer=(A.mean(0) / tot.mean()).tolist(), heads_share=dict(zip(names, Bs.tolist())),
           top_heads=[(names[i], float(Bs[i])) for i in np.argsort(-Bs)[:20]])
print(json.dumps({k: out[k] for k in ("n", "total", "resid_share_by_layer", "top_heads")}), flush=True)

rs = np.random.default_rng(0)
def setshare(S):
    e = []
    for ib, lb, zs in cache:
        m.patch = {}
        for l, h in S: m.patch.setdefault(l, []).append((h, len(ib) - 1, zs[l][m.hs(l, h)]))
        e.append(m.lp(ib)[-1] - lb); m.patch = {}
    return np.array(e)
out["sets"] = []
ALLC = {h for S in SETS for h in S}
for S in SETS:
    e = setshare(S); R, RN = [], []
    for _ in range(10):
        r = []
        for l, _h in S: r.append((l, int(rs.choice([h for h in range(m.H) if (l, h) not in ALLC and (l, h) not in r]))))
        R.append(float(setshare(r).mean() / tot.mean())); RN.append(",".join(f"L{l}H{h}" for l, h in r))
    out["sets"].append(dict(set=",".join(f"L{l}H{h}" for l, h in S), share=float(e.mean() / tot.mean()),
                            share_ci95=pct([e[i].mean() / tot[i].mean() for i in idx]), random_share=R, random_sets=RN))
    print("set", json.dumps(out["sets"][-1]), flush=True)
save_json(out, f"xm_e05_{MODEL}_{PART}_{TAG}{'_smoke' if SMOKE else ''}.json")
