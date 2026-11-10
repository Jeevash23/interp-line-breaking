"""c2_02 (CONFIRMATORY, PROTOCOL-2.md R2): replication of PROTOCOL.md H2 on the test split.
A copy of the frozen code/conf02_width.py; the only changes are this docstring, the split names
allowed and the output file name. Construction, thresholds and seeds are unchanged.

Usage: c2_02_width.py test              (the frozen run)
       c2_02_width.py train --smoke     (pipeline check only; 8 items)
Items (width 60, one per paragraph, seed-100 paragraph order, random choices seed 200): line k
drawn from 3..min(5, n_lines-2); the first word end j >= 2 on it with 4..14 characters
remaining ending in a letter; previous line shortened by dropping final words until >= 10
characters shorter (item skipped otherwise, or if the current line tokenises differently).
WIDTH effect: log P(newline) when the base run's line-start newline residual after layer 5 is
replaced by its state in the short-previous-line run, minus base.
Freezing: the named heads' attention-output input slices at the decision token are held at
their values in the unedited base run while the width patch is applied.
Mediation share = 1 - mean(frozen effect) / mean(width effect).
Groups: L8H15 alone; {L8H15, L17H1, L6H2}; 10 random triples of heads from layers 6-23
excluding those three (drawn once, numpy seed 0). 95% intervals: 2,000 item bootstrap, seed 0.
"""
import sys, json, random, numpy as np, torch
from common import *

SPLIT = sys.argv[1]; SMOKE = "--smoke" in sys.argv
assert SPLIT in ("train", "test")
NITEMS = 8 if SMOKE else 150
NBOOT = 50 if SMOKE else 2000
W = 60
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
READ = [(8, 15), (17, 1), (6, 2)]
pool = [(l, h) for l in range(6, 24) for h in range(Hn) if (l, h) not in READ]
rs = np.random.default_rng(0)
RANDOM = [[pool[i] for i in rs.choice(len(pool), 3, replace=False)] for _ in range(10)]
GROUPS = {"L8H15": [(8, 15)], "readers3": READ}
for q, tr in enumerate(RANDOM): GROUPS[f"random{q}"] = tr

cap, res5 = {}, {}
for l in range(6, 24):
    layers[l].attention.dense.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(l, a[0][:, -1].detach().clone()))
layers[5].register_forward_hook(lambda m, a, o: res5.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().clone()))
FREEZE, EDIT = {}, {}
def fz(l):
    def f(m, a):
        lst = FREEZE.get(l)
        if not lst: return None
        z = a[0].clone()
        for h, vec in lst: z[:, -1, h * dh:(h + 1) * dh] = vec
        return (z,)
    return f
for l in range(6, 24): layers[l].attention.dense.register_forward_pre_hook(fz(l))
def ed(m, a, o):
    if not EDIT: return None
    x = (o[0] if isinstance(o, tuple) else o).clone(); x[:, EDIT["pos"]] = EDIT["vec"]
    return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
layers[5].register_forward_hook(ed)
def lp_of(ids):
    with torch.no_grad():
        hid = model.gpt_neox(torch.tensor([ids], device="cuda")).last_hidden_state[0, -1]
        return torch.log_softmax(model.lm_head(hid).float(), -1)[nl].item()

rng = random.Random(200)
rows = []
for p in split_paragraphs(SPLIT, seed=100):
    if len(rows) >= NITEMS: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1))
    words = lines[k].split(" ")
    js = [j for j in range(2, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    prefix = " ".join(words[: js[0] + 1])
    before = lines[:k]
    pw = before[-1].split(" "); short = before[-1]
    while len(pw) > 2 and len(before[-1]) - len(short) < 10:
        pw = pw[:-1]; short = " ".join(pw)
    if len(before[-1]) - len(short) < 10: continue
    ib = tok("\n".join(before) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    is_ = tok("\n".join(before[:-1] + [short]) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    nb = [i for i, t in enumerate(ib) if t == nl][-1]; ns = [i for i, t in enumerate(is_) if t == nl][-1]
    if len(ib) - nb != len(is_) - ns: continue
    FREEZE.clear(); EDIT.clear()
    lb = lp_of(ib); clean = {l: v[0].clone() for l, v in cap.items()}
    lp_of(is_); vnl = res5["x"][0, ns].clone()
    EDIT.update(pos=nb, vec=vnl)
    lw = lp_of(ib)
    row = dict(eff=lw - lb)
    for gname, heads in GROUPS.items():
        FREEZE.clear()
        for l, h in heads: FREEZE.setdefault(l, []).append((h, clean[l][h * dh:(h + 1) * dh]))
        row[gname] = lp_of(ib) - lb
    FREEZE.clear(); EDIT.clear()
    rows.append(row)
    if len(rows) % 25 == 0: print(len(rows), "items", flush=True)

E = np.array([r["eff"] for r in rows])
br = np.random.default_rng(0); idx = [br.integers(0, len(rows), len(rows)) for _ in range(NBOOT)]
def share(g):
    F = np.array([r[g] for r in rows]); s = 1 - F.mean() / E.mean()
    bs = [1 - F[i].mean() / E[i].mean() for i in idx]
    return dict(share=float(s), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
eff_ci = [float(np.percentile([E[i].mean() for i in idx], q)) for q in (2.5, 97.5)]
shares = {g: share(g) for g in GROUPS}
rand_mean = float(np.mean([shares[f"random{q}"]["share"] for q in range(10)]))
pos_ok = eff_ci[0] > 0
verd = dict(positive_control=bool(pos_ok),
            H2_L8H15=bool(pos_ok and shares["L8H15"]["share"] >= 0.25),
            H2_readers3=bool(pos_ok and shares["readers3"]["share"] >= 0.50),
            H2_random=bool(pos_ok and rand_mean < 0.15))
out = dict(split=SPLIT, smoke=SMOKE, n=len(rows), width_effect=float(E.mean()), width_effect_ci95=eff_ci,
           shares=shares, random_triples=[[f"L{l}H{h}" for l, h in t] for t in RANDOM], random_mean_share=rand_mean,
           verdicts=verd, H2_pass=bool(all(verd.values())), rows=rows)
save_json(out, f"c2_02_width_{SPLIT}{'_smoke' if SMOKE else ''}.json")
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
