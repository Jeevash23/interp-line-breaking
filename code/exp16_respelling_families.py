"""exp16: a family of same-length re-spellings. Which ones fool the counting heads in the
same direction, and does the behaviour then follow? Exploratory.

exp13: under the dotted re-spelling, patching single components at the decision token
from the dotted run into the base run gives L5H7 -3.1, L5H6 +1.1, L4H14 +0.9 (log P(newline)),
while the behaviour moves only +0.12. The heads disagree.

Here two earlier words (>= 5 letters) of the current line are re-spelt, keeping the
character count, in one of several ways. For each family and item:
  behaviour : lp(respelt) - lp(base) at the decision point (width 60, 3..14 left)
  vote(head): lp(base with that head's decision-token output taken from the respelt run) - lp(base)
              for L2H7, L4H10, L4H14, L4H15, L5H6, L5H7, and for all six together
  resid@L   : lp(base with the whole residual after layer L at the decision token taken from
              the respelt run) - lp(base), L in {3,4,5,6,8,12,16,23}
Question: across families, does the behaviour follow the heads when they agree?
"""
import json, random, string, numpy as np, torch
from common import *

W = 60
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
HEADS = [(2, 7), (4, 10), (4, 14), (4, 15), (5, 6), (5, 7)]
RL = [3, 4, 5, 6, 8, 12, 16, 23]

zc, rc = {}, {}
for l in {l for l, _ in HEADS}:
    layers[l].attention.dense.register_forward_pre_hook(lambda m, a, l=l: zc.__setitem__(l, a[0][:, -1].detach().clone()))
for l in RL:
    layers[l].register_forward_hook(lambda m, a, o, l=l: rc.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[:, -1].detach().clone()))
PZ, PR = {}, {}
def zhook(l):
    def f(m, a):
        lst = PZ.get(l)
        if not lst: return None
        z = a[0].clone()
        for row, h, vec in lst: z[row, -1, h * dh:(h + 1) * dh] = vec
        return (z,)
    return f
def rhook(l):
    def f(m, a, o):
        lst = PR.get(l)
        if not lst: return None
        x = (o[0] if isinstance(o, tuple) else o).clone()
        for row, vec in lst: x[row, -1] = vec
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
    return f
for l in {l for l, _ in HEADS}: layers[l].attention.dense.register_forward_pre_hook(zhook(l))
for l in RL: layers[l].register_forward_hook(rhook(l))

def lp_of(ids, B=1):
    with torch.no_grad():
        hid = model.gpt_neox(torch.tensor([ids] * B, device="cuda")).last_hidden_state[:, -1]
        return torch.log_softmax(model.lm_head(hid).float(), -1)[:, nl].cpu().numpy()

rng_letters = random.Random(99)
FAM = {
    "dotted": lambda w: "".join(c if k % 2 == 0 else "." for k, c in enumerate(w)),
    "dashed": lambda w: "".join(c if k % 2 == 0 else "-" for k, c in enumerate(w)),
    "commas": lambda w: "".join(c if k % 2 == 0 else "," for k, c in enumerate(w)),
    "upper": lambda w: w.upper(),
    "camel": lambda w: "".join(c.upper() if k % 2 == 0 else c.lower() for k, c in enumerate(w)),
    "digits": lambda w: "".join(rng_letters.choice("123456789") for _ in w),
    "scramble": lambda w: "".join(rng_letters.choice(string.ascii_lowercase) for _ in w),
    "repeat": lambda w: w[0].lower() * len(w),
}

rng = random.Random(16)
rows = []
for p in train_paragraphs(seed=16):
    if len(rows) >= 80: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1))
    words = lines[k].split(" ")
    js = [j for j in range(3, len(words)) if 3 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    j = js[0]
    cand = [i for i in range(j) if len(words[i]) >= 5 and words[i].isalpha()]
    if len(cand) < 2: continue
    pick = rng.sample(cand, 2)
    before = "\n".join(lines[:k]) + "\n"
    ib = tok(before + " ".join(words[: j + 1]), add_special_tokens=False)["input_ids"]
    PZ.clear(); PR.clear()
    lb = lp_of(ib)[0]
    row = dict(rem=W - len(" ".join(words[: j + 1])))
    for fam, fn in FAM.items():
        ws = list(words[: j + 1])
        for i in pick: ws[i] = fn(ws[i])
        txt = before + " ".join(ws)
        assert len(txt) == len(before + " ".join(words[: j + 1]))
        ir = tok(txt, add_special_tokens=False)["input_ids"]
        PZ.clear(); PR.clear()
        lr = lp_of(ir)[0]
        zr = {l: v[0].clone() for l, v in zc.items()}; rr = {l: v[0].clone() for l, v in rc.items()}
        # batch: one row per head, one row for all heads, one row per resid layer
        PZ.clear(); PR.clear()
        B = len(HEADS) + 1 + len(RL)
        for row_i, (l, h) in enumerate(HEADS):
            PZ.setdefault(l, []).append((row_i, h, zr[l][h * dh:(h + 1) * dh]))
            PZ.setdefault(l, []).append((len(HEADS), h, zr[l][h * dh:(h + 1) * dh]))
        for q, l in enumerate(RL): PR.setdefault(l, []).append((len(HEADS) + 1 + q, rr[l]))
        out = lp_of(ib, B) - lb
        PZ.clear(); PR.clear()
        row[fam] = dict(behaviour=float(lr - lb), ntok_extra=len(ir) - len(ib),
                        votes={f"L{l}H{h}": float(out[q]) for q, (l, h) in enumerate(HEADS)},
                        all_heads=float(out[len(HEADS)]),
                        resid={str(l): float(out[len(HEADS) + 1 + q]) for q, l in enumerate(RL)})
    rows.append(row)
    if len(rows) % 20 == 0: print(len(rows), "items", flush=True)

summ = {}
for fam in FAM:
    R = [r[fam] for r in rows]
    beh = np.array([x["behaviour"] for x in R])
    summ[fam] = dict(behaviour=float(beh.mean()), behaviour_se=float(beh.std() / np.sqrt(len(beh))),
                     ntok_extra=float(np.mean([x["ntok_extra"] for x in R])),
                     votes={k: float(np.mean([x["votes"][k] for x in R])) for k in R[0]["votes"]},
                     all_heads=float(np.mean([x["all_heads"] for x in R])),
                     resid={k: float(np.mean([x["resid"][k] for x in R])) for k in R[0]["resid"]})
save_json(dict(summary=summ, rows=rows), "exp16_respelling_families.json")
print(json.dumps(summ, indent=1))
