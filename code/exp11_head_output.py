"""exp11: why does the dotted-word manipulation lower the heads' newline share but not
change behaviour (exp10)? Exploratory.

(a) Train ridge probes for 'characters used' on the outputs of L5H6 and L5H7 (and their
    sum) at natural decision points (width 60).
(b) For the exp10-style pairs (base vs split, base vs upper), read the probe on each
    head's output: does the decoded count rise under 'split' (the head is fooled) or not?
(c) Patch the two heads' outputs at the decision point from the manipulated run into the
    base run: change in log P(newline). And the converse: patch the base outputs of the
    two heads into the manipulated run.
"""
import json, random, numpy as np, torch
from common import *

W = 60
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
HEADS = [6, 7]
Wd = layers[5].attention.dense.weight.detach().float()
cap = {}
layers[5].attention.dense.register_forward_pre_hook(lambda m, a: cap.__setitem__("z", a[0].detach().clone()))

def head_out(z, h): return z[..., h * dh:(h + 1) * dh].float() @ Wd[:, h * dh:(h + 1) * dh].T

# (a) probes on natural data
X = {h: [] for h in HEADS}; Y = []
for p in train_paragraphs(seed=11)[:200]:
    if len(Y) > 5000: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 5: continue
    ids, recs = word_end_positions(tok, lines, W)
    if len(ids) > 700: continue
    with torch.no_grad(): model.gpt_neox(torch.tensor([ids], device="cuda"))
    z = cap["z"][0]
    I = torch.tensor([r["i"] for r in recs], device="cuda")
    for h in HEADS: X[h].append(head_out(z[I], h).cpu().numpy())
    Y += [r["used"] for r in recs]
Y = np.array(Y, float)
probes = {}
for h in HEADS + ["sum"]:
    Xh = np.concatenate(X[h]) if h != "sum" else np.concatenate(X[6]) + np.concatenate(X[7])
    mu, sd = Xh.mean(0), Xh.std(0) + 1e-6; A = (Xh - mu) / sd
    w = np.linalg.solve(A.T @ A + 100 * np.eye(D), A.T @ (Y - Y.mean()))
    pred = A @ w + Y.mean(); r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    probes[h] = (mu, sd, w, Y.mean(), float(r2))
def decode(vec, h):
    mu, sd, w, b, _ = probes[h]; return float(((vec - mu) / sd) @ w + b)

def run(text, patch=None):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    hk = None
    if patch is not None:
        def f(m, a):
            z = a[0].clone()
            for h in HEADS: z[0, -1, h * dh:(h + 1) * dh] = patch[0, -1, h * dh:(h + 1) * dh]
            return (z,)
        hk = layers[5].attention.dense.register_forward_pre_hook(f)
    with torch.no_grad():
        hid = model.gpt_neox(torch.tensor([ids], device="cuda")).last_hidden_state[0, -1]
        lp = torch.log_softmax(model.lm_head(hid).float(), -1)[nl].item()
    if hk: hk.remove()
    z = cap["z"].clone()
    return lp, z

def dotted(w): return "".join(ch if k % 2 == 0 else "." for k, ch in enumerate(w))
rng = random.Random(2)
rows = []
for p in train_paragraphs(seed=10):
    if len(rows) >= 300: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, len(lines) - 1)
    words = lines[k].split(" ")
    js = [j for j in range(3, len(words)) if 3 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    j = js[0]
    cand = [i for i in range(j) if len(words[i]) >= 5 and words[i].isalpha()]
    if len(cand) < 2: continue
    pick = rng.sample(cand, 2)
    before = "\n".join(lines[:k]) + "\n"
    ws = list(words[: j + 1]); wu = list(words[: j + 1])
    for i in pick: ws[i] = dotted(ws[i]); wu[i] = wu[i].upper()
    tb, tsp, tu = (before + " ".join(x) for x in (words[: j + 1], ws, wu))
    lb, zb = run(tb); ls, zs = run(tsp); lu, zu = run(tu)
    row = dict(used=len(" ".join(words[: j + 1])), d_split=ls - lb, d_upper=lu - lb)
    for nm, z in (("b", zb), ("s", zs), ("u", zu)):
        o6 = head_out(z[0, -1], 6).cpu().numpy(); o7 = head_out(z[0, -1], 7).cpu().numpy()
        row[f"dec6_{nm}"] = decode(o6, 6); row[f"dec7_{nm}"] = decode(o7, 7); row[f"decsum_{nm}"] = decode(o6 + o7, "sum")
    row["patch_split_into_base"] = run(tb, patch=zs)[0] - lb
    row["patch_base_into_split"] = run(tsp, patch=zb)[0] - ls
    row["patch_upper_into_base"] = run(tb, patch=zu)[0] - lb
    rows.append(row)
A = lambda k: float(np.mean([r[k] for r in rows]))
out = dict(n=len(rows), probe_r2={str(h): probes[h][4] for h in probes},
           decoded_used={nm: dict(L5H6=A(f"dec6_{nm}"), L5H7=A(f"dec7_{nm}"), both=A(f"decsum_{nm}")) for nm in "bsu"},
           true_used=A("used"), d_split=A("d_split"), d_upper=A("d_upper"),
           patch_split_into_base=A("patch_split_into_base"), patch_base_into_split=A("patch_base_into_split"),
           patch_upper_into_base=A("patch_upper_into_base"))
save_json(dict(summary=out, rows=rows), "exp11_head_output.json")
print(json.dumps(out, indent=1))
