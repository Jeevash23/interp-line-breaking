"""exp13: which components cancel the dotted-word illusion? Exploratory.

exp10/exp11: re-spelling two earlier words of the line as alternating letters and dots
moves L5H6 and L5H7 (patching both heads' outputs from the dotted run into the base run
gives -3.0 in log P(newline)), yet the behaviour barely moves (+0.05). Something else
must push the other way.

For each item (width 60, 3..14 characters left; same construction as exp10, new
paragraph seed), patch the output of ONE component at the decision token (every head
of layers 0..11 via its slice of the attention-output input, and every MLP of layers
0..11) from the dotted run into the base run ("noising"), and from the base run into the
dotted run ("denoising"). Change in log P(newline) relative to the unpatched run.

Components with large positive noising effects are the ones that over-count under the
dots and offset L5H6 and L5H7.
"""
import json, random, numpy as np, torch
from common import *

W = 60
LMAX = 12
tok, model = load("pythia-410m")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
COMPS = [("h", l, h) for l in range(LMAX) for h in range(Hn)] + [("m", l, None) for l in range(LMAX)]

cap = {}
for l in range(LMAX):
    layers[l].attention.dense.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(("z", l), a[0][:, -1].detach().clone()))
    layers[l].mlp.register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("m", l), o[:, -1].detach().clone()))

PATCH = {}  # ("z", l) -> list of (row, slice or None, vec)
def zhook(l):
    def f(m, a):
        lst = PATCH.get(("z", l))
        if not lst: return None
        z = a[0].clone()
        for row, h, vec in lst: z[row, -1, h * dh:(h + 1) * dh] = vec
        return (z,)
    return f
def mhook(l):
    def f(m, a, o):
        lst = PATCH.get(("m", l))
        if not lst: return None
        o = o.clone()
        for row, _, vec in lst: o[row, -1] = vec
        return o
    return f
for l in range(LMAX):
    layers[l].attention.dense.register_forward_pre_hook(zhook(l))
    layers[l].mlp.register_forward_hook(mhook(l))

def lp_of(ids, B=1):
    with torch.no_grad():
        hid = model.gpt_neox(torch.tensor([ids] * B, device="cuda")).last_hidden_state[:, -1]
        return torch.log_softmax(model.lm_head(hid).float(), -1)[:, nl].cpu().numpy()

def clean(ids):
    PATCH.clear(); lp = lp_of(ids)[0]
    return lp, {k: v[0].clone() for k, v in cap.items()}

def patched(ids, src, chunk=56):
    """Run ids once per component, patching that component at the last position from src."""
    out = np.zeros(len(COMPS))
    for s in range(0, len(COMPS), chunk):
        part = COMPS[s:s + chunk]
        PATCH.clear()
        for row, (kind, l, h) in enumerate(part):
            if kind == "h":
                PATCH.setdefault(("z", l), []).append((row, h, src[("z", l)][h * dh:(h + 1) * dh]))
            else:
                PATCH.setdefault(("m", l), []).append((row, None, src[("m", l)]))
        out[s:s + len(part)] = lp_of(ids, len(part))
    PATCH.clear()
    return out

def dotted(w): return "".join(ch if k % 2 == 0 else "." for k, ch in enumerate(w))

rng = random.Random(13)
rows = []
for p in train_paragraphs(seed=13):
    if len(rows) >= 120: break
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
    ws = list(words[: j + 1])
    for i in pick: ws[i] = dotted(ws[i])
    ib = tok(before + " ".join(words[: j + 1]), add_special_tokens=False)["input_ids"]
    isp = tok(before + " ".join(ws), add_special_tokens=False)["input_ids"]
    lb, cb = clean(ib); ls, cs = clean(isp)
    noise = patched(ib, cs) - lb
    denoise = patched(isp, cb) - ls
    rows.append(dict(total=float(ls - lb), noise=noise.tolist(), denoise=denoise.tolist()))
    if len(rows) % 20 == 0: print(len(rows), "items", flush=True)

N = np.array([r["noise"] for r in rows]); Dn = np.array([r["denoise"] for r in rows])
names = [f"L{l}H{h}" if kind == "h" else f"MLP{l}" for kind, l, h in COMPS]
order = np.argsort(-np.abs(N.mean(0)))
top = [dict(comp=names[i], noise=float(N[:, i].mean()), noise_se=float(N[:, i].std() / np.sqrt(len(N))),
            denoise=float(Dn[:, i].mean())) for i in order[:30]]
summ = dict(n=len(rows), total=float(np.mean([r["total"] for r in rows])),
            sum_noise_heads=float(N[:, :LMAX * Hn].mean(0).sum()), sum_noise_mlps=float(N[:, LMAX * Hn:].mean(0).sum()),
            top=top)
save_json(dict(summary=summ, names=names, noise_mean=N.mean(0).tolist(), denoise_mean=Dn.mean(0).tolist(), rows=rows),
          "exp13_illusion_decomposition.json")
print(json.dumps(summ, indent=1))
