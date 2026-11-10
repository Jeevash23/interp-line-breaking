"""exp07: does the previous newline token carry the previous line's length, and is it
read by L5H6? Exploratory.

Pairs: base prompt (lines wrapped at W, ending at a decision point on line k, 3..14
characters left) and 'short' prompt (same, but line k-1 lost its last words, >= 10
characters shorter). In exp06 the short version raised log P(newline) by about 1.2.

Measures, per pair:
  total     : lp(short) - lp(base)
  patch@l   : run base, but replace the residual stream at the previous-newline token
              (output of layer l) with the short run's residual at its own previous-newline
              token. lp(patched) - lp(base).  l in {2,3,4,5,6,8}
  patch@4 with L5H6 ablated (mean over positions) in both runs: does the patch still work?
  patch@4 with a random head of layer 5 ablated instead (control)
  attention of L5H6 (and L5H7) from the decision point to the previous newline, base vs short.
"""
import sys, json, random, numpy as np, torch
from common import *

W = 60
MNAME = "pythia-410m"
tok, model = load(MNAME, attn="eager")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn

def ids_of(text): return tok(text, add_special_tokens=False)["input_ids"]

# mean z of layer 5 (for head ablation), from a few paragraphs
zs = []
cap = {}
h = layers[5].attention.dense.register_forward_pre_hook(lambda m, a: cap.__setitem__("z", a[0].detach()))
with torch.no_grad():
    for p in train_paragraphs(seed=6)[:20]:
        model.gpt_neox(torch.tensor([ids_of(p)[:400]], device="cuda")); zs.append(cap["z"][0].mean(0))
h.remove()
zmean5 = torch.stack(zs).mean(0)

def forward(ids, patch=None, ablate=None, want_att=False):
    """patch=(layer, pos, vec) replaces residual output of layer at pos; ablate=head index in layer 5."""
    hs = []
    if patch is not None:
        l, pos, vec = patch
        def f(m, a, o):
            x = o[0] if isinstance(o, tuple) else o
            x = x.clone(); x[0, pos] = vec
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        hs.append(layers[l].register_forward_hook(f))
    if ablate is not None:
        def g(m, a, hh=ablate):
            z = a[0].clone(); z[..., hh * dh:(hh + 1) * dh] = zmean5[hh * dh:(hh + 1) * dh]; return (z,)
        hs.append(layers[5].attention.dense.register_forward_pre_hook(g))
    res = {}
    def keep(l):
        return layers[l].register_forward_hook(lambda m, a, o: res.__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach()))
    hs += [keep(l) for l in [2, 3, 4, 5, 6, 8]]
    with torch.no_grad():
        out = model.gpt_neox(torch.tensor([ids], device="cuda"), output_attentions=want_att)
        lp = torch.log_softmax(model.lm_head(out.last_hidden_state[0, -1]).float(), -1)[nl].item()
    for x in hs: x.remove()
    att = None
    if want_att:
        att = out.attentions[5][0, :, -1].float().cpu().numpy()  # (H, T)
    return lp, res, att

rows = []
rng = random.Random(1)
for p in train_paragraphs(seed=7):
    if len(rows) >= 300: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, len(lines) - 1)
    words = lines[k].split(" ")
    js = [j for j in range(2, len(words)) if 3 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    j = js[0]
    prefix = " ".join(words[: j + 1])
    before = lines[:k]
    pw = before[-1].split(" "); short = before[-1]
    while len(pw) > 2 and len(before[-1]) - len(short) < 10:
        pw = pw[:-1]; short = " ".join(pw)
    tb = "\n".join(before) + "\n" + prefix
    ts = "\n".join(before[:-1] + [short]) + "\n" + prefix
    ib, is_ = ids_of(tb), ids_of(ts)
    nb = [i for i, t in enumerate(ib) if t == nl][-1]; ns = [i for i, t in enumerate(is_) if t == nl][-1]
    lpb, rb, ab = forward(ib, want_att=True)
    lps, rs, as_ = forward(is_, want_att=True)
    row = dict(rem=W - len(prefix), dshort=len(before[-1]) - len(short), total=lps - lpb,
               att56_base=float(ab[6, nb]), att56_short=float(as_[6, ns]), att57_base=float(ab[7, nb]), att57_short=float(as_[7, ns]))
    for l in [2, 3, 4, 5, 6, 8]:
        row[f"patch{l}"] = forward(ib, patch=(l, nb, rs[l][0, ns]))[0] - lpb
    # with L5H6 ablated in both runs
    lpb_a, _, _ = forward(ib, ablate=6); lps_a, rs_a, _ = forward(is_, ablate=6)
    row["total_L5H6abl"] = lps_a - lpb_a
    row["patch4_L5H6abl"] = forward(ib, patch=(4, nb, rs_a[4][0, ns]), ablate=6)[0] - lpb_a
    ctl = rng.choice([h_ for h_ in range(Hn) if h_ not in (6, 7)])
    lpb_c, _, _ = forward(ib, ablate=ctl); lps_c, rs_c, _ = forward(is_, ablate=ctl)
    row["total_ctlabl"] = lps_c - lpb_c
    row["patch4_ctlabl"] = forward(ib, patch=(4, nb, rs_c[4][0, ns]), ablate=ctl)[0] - lpb_c
    row["ctl_head"] = ctl
    rows.append(row)

keys = [k for k in rows[0] if k not in ("ctl_head",)]
summ = {k: float(np.mean([r[k] for r in rows])) for k in keys}
summ["n"] = len(rows)
save_json(dict(summary=summ, rows=rows), "exp07_newline_key.json")
print(json.dumps({k: round(v, 3) for k, v in summ.items()}, indent=1))
