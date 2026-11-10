"""exp10: a prediction from the dilution reading, with mediation. Exploratory.

Dilution reading (exp08, exp09): L5H6 and L5H7 count characters through the share of
attention left on the line-start newline; line tokens draw attention roughly in
proportion to their length, but single-character tokens draw more than their length.
Prediction nobody would make from a pure character counter: re-spelling earlier words
of the line with the same number of characters but many one-character tokens
(e.g. 'region' -> 'r-g-on' style is not same text; we use dotted initials and hyphen
splits that keep the character count exactly) makes the model over-count and break
earlier; the size should follow the drop in the two heads' newline shares.

Conditions at a decision point (width 60, 3..14 characters left), editing two earlier
words of the current line, keeping their character count:
  base   : original
  split  : each edited word w of length n >= 5 becomes w[0] + '.' + w[2] + '.' + ... alternating
           letters and dots (same length, many 1-character tokens)
  upper  : the same words fully upper-cased (same length, a few odd tokens; control)
Measures: change in log P(newline); change in L5H6 and L5H7 newline shares; a
prediction of the log P change from the share changes via exp09's dose-response
(piecewise-linear interpolation of mean dlogp against mean share, per head, summed).
"""
import json, random, numpy as np, torch
from common import *
import transformers.models.gpt_neox.modeling_gpt_neox as gm

W = 60
tok, model = load("pythia-410m", attn="eager")
nl = newline_id(tok)
SH = {}
_orig = gm.eager_attention_forward
def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    out, w = _orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
    if module.layer_idx == 5:
        SH["w"] = w[0, :, -1].float().cpu().numpy()  # (H, T)
    return out, w
gm.eager_attention_forward = patched

e9 = json.loads((OUT / "exp09_attention_edit.json").read_text())
def curve(head):
    d = e9[head]; xs = np.array([d["mean_share"][k] for k in d["mean_share"]]); ys = np.array([d["mean_dlogp"][k] for k in d["mean_dlogp"]])
    o = np.argsort(xs); return xs[o], ys[o]
C6, C7 = curve("L5H6"), curve("L5H7")
def pred_from_share(s6b, s6, s7b, s7):
    f6 = lambda s: np.interp(s, *C6); f7 = lambda s: np.interp(s, *C7)
    return (f6(s6) - f6(s6b)) + (f7(s7) - f7(s7b))

def run(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        hid = model.gpt_neox(torch.tensor([ids], device="cuda")).last_hidden_state[0, -1]
        lp = torch.log_softmax(model.lm_head(hid).float(), -1)[nl].item()
    pn = [i for i, t in enumerate(ids) if t == nl][-1]
    return lp, float(SH["w"][6, pn]), float(SH["w"][7, pn]), len(ids)

def dotted(w):
    return "".join(ch if k % 2 == 0 else "." for k, ch in enumerate(w))

rng = random.Random(2)
rows = []
for p in train_paragraphs(seed=10):
    if len(rows) >= 400: break
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
    base, split, upper = (before + " ".join(x) for x in (words[: j + 1], ws, wu))
    assert len(base) == len(split) == len(upper)
    b, s, u = run(base), run(split), run(upper)
    rows.append(dict(rem=W - len(" ".join(words[: j + 1])),
                     d_split=s[0] - b[0], d_upper=u[0] - b[0],
                     s6=(b[1], s[1], u[1]), s7=(b[2], s[2], u[2]), ntok=(b[3], s[3], u[3]),
                     pred_split=float(pred_from_share(b[1], s[1], b[2], s[2])), pred_upper=float(pred_from_share(b[1], u[1], b[2], u[2]))))

A = lambda k: np.array([r[k] for r in rows])
out = dict(n=len(rows),
           split=dict(dlogp=float(A("d_split").mean()), median=float(np.median(A("d_split"))), pred_from_shares=float(A("pred_split").mean()),
                      corr_pred_obs=float(np.corrcoef(A("pred_split"), A("d_split"))[0, 1]),
                      share6=(float(np.mean([r["s6"][0] for r in rows])), float(np.mean([r["s6"][1] for r in rows]))),
                      share7=(float(np.mean([r["s7"][0] for r in rows])), float(np.mean([r["s7"][1] for r in rows]))),
                      extra_tokens=float(np.mean([r["ntok"][1] - r["ntok"][0] for r in rows]))),
           upper=dict(dlogp=float(A("d_upper").mean()), median=float(np.median(A("d_upper"))), pred_from_shares=float(A("pred_upper").mean()),
                      corr_pred_obs=float(np.corrcoef(A("pred_upper"), A("d_upper"))[0, 1]),
                      share6=(float(np.mean([r["s6"][0] for r in rows])), float(np.mean([r["s6"][2] for r in rows]))),
                      share7=(float(np.mean([r["s7"][0] for r in rows])), float(np.mean([r["s7"][2] for r in rows]))),
                      extra_tokens=float(np.mean([r["ntok"][2] - r["ntok"][0] for r in rows]))),
           frac_split_raises=float((A("d_split") > 0).mean()))
save_json(dict(summary=out, rows=rows), "exp10_illusion_mediation.json")
print(json.dumps(out, indent=1))
