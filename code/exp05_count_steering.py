"""exp05: is the decodable character count used? Exploratory.

At one layer, at a word-end decision point, shift the residual stream from the mean
state for "u characters used" to the mean state for "u + delta characters used"
(class means from a separate set of paragraphs, same width). If the model reads its
character count from this layer, the newline probability should move as if delta
more characters were used: towards the clean hazard at (remaining - delta).

Controls: the same-norm shift in a random direction; and shifting by the
difference between two class means of tokens-used-on-line at matched characters
(not done here; tokens were already shown to add nothing behaviourally).
"""
import sys, json, numpy as np, torch
from common import *

MNAME = "pythia-410m"
W = int(sys.argv[1]) if len(sys.argv) > 1 else 60
LAYERS = [2, 3, 4, 5, 6, 8, 10]
DELTAS = [-10, -6, -3, 3, 6, 10]
tok, model = load(MNAME)
nl = newline_id(tok)
layers = model.gpt_neox.layers

def items_for(seed, n, maxlen=700):
    out = []
    for p in train_paragraphs(seed=seed):
        if len(out) >= n: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 6: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > maxlen: continue
        out.append((ids, recs))
    return out

fit_items = items_for(seed=2, n=60)
test_items = items_for(seed=3, n=25)

cap = {}
def grab(l):
    return layers[l].register_forward_hook(lambda m, a, o: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach()))

# class means of residual by characters used, per layer
hs = [grab(l) for l in LAYERS]
sums = {l: {} for l in LAYERS}; cnts = {}
clean_hazard = {}
with torch.no_grad():
    for ids, recs in fit_items:
        out = model(torch.tensor([ids], device="cuda"))
        lp = torch.log_softmax(out.logits[0].float(), -1)
        for r in recs:
            u = r["used"]; cnts[u] = cnts.get(u, 0) + 1
            clean_hazard.setdefault(r["remaining"], []).append(float(lp[r["i"], nl].exp()))
            for l in LAYERS:
                v = cap[l][0, r["i"]].float()
                sums[l][u] = sums[l].get(u, 0) + v
for h_ in hs: h_.remove()
means = {l: {u: (s / cnts[u]) for u, s in sums[l].items() if cnts[u] >= 8} for l in LAYERS}
hazard = {r: float(np.mean(v)) for r, v in clean_hazard.items() if len(v) >= 8}

def run_batch(ids, l, vecs):
    """ids truncated at the decision point; one batch row per added vector (added at the last position)."""
    V = torch.stack(vecs, 0)  # (B, D)
    def f(m, a, o):
        hs_ = o[0] if isinstance(o, tuple) else o
        hs_ = hs_.clone(); hs_[:, -1] += V
        return (hs_,) + tuple(o[1:]) if isinstance(o, tuple) else hs_
    h = layers[l].register_forward_hook(f)
    with torch.no_grad():
        x = torch.tensor([ids] * len(vecs), device="cuda")
        hid = model.gpt_neox(x).last_hidden_state[:, -1]
        lp = torch.log_softmax(model.lm_head(hid).float(), -1)[:, nl]
    h.remove()
    return lp.cpu().numpy()

rng = torch.Generator(device="cuda").manual_seed(0)
results = {}
for l in LAYERS:
    rows = []
    for ids, recs in test_items:
        for r in recs:
            u, rem = r["used"], r["remaining"]
            if not (4 <= rem <= 16) or u not in means[l]: continue
            ds = [d for d in DELTAS if (u + d) in means[l] and (rem - d) in hazard]
            if not ds: continue
            vecs = [torch.zeros_like(means[l][u])]
            for d in ds:
                vec = means[l][u + d] - means[l][u]
                rnd = torch.randn(vec.shape, generator=rng, device="cuda"); rnd = rnd / rnd.norm() * vec.norm()
                vecs += [vec, rnd]
            out = run_batch(ids[: r["i"] + 1], l, vecs)
            for k, d in enumerate(ds):
                rows.append((rem, d, out[0], out[1 + 2 * k], out[2 + 2 * k], np.log(hazard[rem]), np.log(hazard[rem - d])))
    a = np.array(rows)
    if len(a) == 0: continue
    res = {}
    for d in DELTAS:
        m = a[:, 1] == d
        if m.sum() == 0: continue
        pred = a[m, 6] - a[m, 5]; got = a[m, 3] - a[m, 2]; rnd = a[m, 4] - a[m, 2]
        res[d] = dict(n=int(m.sum()), pred=float(pred.mean()), steer=float(got.mean()), random=float(rnd.mean()),
                      corr_pred_steer=float(np.corrcoef(pred, got)[0, 1]) if m.sum() > 3 else None)
    results[l] = res
    print("layer", l, json.dumps({d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()} for d, x in res.items()}), flush=True)
save_json(dict(width=W, layers=results, hazard=hazard), f"exp05_count_steering_W{W}.json")
