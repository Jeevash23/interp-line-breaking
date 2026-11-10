"""c3_04_paths (CONFIRMATORY, PROTOCOL-3.md): round 3 on new text (Wikipedia articles created 2025-26,
part conf-pythia410m). A copy of the frozen code/c2_04_paths.py; the only changes are this header, the
text loader (newtext.load_paras in place of split_paragraphs), the split name allowed and the
output file name.
Usage: c3_04_paths.py conf-pythia410m   (the frozen run);  c3_04_paths.py train --smoke   (pipeline check only)

Original docstring:
c2_04 (CONFIRMATORY, PROTOCOL-2.md H5): two ways the reader heads take up the width.

Usage: c2_04_paths.py test              (the frozen run)
       c2_04_paths.py train --smoke     (pipeline check only; 8 items)
Exploration (exp17, train): with the width patch (the base run's line-start newline residual after
layer 5 replaced by its state in the short-previous-line run), applying the patch only to one
head's KEY at the newline position, or only to its VALUE, gave L8H15 key +0.04, value +0.29;
L17H1 key +0.12, value +0.02.
Items: exactly as code/c2_02_width.py (width 60, one per paragraph, seed-100 paragraph order,
random choices seed 200, previous line >= 10 characters shorter, current line tokenised alike),
up to 150 items. Attention is computed eagerly here (needed to patch keys and values), so the
width effect is re-measured in this script.
For each head (L8H15, L17H1; control L8H11), in the base run: replace only that head's key (or
only its value) at the newline position with the one it has in the width-patched run; change
in log P(newline) at the decision point.
H5a: L8H15 value-only minus key-only > 0 (95% item-bootstrap interval above 0).
H5b: L17H1 key-only minus value-only > 0 (95% item-bootstrap interval above 0).
Positive control: width effect interval above 0. 2,000 item bootstrap resamples, seed 0.
"""
import sys, json, random, numpy as np, torch
import transformers.models.gpt_neox.modeling_gpt_neox as gm
from common import *
from newtext import load_paras

SPLIT = sys.argv[1]; SMOKE = "--smoke" in sys.argv
assert SPLIT in ("train", "conf-pythia410m")
NITEMS = 8 if SMOKE else 150
NBOOT = 50 if SMOKE else 2000
W = 60
tok, model = load("pythia-410m", attn="eager")
nl = newline_id(tok)
layers = model.gpt_neox.layers
HEADS = [(8, 15), (17, 1), (8, 11)]
HL = {l for l, _ in HEADS}
ST = dict(capture=False, kv={}, kpatch=None, vpatch=None)
_orig = gm.eager_attention_forward
def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    l = module.layer_idx
    if l not in HL:
        return _orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
    if ST["capture"]:
        ST["kv"][l] = (key.detach().clone(), value.detach().clone())
    if ST["kpatch"] and ST["kpatch"][0] == l:
        _, h, pos, vec = ST["kpatch"]; key = key.clone(); key[:, h, pos] = vec
    if ST["vpatch"] and ST["vpatch"][0] == l:
        _, h, pos, vec = ST["vpatch"]; value = value.clone(); value[:, h, pos] = vec
    s = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None: s = s + attention_mask
    w = torch.softmax(s, -1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(w, value).transpose(1, 2).contiguous()
    return out, w
gm.eager_attention_forward = patched
res5 = {}
layers[5].register_forward_hook(lambda m, a, o: res5.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().clone()))
EDIT = {}
def ed(m, a, o):
    if not EDIT: return None
    x = (o[0] if isinstance(o, tuple) else o).clone(); x[:, EDIT["pos"]] = EDIT["vec"]
    return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
layers[5].register_forward_hook(ed)
def lp_of(ids):
    with torch.no_grad():
        hid = model.gpt_neox(torch.tensor([ids], device="cuda")).last_hidden_state[0, -1]
        return torch.log_softmax(model.lm_head(hid).float(), -1)[nl].item()
def reset(): ST.update(capture=False, kpatch=None, vpatch=None); EDIT.clear()

rng = random.Random(200)
rows = []
for p in load_paras(SPLIT, seed=100):
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
    reset(); lb = lp_of(ib)
    reset(); lp_of(is_); vnl = res5["x"][0, ns].clone()
    reset(); EDIT.update(pos=nb, vec=vnl); ST["capture"] = True; lw = lp_of(ib); kv_w = dict(ST["kv"])
    row = dict(eff=lw - lb)
    for l, h in HEADS:
        reset(); ST["kpatch"] = (l, h, nb, kv_w[l][0][0, h, nb]); lk = lp_of(ib)
        reset(); ST["vpatch"] = (l, h, nb, kv_w[l][1][0, h, nb]); lv = lp_of(ib)
        row[f"L{l}H{h}"] = dict(K=lk - lb, V=lv - lb)
    reset()
    rows.append(row)
    if len(rows) % 25 == 0: print(len(rows), "items", flush=True)

n = len(rows); br = np.random.default_rng(0); idx = [br.integers(0, n, n) for _ in range(NBOOT)]
def stat(a):
    a = np.asarray(a); return dict(mean=float(a.mean()), ci95=[float(np.percentile([a[i].mean() for i in idx], q)) for q in (2.5, 97.5)])
E = [r["eff"] for r in rows]
summ = dict(width_effect=stat(E))
for l, h in HEADS:
    nm = f"L{l}H{h}"
    summ[nm] = dict(K=stat([r[nm]["K"] for r in rows]), V=stat([r[nm]["V"] for r in rows]))
summ["L8H15_V_minus_K"] = stat([r["L8H15"]["V"] - r["L8H15"]["K"] for r in rows])
summ["L17H1_K_minus_V"] = stat([r["L17H1"]["K"] - r["L17H1"]["V"] for r in rows])
pos_ok = summ["width_effect"]["ci95"][0] > 0
verd = dict(positive_control=bool(pos_ok), H5a=bool(pos_ok and summ["L8H15_V_minus_K"]["ci95"][0] > 0),
            H5b=bool(pos_ok and summ["L17H1_K_minus_V"]["ci95"][0] > 0))
out = dict(split=SPLIT, smoke=SMOKE, n=n, summary=summ, verdicts=verd, H5_pass=bool(all(verd.values())), rows=rows)
save_json(out, f"c3_04_paths_{SPLIT}{'_smoke' if SMOKE else ''}.json")
print(json.dumps(dict(n=n, summary=summ, verdicts=verd), indent=1))
