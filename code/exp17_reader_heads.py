"""exp17: how do the reader heads L6H2, L8H15 and L17H1 combine the count and the width?
Exploratory.

exp14: freezing L8H15, L17H1 or L6H2 at the decision token removes 47%, 27% and 23% of the
width effect (previous line shortened; newline state patched after layer 5) and 13%, 12%
and 9% of the count effect (class-mean shift d = +6 at the decision token after layer 5).

For each reader head (control: L8H11) and the same two sources:
 (a) attention share from the decision point to the line-start newline, natural data, and
     its correlation with characters used and characters remaining;
 (b) WIDTH, split by path: in the base run, replace only this head's KEY at the newline
     position with the key it has in the width-patched run (QK path), or only its VALUE
     (OV path). Change in log P(newline); also the head's newline share under each.
 (c) COUNT: with the count shift applied, the change in this head's newline share
     (a query-side comparison would change it), and the effect of freezing only this head's
     attention PATTERN at the decision token to the unshifted pattern (values free).
"""
import json, random, numpy as np, torch
import transformers.models.gpt_neox.modeling_gpt_neox as gm
from common import *

W = 60
tok, model = load("pythia-410m", attn="eager")
nl = newline_id(tok)
layers = model.gpt_neox.layers
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
HEADS = [(6, 2), (8, 15), (17, 1), (8, 11)]
HL = {l for l, _ in HEADS}

ST = dict(capture=False, kv={}, att={}, kpatch=None, vpatch=None, wfix=None)
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
    if ST["wfix"] and ST["wfix"][0] == l:
        _, h, row = ST["wfix"]; w = w.clone(); w[:, h, -1] = row
    ST["att"][l] = w[:, :, -1].detach().clone()
    out = torch.matmul(w, value).transpose(1, 2).contiguous()
    return out, w
gm.eager_attention_forward = patched

res5 = {}
layers[5].register_forward_hook(lambda m, a, o: res5.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().clone()))
EDIT = {}
def edit5(m, a, o):
    if not EDIT: return None
    x = (o[0] if isinstance(o, tuple) else o).clone()
    for pos, vec, mode in EDIT["list"]:
        if mode == "add": x[:, pos] += vec
        else: x[:, pos] = vec
    return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
layers[5].register_forward_hook(edit5)

def lp_of(ids):
    with torch.no_grad():
        hid = model.gpt_neox(torch.tensor([ids], device="cuda")).last_hidden_state[:, -1]
        return torch.log_softmax(model.lm_head(hid).float(), -1)[0, nl].item()

def reset(): ST.update(capture=False, kpatch=None, vpatch=None, wfix=None); EDIT.clear()

# class means after layer 5 by characters used (as exp14)
sums, cnts = {}, {}
nat = []
for p in train_paragraphs(seed=2):
    if sum(cnts.values()) > 9000: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    ids, recs = word_end_positions(tok, lines, W)
    if len(ids) > 700: continue
    reset(); lp_of(ids)
    for r in recs:
        u = r["used"]; v = res5["x"][0, r["i"]]
        sums[u] = sums.get(u, 0) + v; cnts[u] = cnts.get(u, 0) + 1
means = {u: s / cnts[u] for u, s in sums.items() if cnts[u] >= 8}

# (a) natural attention profile, separate paragraphs (full-sequence attention needed: use output_attentions)
A_rows = []
for p in train_paragraphs(seed=17)[:60]:
    if len(A_rows) > 3000: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    ids, recs = word_end_positions(tok, lines, W)
    if len(ids) > 700: continue
    reset()
    with torch.no_grad(): out = model.gpt_neox(torch.tensor([ids], device="cuda"), output_attentions=True)
    nls = [i for i, t in enumerate(ids) if t == nl]
    for r in recs:
        i = r["i"]; pn = [x for x in nls if x < i][-1]
        A_rows.append([r["used"], r["remaining"]] + [float(out.attentions[l][0, h, i, pn]) for l, h in HEADS])
A = np.array(A_rows)
prof = {f"L{l}H{h}": dict(mean_share=float(A[:, 2 + q].mean()), corr_used=float(np.corrcoef(A[:, 2 + q], A[:, 0])[0, 1]),
                          corr_remaining=float(np.corrcoef(A[:, 2 + q], A[:, 1])[0, 1])) for q, (l, h) in enumerate(HEADS)}
print("profile", json.dumps(prof), flush=True)

rng = random.Random(17)
rows = []
for p in train_paragraphs(seed=18):
    if len(rows) >= 120: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1))
    words = lines[k].split(" ")
    js = [j for j in range(2, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    j = js[0]
    prefix = " ".join(words[: j + 1]); u = len(prefix)
    if u not in means or (u + 6) not in means: continue
    before = lines[:k]
    pw = before[-1].split(" "); short = before[-1]
    while len(pw) > 2 and len(before[-1]) - len(short) < 10:
        pw = pw[:-1]; short = " ".join(pw)
    if len(before[-1]) - len(short) < 10: continue
    ib = tok("\n".join(before) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    is_ = tok("\n".join(before[:-1] + [short]) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    nb = [i for i, t in enumerate(ib) if t == nl][-1]; ns = [i for i, t in enumerate(is_) if t == nl][-1]
    if len(ib) - nb != len(is_) - ns: continue
    row = {}
    # base
    reset(); ST["capture"] = True; lb = lp_of(ib); kv_b = dict(ST["kv"]); att_b = {l: v.clone() for l, v in ST["att"].items()}
    # width-patched run (newline state after layer 5 from the short prompt)
    reset(); lp_of(is_); vnl = res5["x"][0, ns].clone()
    reset(); EDIT["list"] = [(nb, vnl, "set")]; ST["capture"] = True; lw = lp_of(ib); kv_w = dict(ST["kv"]); att_w = {l: v.clone() for l, v in ST["att"].items()}
    # count-shifted run
    reset(); EDIT["list"] = [(len(ib) - 1, means[u + 6] - means[u], "add")]; lc = lp_of(ib); att_c = {l: v.clone() for l, v in ST["att"].items()}
    row.update(eff_width=lw - lb, eff_count=lc - lb)
    for l, h in HEADS:
        nm = f"L{l}H{h}"
        # (b) width, key-only and value-only at the newline position
        reset(); ST["kpatch"] = (l, h, nb, kv_w[l][0][0, h, nb]); lk = lp_of(ib); sk = float(ST["att"][l][0, h, nb])
        reset(); ST["vpatch"] = (l, h, nb, kv_w[l][1][0, h, nb]); lv = lp_of(ib)
        # (c) count shift with this head's pattern fixed to the unshifted one
        reset(); EDIT["list"] = [(len(ib) - 1, means[u + 6] - means[u], "add")]; ST["wfix"] = (l, h, att_b[l][0, h]); lcf = lp_of(ib)
        row[nm] = dict(share_base=float(att_b[l][0, h, nb]), share_width=float(att_w[l][0, h, nb]), share_count=float(att_c[l][0, h, nb]),
                       width_K_only=lk - lb, width_V_only=lv - lb, share_width_K_only=sk,
                       count_pattern_fixed=lcf - lb)
    rows.append(row)
    if len(rows) % 20 == 0: print(len(rows), "items", flush=True)
reset()
M = lambda f: float(np.mean([f(r) for r in rows]))
summ = dict(n=len(rows), profile=prof, eff_width=M(lambda r: r["eff_width"]), eff_count=M(lambda r: r["eff_count"]),
            heads={f"L{l}H{h}": {k: M(lambda r, k=k, nm=f"L{l}H{h}": r[nm][k]) for k in rows[0][f"L{l}H{h}"]} for l, h in HEADS})
save_json(dict(summary=summ, rows=rows), "exp17_reader_heads.json")
print(json.dumps(summ, indent=1))
