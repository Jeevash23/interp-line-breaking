"""exp09: causal test of the dilution reading. Exploratory.

At a word-end decision point (3..16 characters left, width 60), add delta to one head's
pre-softmax score from the decision point to the previous newline (and only there).
If L5H6's newline share is the character count, raising it (delta > 0) should make the
model act as if fewer characters were used (lower P(newline)), lowering it the reverse,
graded with delta. Controls: the same edit in L5H7, L4H10 and L5H3.

Also reported: the newline share after the edit, and the change in log P(newline).
"""
import json, numpy as np, torch
import transformers.models.gpt_neox.modeling_gpt_neox as gm
from common import *

W = 60
tok, model = load("pythia-410m", attn="eager")
nl = newline_id(tok)
DELTAS = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
STATE = dict(layer=None, head=None, key=None, deltas=None, share=None)
_orig = gm.eager_attention_forward

def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    if STATE["layer"] is None or module.layer_idx != STATE["layer"]:
        return _orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
    s = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None: s = s + attention_mask
    d = torch.tensor(STATE["deltas"], device=s.device, dtype=s.dtype)
    s[:, STATE["head"], -1, STATE["key"]] += d
    w = torch.softmax(s, -1, dtype=torch.float32).to(query.dtype)
    STATE["share"] = w[:, STATE["head"], -1, STATE["key"]].float().cpu().numpy()
    out = torch.matmul(w, value).transpose(1, 2).contiguous()
    return out, w
gm.eager_attention_forward = patched

items = []
for p in train_paragraphs(seed=9):
    if len(items) >= 30: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    ids, recs = word_end_positions(tok, lines, W)
    if len(ids) > 700: continue
    items.append((ids, recs))

HEADS = [(5, 6), (5, 7), (4, 10), (5, 3)]
res = {}
for (l, h) in HEADS:
    rows = []
    for ids, recs in items:
        nls = [i for i, t in enumerate(ids) if t == nl]
        for r in recs:
            if not (3 <= r["remaining"] <= 16): continue
            i = r["i"]; pn = [x for x in nls if x < i][-1]
            STATE.update(layer=l, head=h, key=pn, deltas=DELTAS)
            with torch.no_grad():
                x = torch.tensor([ids[: i + 1]] * len(DELTAS), device="cuda")
                hid = model.gpt_neox(x).last_hidden_state[:, -1]
                lp = torch.log_softmax(model.lm_head(hid).float(), -1)[:, nl].cpu().numpy()
            rows.append(np.r_[r["remaining"], lp - lp[DELTAS.index(0.0)], STATE["share"]])
    STATE["layer"] = None
    a = np.array(rows)
    k = len(DELTAS)
    res[f"L{l}H{h}"] = dict(n=len(a), mean_dlogp={d: float(a[:, 1 + j].mean()) for j, d in enumerate(DELTAS)},
                           mean_share={d: float(a[:, 1 + k + j].mean()) for j, d in enumerate(DELTAS)})
    print(f"L{l}H{h}", "n", len(a), " ".join(f"{d:+.1f}:{a[:,1+j].mean():+.2f}(share {a[:,1+k+j].mean():.2f})" for j, d in enumerate(DELTAS)), flush=True)
save_json(res, "exp09_attention_edit.json")
