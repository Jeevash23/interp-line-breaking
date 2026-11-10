"""exp15: do Pythia-1B and GPT-2 Small have line-start counting heads? Exploratory.

Same pipeline for three models (Pythia-410M as the reference, Pythia-1B, GPT-2 Small),
width 60, training-split paragraphs:
 (a) For every head, at word-end decision points: mean attention share on the previous
     newline token, and its correlation with characters used.
 (b) Candidate counting heads: the 4 heads with the most negative correlation among heads
     with mean share >= 0.05. Control: the head with the highest mean share among those
     with |corr| < 0.3 (if any).
 (c) Causal edit (exp09 method): add d in {-1, +1} to the candidate head's pre-softmax score
     from the decision point to the previous newline; change in log P(newline), on points
     with 3..16 characters left.
"""
import sys, json, numpy as np, torch
from common import *
import transformers.models.gpt_neox.modeling_gpt_neox as gm
import transformers.models.gpt2.modeling_gpt2 as g2

W = 60
STATE = dict(layer=None, head=None, key=None, deltas=None)

def make_patched(orig):
    def patched(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        if STATE["layer"] is None or module.layer_idx != STATE["layer"]:
            return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
        if scaling is None: scaling = query.size(-1) ** -0.5
        s = torch.matmul(query, key.transpose(-1, -2)).float() * scaling
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                s = s.masked_fill(~attention_mask, float("-inf"))
            else:
                s = s + attention_mask.float()
        d = torch.tensor(STATE["deltas"], device=s.device, dtype=s.dtype)
        s[:, STATE["head"], -1, STATE["key"]] += d
        w = torch.softmax(s, -1).to(value.dtype)
        out = torch.matmul(w, value).transpose(1, 2).contiguous()
        return out, w
    return patched
gm.eager_attention_forward = make_patched(gm.eager_attention_forward)
g2.eager_attention_forward = make_patched(g2.eager_attention_forward)

def base_model(model):
    return model.gpt_neox if hasattr(model, "gpt_neox") else model.transformer
def head_logits(model, hid):
    return model.lm_head(hid).float() if hasattr(model, "lm_head") else model.embed_out(hid).float()

results = {}
for mname in ["pythia-410m", "pythia-1b", "gpt2"]:
    tok, model = load(mname, attn="eager")
    nl = newline_id(tok)
    bm = base_model(model)
    items = []
    for p in train_paragraphs(seed=15):
        if len(items) >= 25: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 6: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        items.append((ids, recs))
    # (a)
    shares, used, labs, lps = [], [], [], []
    with torch.no_grad():
        for ids, recs in items:
            out = bm(torch.tensor([ids], device="cuda"), output_attentions=True)
            A = torch.stack([a[0] for a in out.attentions]).float()  # (L, H, T, T)
            lp = torch.log_softmax(head_logits(model, out.last_hidden_state[0]), -1)[:, nl].cpu().numpy()
            nls = [i for i, t in enumerate(ids) if t == nl]
            for r in recs:
                i = r["i"]; pn = [x for x in nls if x < i][-1]
                shares.append(A[:, :, i, pn].cpu().numpy()); used.append(r["used"]); labs.append(r["label"]); lps.append(lp[i])
            del out, A
    S = np.array(shares); U = np.array(used, float)
    L, H = S.shape[1:]
    mean_share = S.mean(0)
    corr = np.array([[np.corrcoef(S[:, l, h], U)[0, 1] if S[:, l, h].std() > 1e-6 else 0 for h in range(H)] for l in range(L)])
    flat = [(l, h) for l in range(L) for h in range(H)]
    cands = sorted([x for x in flat if mean_share[x] >= 0.05], key=lambda x: corr[x])[:4]
    ctl = sorted([x for x in flat if abs(corr[x]) < 0.3 and x not in cands], key=lambda x: -mean_share[x])[:1]
    # (c)
    edits = {}
    for (l, h) in cands + ctl:
        rows = []
        for ids, recs in items[:15]:
            nls = [i for i, t in enumerate(ids) if t == nl]
            for r in recs:
                if not (3 <= r["remaining"] <= 16): continue
                i = r["i"]; pn = [x for x in nls if x < i][-1]
                STATE.update(layer=l, head=h, key=pn, deltas=[-1.0, 0.0, 1.0])
                with torch.no_grad():
                    hid = bm(torch.tensor([ids[: i + 1]] * 3, device="cuda")).last_hidden_state[:, -1]
                    lp = torch.log_softmax(head_logits(model, hid), -1)[:, nl].cpu().numpy()
                rows.append((lp[0] - lp[1], lp[2] - lp[1]))
        STATE["layer"] = None
        a = np.array(rows)
        edits[f"L{l}H{h}"] = dict(role="control" if (l, h) in ctl else "candidate", n=len(a),
                                  mean_share=float(mean_share[l, h]), corr_used=float(corr[l, h]),
                                  dlogp_minus1=float(a[:, 0].mean()), dlogp_plus1=float(a[:, 1].mean()))
    top_share = sorted(flat, key=lambda x: -mean_share[x])[:8]
    results[mname] = dict(n_points=len(U), auroc=auroc(lps, labs),
                          top_by_share=[dict(head=f"L{l}H{h}", share=float(mean_share[l, h]), corr_used=float(corr[l, h])) for l, h in top_share],
                          edits=edits,
                          n_heads_corr_below_m05=int((corr < -0.5).sum()), n_heads=int(L * H))
    print(mname, json.dumps(results[mname], indent=1), flush=True)
    del model; torch.cuda.empty_cache()
save_json(results, "exp15_cross_model.json")
