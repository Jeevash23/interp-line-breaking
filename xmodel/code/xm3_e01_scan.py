"""D1-X Amendment 3, exp e01 (EXPLORATORY, CHARTER.md): behaviour and single-head scans on a model's exploration part.
Usage: xm_e01_scan.py <model> <part> [--smoke]
 1 Behaviour at widths 40/60/80 (40 paragraphs each): AUROC over all letter-ending decision
   points; separation = mean newline log-odds at breaks minus at non-breaks with <= 12 left.
 2 Attention to the line-start newline at width 60 (25 paragraphs, eager attention): mean share
   per head and its correlation with characters used.
 3 Single-head mean-ablation (all positions; mean of that head's o-proj input over all positions
   of 30 WikiText-2 train paragraphs, seed 6, wrapped at 40/60/80): separation and token
   coefficient on 30 paragraphs at width 60, for every head. Also every MLP (mean output).
"""
import sys, json, numpy as np, torch
from xm3_common import *

MODEL, PART = sys.argv[1], sys.argv[2]; SMOKE = "--smoke" in sys.argv
m = M(MODEL, attn="eager"); tok = m.tok
paras = load_paras(PART, seed=0)
N1, N2, N3 = (4, 3, 4) if SMOKE else (40, 25, 30)
# Amendment 3: "--chunk K/N" runs only the K-th of N slices of the single-head scan (MLPs in the last slice),
# skipping parts 1 and 2 ("--chunk 0/N" runs parts 1 and 2 only), so that each GPU job stays under about 20 minutes. Measures are unchanged.
CHUNK = tuple(int(x) for x in sys.argv[sys.argv.index("--chunk") + 1].split("/")) if "--chunk" in sys.argv else None
out = dict(model=MODEL, part=PART, smoke=SMOKE, L=m.L, H=m.H)

# means
acc = {l: [] for l in range(m.L)}; macc = {l: [] for l in range(m.L)}
mlp_out = {}
mlps = [lay.mlp for lay in m.layers]
for l, mm in enumerate(mlps): mm.register_forward_hook(lambda mod, a, o, l=l: mlp_out.__setitem__(l, (o[0] if isinstance(o, tuple) else o)))
MPATCH = {}
def mlp_patch(l):
    def f(mod, a, o):
        if l not in MPATCH: return None
        x = (o[0] if isinstance(o, tuple) else o).clone(); x[:, m.off:] = MPATCH[l].to(x.dtype)  # never the BOS position (Amendment 3)
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
    return f
for l, mm in enumerate(mlps): mm.register_forward_hook(mlp_patch(l))
for q, p in enumerate(train_paragraphs(seed=6)[:30]):
    lines = wrap(p, (40, 60, 80)[q % 3])
    if lines is None: continue
    ids = tok("\n".join(lines), add_special_tokens=False)["input_ids"][:700]
    m.lp(ids, capture=True)
    for l in range(m.L): acc[l].append(m.cap[l].float().mean(0)); macc[l].append(mlp_out[l][0, m.off:].float().mean(0))
ZM = {l: torch.stack(acc[l]).mean(0) for l in range(m.L)}; MM = {l: torch.stack(macc[l]).mean(0) for l in range(m.L)}

if CHUNK is None or CHUNK[0] == 0:  # chunk 0/N: parts 1 and 2 only
    # 1 behaviour
    beh = {}
    for W in (40, 60, 80):
        sc, lab, br, nb = [], [], [], []
        for ids, recs, _ in wrapped_items(paras, W, N1, tok=tok):
            lp = m.lp(ids)
            for r in recs:
                v = logodds(lp[r["i"]]); sc.append(v); lab.append(r["label"])
                if r["label"] == 1: br.append(v)
                elif r["remaining"] <= 12: nb.append(v)
        beh[W] = dict(n_points=len(sc), n_breaks=int(np.sum(lab)), auroc=auroc(sc, lab), separation=float(np.mean(br) - np.mean(nb)))
        print("behaviour", W, beh[W], flush=True)
    out["behaviour"] = beh; out["gate_auroc60"] = beh[60]["auroc"]; out["gate_pass"] = bool(beh[60]["auroc"] >= 0.90)

    # 2 attention to the line-start newline
    S, U = [], []
    for ids, recs, _ in wrapped_items(paras, 60, N2, tok=tok):
        with torch.no_grad(): o = m.model(torch.tensor([ids], device="cuda"), output_attentions=True)
        A = torch.stack([a[0].float() for a in o.attentions])  # L,H,T,T
        nls = m.nlpos(ids)
        for r in recs:
            i = r["i"]; pn = [x for x in nls if x < i][-1]
            S.append(A[:, :, i, pn].cpu().numpy()); U.append(r["used"])
        del o, A
    S = np.array(S); U = np.array(U, float)
    share = S.mean(0); corr = np.array([[np.corrcoef(S[:, l, h], U)[0, 1] if S[:, l, h].std() > 1e-6 else 0.0 for h in range(m.H)] for l in range(m.L)])
    out["attn"] = dict(n_points=len(U), share=share.tolist(), corr_used=corr.tolist())
    print("attention done", flush=True)

# 3 ablation scans
items = wrapped_items(paras, 60, N3, tok=tok)
def measure():
    br, nb, X, y = [], [], [], []
    for ids, recs, _ in items:
        lp = m.lp(ids)
        for r in recs:
            v = logodds(lp[r["i"]])
            if r["label"] == 1: br.append(v)
            elif r["remaining"] <= 12: nb.append(v)
            if r["remaining"] <= 30: X.append((r["remaining"], r["ntok_line"])); y.append(v)
    return float(np.mean(br) - np.mean(nb)), token_coef(np.array(X, float), np.array(y))
sep0, tc0 = measure()
abl = {}
HL = m.heads() if CHUNK is None else [x for i, x in enumerate(m.heads()) if i * CHUNK[1] // len(m.heads()) == CHUNK[0] - 1]
for l, h in HL:
    m.patch = {l: [(h, slice(None), ZM[l][m.hs(l, h)])]}
    abl[f"L{l}H{h}"] = measure(); m.patch = {}
mabl = {}
for l in (range(m.L) if CHUNK is None or CHUNK[0] == CHUNK[1] else []):
    MPATCH[l] = MM[l]; mabl[f"MLP{l}"] = measure(); MPATCH.clear()
out["ablation"] = dict(intact=dict(separation=sep0, token_coef=tc0), heads={k: dict(separation=v[0], token_coef=v[1]) for k, v in abl.items()},
                       mlps={k: dict(separation=v[0], token_coef=v[1]) for k, v in mabl.items()})
hs = sorted(abl, key=lambda k: abl[k][0])[:10]; tcs = sorted(abl, key=lambda k: -abl[k][1])[:10]
out["top_separation_drop"] = [(k, abl[k][0]) for k in hs]; out["top_token_coef"] = [(k, abl[k][1]) for k in tcs]
if CHUNK is None or CHUNK[0] == 0: out["top_newline_share"] = sorted([(f"L{l}H{h}", float(share[l, h]), float(corr[l, h])) for l, h in m.heads()], key=lambda x: -x[1])[:12]
print(json.dumps(dict(intact=out["ablation"]["intact"], sep=out["top_separation_drop"], tok=out["top_token_coef"], nlshare=out.get("top_newline_share")), indent=0), flush=True)
save_json(out, f"xm_e01_{MODEL}_{PART}{'_smoke' if SMOKE else ''}{f'_chunk{CHUNK[0]}of{CHUNK[1]}' if CHUNK else ''}.json")
