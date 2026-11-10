"""D1-X Amendment 3: does Gemma-4-E2B fit, and do the hooks behave? (set-up check, not a result)
Usage: xm3_gemma_fit.py
 * loads the text decoder on the GPU (bfloat16, eager attention); reports GPU memory after loading and
   the peak for a 700-token input with captures and with attention weights returned, and the
   process's peak RAM;
 * BOS and chunking: m.lp and m.nll equal the raw model's full output with BOS prepended (BOS
   position dropped), on 200 tokens;
 * hooks: patching a head with its own captured value changes nothing; patching it with zeros
   changes the output (layer 4, a global layer with 512-wide heads, and layer 0, 256-wide);
 * behaviour: AUROC and separation at width 60 on 8 WikiText-2 train paragraphs, and mean next-token
   loss, as a check that the loading and the per-layer embedding lookup are right.
Amendment 3 rule: skip Gemma if the peak exceeds about 5.5 GB of GPU memory for 700-token inputs."""
import json, resource, time, numpy as np, torch
from xm3_common import *

t0 = time.time()
m = M("gemma4-e2b", attn="eager"); tok = m.tok
out = dict(L=m.L, H=m.H, dhl=m.dhl, off=m.off, load_s=round(time.time() - t0, 1),
           gpu_after_load_gb=torch.cuda.memory_allocated() / 2**30)
paras = load_paras("train", seed=0)
items = wrapped_items(paras, 60, 8, tok=tok)
txt = []  # wrapped paragraphs joined until the input reaches 700 tokens (the scripts' maximum length)
for p in paras:
    lines = wrap(p, 60)
    if lines is None: continue
    txt += lines
    x = tok("\n".join(txt), add_special_tokens=False)["input_ids"]
    if len(x) >= 700: ids = x[:700]; break
out["long_len"] = len(ids)
torch.cuda.reset_peak_memory_stats()
lp = m.lp(ids, capture=True)
out["peak_lp_capture_gb"] = torch.cuda.max_memory_allocated() / 2**30
torch.cuda.reset_peak_memory_stats()
with torch.no_grad(): o = m.model(torch.tensor([ids], device="cuda"), output_attentions=True)
out["peak_attn_gb"] = torch.cuda.max_memory_allocated() / 2**30
out["attn_shape"] = list(o.attentions[0].shape); out["attn_shapes_distinct"] = sorted({tuple(a.shape) for a in o.attentions}); del o

# BOS, chunked log P(newline) and chunked loss, against the raw model's full output (200 tokens)
raw = m.model._m; ids_s = ids[:200]
with torch.no_grad():
    lg = raw(torch.tensor([[tok.bos_token_id] + ids_s], device="cuda")).logits[0, 1:].float()
ref = (lg[:, m.nl] - torch.logsumexp(lg, -1)).cpu().numpy()
refnll = (torch.logsumexp(lg[:-1], -1) - lg[:-1].gather(1, torch.tensor(ids_s[1:], device="cuda")[:, None])[:, 0]).cpu().numpy()
out["bos_lp_max_abs_diff"] = float(np.abs(ref - m.lp(ids_s)).max())
out["nll_max_abs_diff"] = float(np.abs(refnll - m.nll(ids_s)).max())
del lg

# hook checks
cap = {l: m.cap[l].clone() for l in (0, 4)}
for l in (0, 4):
    m.patch = {l: [(3, slice(None), cap[l][:, m.hs(l, 3)])]}; same = m.lp(ids)
    m.patch = {l: [(3, slice(None), torch.zeros(m.dhl[l]))]}; zero = m.lp(ids); m.patch = {}
    out[f"hook_L{l}"] = dict(width=m.dhl[l], self_patch_max_diff=float(np.abs(same - lp).max()), zero_patch_mean_diff=float(np.abs(zero - lp).mean()))

# behaviour and loss
sc, lab, br, nb, ce = [], [], [], [], []
for ids2, recs, _ in items:
    lp2 = m.lp(ids2); ce.append(float(m.nll(ids2).mean()))
    for r in recs:
        v = logodds(lp2[r["i"]]); sc.append(v); lab.append(r["label"])
        if r["label"] == 1: br.append(v)
        elif r["remaining"] <= 12: nb.append(v)
out["behaviour60"] = dict(n_points=len(sc), auroc=auroc(sc, lab), separation=float(np.mean(br) - np.mean(nb)), mean_ce=float(np.mean(ce)))
out["peak_overall_gb"] = torch.cuda.max_memory_allocated() / 2**30
out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
out["fits"] = bool(max(out["peak_lp_capture_gb"], out["peak_attn_gb"]) <= 5.5)
print(json.dumps(out, indent=1), flush=True)
save_json(out, "xm3_gemma_fit.json")
