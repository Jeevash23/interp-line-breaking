"""exp29b (EXPLORATORY, CPU): where the newline sits in Pythia-410M's unembedding. Centred unembedding scaled by
the final LayerNorm weight (as exp29). Reports the singular spectrum's tail, the share of the newline's
centred direction in the 24 weakest directions, the same share for every other token (distribution), and the
newline's rank by that share; plus the output weights of named neurons projected on the newline direction
and on the weak directions excluding it."""
import json, sys, numpy as np, torch
from safetensors.torch import load_file
from pathlib import Path
f = sorted(Path.home().glob(".cache/huggingface/hub/models--EleutherAI--pythia-410m/snapshots/*/model.safetensors"))
sd = {}
for p in f: sd.update(load_file(str(p)))
WU = sd["embed_out.weight"].float(); g = sd["gpt_neox.final_layer_norm.weight"].float()
nl = 187
Wc = WU * g; Wc = Wc - Wc.mean(0, keepdim=True)
ev, evec = torch.linalg.eigh(Wc.T @ Wc); null = evec[:, :24]
sh = ((Wc @ null).norm(dim=1) ** 2 / Wc.norm(dim=1) ** 2)
u = Wc[nl] / Wc[nl].norm()
out = dict(sv_tail=[float(x) for x in ev[:30].sqrt()], sv_median=float(ev.sqrt().median()), nl_null_share=float(sh[nl]),
           token_null_share_median=float(sh.median()), token_null_share_p99=float(torch.quantile(sh, 0.99)),
           nl_rank_by_null_share=int((sh > sh[nl]).sum()) + 1, nl_norm=float(Wc[nl].norm()), token_norm_median=float(Wc.norm(dim=1).median()))
neur = {}
for name in sys.argv[1:]:
    l, i = name[3:].split("."); w = sd[f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.weight"].float()[:, int(i)]
    wn = w / w.norm(); pn = null @ (null.T @ wn); pn_perp = pn - (pn @ u) * u
    logit = Wc @ w  # effect on each token's centred logit per unit activation (before LayerNorm scale)
    top = torch.topk(-logit, 8).indices.tolist(); bot = torch.topk(logit, 8).indices.tolist()
    neur[name] = dict(norm=float(w.norm()), cos_nl=float(wn @ u), null_share=float((null.T @ wn).norm() ** 2),
                      null_share_without_nl=float(pn_perp.norm() ** 2), nl_logit=float(logit[nl]), nl_logit_rank_from_bottom=int((logit < logit[nl]).sum()) + 1,
                      most_suppressed=top, most_promoted=bot, logit_sd=float(logit.std()))
out["neurons"] = neur
print(json.dumps(out, indent=1))
(Path(__file__).resolve().parent.parent / "outputs" / "exp29b_unembed.json").write_text(json.dumps(out, indent=1))
