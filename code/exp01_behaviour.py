"""exp01: behaviour map. How well does each model predict line breaks in greedily
wrapped WikiText-2 (train split) paragraphs, by width? Exploratory.

Writes outputs/exp01_records_<model>.npz (one row per decision point) and
outputs/exp01_summary.json.
"""
import sys, time, json
import numpy as np, torch
from common import *

WIDTHS = [20, 30, 40, 50, 60, 80, 100, 120, 160]
N_PARA = int(sys.argv[2]) if len(sys.argv) > 2 else 60
models = sys.argv[1].split(",") if len(sys.argv) > 1 else ["pythia-410m", "pythia-1b", "gpt2"]

paras = train_paragraphs()
summary = json.loads((OUT / "exp01_summary.json").read_text()) if (OUT / "exp01_summary.json").exists() else {}

for mname in models:
    t0 = time.time()
    tok, model = load(mname)
    nl = newline_id(tok)
    rows = []
    for W in WIDTHS:
        used_paras = 0
        for pi, p in enumerate(paras):
            if used_paras >= N_PARA:
                break
            lines = wrap(p, W)
            if lines is None or len(lines) < 5:
                continue
            ids, recs = word_end_positions(tok, lines, W)
            if not recs or len(ids) > 1500:
                continue
            used_paras += 1
            with torch.no_grad():
                logits = model(torch.tensor([ids], device="cuda")).logits[0].float()
                lp = torch.log_softmax(logits, -1)
            idx = torch.tensor([r["i"] for r in recs], device="cuda")
            lpn = lp[idx, nl].cpu().numpy()
            top = lp[idx].argmax(-1).cpu().numpy()
            for r, a, b in zip(recs, lpn, top):
                rows.append((W, pi, r["i"], r["line"], r["used"], r["remaining"], r["label"], r["nextlen"],
                             r["prevlen"], r["ntok_line"], float(a), int(b == nl)))
    arr = np.array(rows, dtype=float)
    cols = "W,para,i,line,used,remaining,label,nextlen,prevlen,ntok_line,logp_nl,top_is_nl".split(",")
    np.savez_compressed(OUT / f"exp01_records_{mname}.npz", data=arr, cols=np.array(cols))
    res = {}
    for W in WIDTHS:
        m = arr[:, 0] == W
        a = arr[m]
        lab = a[:, 6].astype(int)
        lpn, top = a[:, 10], a[:, 11]
        rem, nxt = a[:, 5], a[:, 7]
        res[W] = dict(
            n_points=int(m.sum()), n_breaks=int(lab.sum()),
            break_acc=float(top[lab == 1].mean()), false_break=float(top[lab == 0].mean()),
            auroc_model=auroc(lpn, lab),
            auroc_remaining_only=auroc(-rem, lab),
            auroc_oracle=auroc(nxt + 1 - rem, lab),
            mean_logp_break=float(lpn[lab == 1].mean()), mean_logp_cont=float(lpn[lab == 0].mean()),
        )
    summary[mname] = dict(by_width=res, n_para=N_PARA, seconds=round(time.time() - t0, 1))
    save_json(summary, "exp01_summary.json")
    print(mname, json.dumps({W: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for W, d in res.items()}, indent=0), flush=True)
    del model; torch.cuda.empty_cache()
