"""exp26: when do the roles appear during Pythia-410M's training? Exploratory and descriptive (round 4,
item 5). WikiText-2 train only. One run per checkpoint: python3 exp26_checkpoints.py <step|final> [--smoke]

Roles are measured with the head indices found in the final model (one training run, so a head keeps
its index; whether it keeps its role is what is measured). Width 60.
  behaviour : AUROC and separation (breaks against non-breaks with <= 12 characters left) on 30
              paragraphs (seed 26); intact token coefficient (spline in characters remaining plus tokens
              on the line; <= 30 characters left; same paragraphs).
  count     : 5-fold ridge R^2 for characters used from the residual after layer 5 (same points).
  counters  : separation kept with COUNT4 (L2H7, L4H15, L5H6, L5H7) mean-ablated at every position;
              L5H6's attention share to the line-start newline and its correlation with characters used.
  corrector : token coefficient with L5H7 mean-ablated at every position, minus intact.
  stored width: width effect (previous line shortened by >= 10 characters; 60 items, exp14 rules) and
              the share removed by COUNT4 mean-ablated at every newline only.
  readers   : share of the width effect removed when the outputs of L6H2, L8H15 and L17H1 at the decision
              token in the short-line prompt are replaced by their values in the base prompt.
  decision  : share of the width effect removed in the same way for MLPs 14, 16, 18; and for MLPs 21, 23.
Means for ablation: 20 train paragraphs (seed 6), all positions, or newline positions only, computed for
each checkpoint.
"""
import sys, os, json, random, numpy as np, torch
from common import *
from transformers import AutoTokenizer, AutoModelForCausalLM

STEP = sys.argv[1]; SMOKE = "--smoke" in sys.argv
NP = 5 if SMOKE else 30; NIT = 6 if SMOKE else 60; NMEAN = 4 if SMOKE else 20
rev = None if STEP == "final" else f"step{STEP}"
tok = AutoTokenizer.from_pretrained(MODELS["pythia-410m"])
kw = dict(dtype=torch.float32, attn_implementation="eager")
src = MODELS["pythia-410m"]
if rev:  # downloaded with curl from the EleutherAI repository's revision into ~/.cache/d1-checkpoints/<rev>/
    src = os.path.expanduser(f"~/.cache/d1-checkpoints/{rev}"); kw.update(use_safetensors=False)
model = AutoModelForCausalLM.from_pretrained(src, **kw).to("cuda").eval()
nl = newline_id(tok); W = 60
layers = model.gpt_neox.layers; NL = len(layers)
D = model.config.hidden_size; Hn = model.config.num_attention_heads; dh = D // Hn
COUNT4 = [(2, 7), (4, 15), (5, 6), (5, 7)]

ST = dict(abl=[], mask=None, which="all", patch={}, cap=False)
MEAN = {"all": {}, "nl": {}}; CAPZ = {}; CAPM = {}; RES5 = {}
def zhook(l):
    def f(m, a):
        z = a[0]
        if ST["cap"]: CAPZ[l] = z[0, -1].detach().clone()
        hs = [h for (ll, h) in ST["abl"] if ll == l]
        pl = [(h, v) for (k, ll, h), v in ST["patch"].items() if k == "h" and ll == l]
        if not hs and not pl: return None
        z = z.clone()
        for h in hs: z[:, ST["mask"], h * dh:(h + 1) * dh] = MEAN[ST["which"]][l][h * dh:(h + 1) * dh]
        for h, v in pl: z[:, -1, h * dh:(h + 1) * dh] = v
        return (z,)
    return f
def mhook(l):
    def f(m, a, o):
        if ST["cap"]: CAPM[l] = o[0, -1].detach().clone()
        v = ST["patch"].get(("m", l, None))
        if v is None: return None
        o = o.clone(); o[:, -1] = v; return o
    return f
for l in range(NL):
    layers[l].attention.dense.register_forward_pre_hook(zhook(l))
    layers[l].mlp.register_forward_hook(mhook(l))
layers[5].register_forward_hook(lambda m, a, o: RES5.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[0].detach()))
MEANCAP = {}
for l in range(6):
    layers[l].attention.dense.register_forward_pre_hook(lambda m, a, l=l: MEANCAP.__setitem__(l, a[0][0].detach()) if ST.get("meancap") else None)

def run(ids, attn=False):
    with torch.no_grad():
        out = model(torch.tensor([ids], device="cuda"), output_attentions=attn)
        lp = torch.log_softmax(out.logits[0].float(), -1)[:, nl]
    return (lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-6))).cpu().numpy(), lp.cpu().numpy(), out.attentions

def paras_ok(seed, n):
    out = []
    for p in train_paragraphs(seed=seed):
        if len(out) >= n: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        ids, recs = word_end_positions(tok, lines, W)
        if len(ids) > 700: continue
        out.append((ids, recs))
    return out

# ablation means for this checkpoint
ST["meancap"] = True; sa = {}; sn = {}; na = nn_ = 0
for ids, _ in paras_ok(6, NMEAN):
    run(ids)
    nlp = [i for i, t in enumerate(ids) if t == nl]
    for l in range(6):
        sa[l] = sa.get(l, 0) + MEANCAP[l].sum(0); sn[l] = sn.get(l, 0) + MEANCAP[l][nlp].sum(0)
    na += len(ids); nn_ += len(nlp)
ST["meancap"] = False
MEAN["all"] = {l: sa[l] / na for l in sa}; MEAN["nl"] = {l: sn[l] / nn_ for l in sn}

def spline(rem): return np.c_[rem, *[np.maximum(rem - k, 0) for k in [2, 5, 8, 12, 16, 24, 32]]]
nat = paras_ok(26, NP)
def natural(abl=None):
    ST["abl"] = abl or []; ST["which"] = "all"
    br, nb, sc, lab, X, y, feats, used, att56 = [], [], [], [], [], [], [], [], []
    for ids, recs in nat:
        ST["mask"] = torch.arange(len(ids), device="cuda")
        lo, _, att = run(ids, attn=abl is None)
        nlp = [i for i, t in enumerate(ids) if t == nl]
        for r in recs:
            v = lo[r["i"]]; sc.append(v); lab.append(r["label"])
            if r["label"] == 1: br.append(v)
            elif r["remaining"] <= 12: nb.append(v)
            if r["remaining"] <= 30: X.append([r["remaining"], r["ntok_line"]]); y.append(v)
            if abl is None:
                feats.append(RES5["x"][r["i"]].float().cpu().numpy()); used.append(r["used"])
                q0 = [q for q in nlp if q < r["i"]][-1]
                att56.append(float(att[5][0, 6, r["i"], q0]))
    ST["abl"] = []
    X = np.array(X, float); y = np.array(y)
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(y)), spline(X[:, 0]), X[:, 1]], y, rcond=None)
    out = dict(sep=float(np.mean(br) - np.mean(nb)), auroc=auroc(sc, lab), token_coef=float(beta[-1]))
    if abl is None:
        F = np.array(feats); u = np.array(used, float)
        rng = np.random.default_rng(0); idx = rng.permutation(len(u)); r2 = []
        for f in range(5):
            te = idx[f::5]; tr = np.setdiff1d(idx, te); mu = F[tr].mean(0); Ft = F[tr] - mu
            w = np.linalg.solve(Ft.T @ Ft + 10 * np.eye(F.shape[1]), Ft.T @ (u[tr] - u[tr].mean()))
            pr = (F[te] - mu) @ w + u[tr].mean(); r2.append(1 - ((u[te] - pr) ** 2).sum() / ((u[te] - u[te].mean()) ** 2).sum())
        out.update(probe_used_L5=float(np.mean(r2)), L5H6_share=float(np.mean(att56)), L5H6_r_used=float(np.corrcoef(att56, u)[0, 1]))
    return out

res = dict(step=STEP)
res["intact"] = natural()
res["COUNT4_all"] = natural(COUNT4)
res["L5H7_all"] = natural([(5, 7)])
res["counters_sep_kept"] = res["COUNT4_all"]["sep"] / res["intact"]["sep"]
res["corrector_token_coef_change"] = res["L5H7_all"]["token_coef"] - res["intact"]["token_coef"]
print(json.dumps(res), flush=True)

rng = random.Random(14); items = []
for p in train_paragraphs(seed=14):
    if len(items) >= NIT: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    k = rng.randrange(3, min(6, len(lines) - 1))
    words = lines[k].split(" ")
    js = [j for j in range(2, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
    if not js: continue
    prefix = " ".join(words[: js[0] + 1])
    before = lines[:k]; pw = before[-1].split(" "); short = before[-1]
    while len(pw) > 2 and len(before[-1]) - len(short) < 10:
        pw = pw[:-1]; short = " ".join(pw)
    if len(before[-1]) - len(short) < 10: continue
    ib = tok("\n".join(before) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    is_ = tok("\n".join(before[:-1] + [short]) + "\n" + prefix, add_special_tokens=False)["input_ids"]
    items.append((ib, is_))
GROUPS = {"readers3": [("h", 6, 2), ("h", 8, 15), ("h", 17, 1)], "MLP14_16_18": [("m", 14, None), ("m", 16, None), ("m", 18, None)],
          "MLP21_23": [("m", 21, None), ("m", 23, None)]}
eff = {k: [] for k in ["intact", "COUNT4_nl"] + list(GROUPS)}
for ib, is_ in items:
    ST["cap"] = True; lb = run(ib)[1][-1]; capz = dict(CAPZ); capm = dict(CAPM); ST["cap"] = False
    ls_ = run(is_)[1][-1]; eff["intact"].append(ls_ - lb)
    for g, comps in GROUPS.items():
        ST["patch"] = {c: (capz[c[1]][c[2] * dh:(c[2] + 1) * dh] if c[0] == "h" else capm[c[1]]) for c in comps}
        eff[g].append(run(is_)[1][-1] - lb); ST["patch"] = {}
    v = []
    for ids in (ib, is_):
        ST["abl"] = COUNT4; ST["which"] = "nl"; ST["mask"] = torch.tensor([i for i, t in enumerate(ids) if t == nl], device="cuda")
        v.append(run(ids)[1][-1]); ST["abl"] = []
    eff["COUNT4_nl"].append(v[1] - v[0])
E = np.mean(eff["intact"])
res["width_effect"] = float(E); res["width_effect_se"] = float(np.std(eff["intact"]) / np.sqrt(len(items)))
for k in eff:
    if k != "intact": res[f"width_share_removed_{k}"] = float(1 - np.mean(eff[k]) / E)
save_json(res, f"exp26_checkpoint_{STEP}" + ("_smoke" if SMOKE else "") + ".json")
print(json.dumps(res, indent=1))
