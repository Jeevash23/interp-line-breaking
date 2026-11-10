"""D1-X exp e06 (EXPLORATORY, author-approved follow-up 6 Nov evening): why does freezing OLMo-2-1B's
readers (L6H2, L7H10, L11H4) remove more than 100% of the stored length's effect?
Usage: xm_e06_overshoot.py <part> [--smoke]   (part: train or explore-olmo1b; never a conf part)
Width items as xm_conf R (xm_common.width_items, seed 200, paragraph order seed 100), up to 100.
P = base prompt with the line-start marker's residual after layer Lw = 3 patched from the
short-previous-line run. Effects are changes in log P(newline) at the decision token against the
base prompt. "Freeze" = set a component's decision-token output to its base-run value.
 A  replicate: Pw (patch effect), Fr (readers frozen), share = 1 - Fr/Pw.
 B  inject: base run plus only the readers' decision-token outputs from P (sufficiency); and each
    head / MLP above Lw injected alone.
 C  opposers: with the readers frozen, each other head / MLP above Lw also frozen; dFr_k = Fr_k - Fr.
    A positive dFr_k means that component pushes against the readers' direction.
 D  ablation baselines (hypothesis: baseline artefact): readers mean-ablated (all-position train
    means) or resample-ablated (each item takes another item's base-run decision outputs) at the
    decision token in both base and P; share = 1 - effect under ablation / Pw.
 E  each reader frozen alone, and L7H10 + L11H4 (additivity with L6H2).
 F  L6H2 routes at the marker (hypothesis: L6H2 counts and reads): base run plus only L6H2's key
    (k_norm output slice) or only its value (v_proj slice) at the marker, taken from P.
 G  position routes: P, but every position strictly between the marker and the decision token
    keeps its base-run residual after every layer >= Lw, so the patch reaches the decision only
    through attention from the decision token itself; Pw_direct and Fr_direct.
Item bootstrap 1,000 resamples, seed 0, for the main shares.
"""
import sys, json, numpy as np, torch
from xm_common import *

PART = sys.argv[1]; SMOKE = "--smoke" in sys.argv
assert PART in ("train", "explore-olmo1b")
m = M("olmo1b"); tok = m.tok; Lw = 3; T6 = (6, 2)
READ = [(6, 2), (7, 10), (11, 4)]
N, NB = (6, 50) if SMOKE else (100, 1000)
items = width_items(load_paras(PART, seed=100), m, N, seed=200)
pct = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
dev = next(m.model.parameters()).device

# MLP output hooks (capture, and patch with optional batch row)
MCAP, MPATCH = {}, {}
def mhook(l):
    def f(mod, a, o):
        if m.capture: MCAP[l] = o[0].detach().clone()
        lst = MPATCH.get(l)
        if not lst: return None
        o = o.clone()
        for pos, val, row in lst: o[row, pos] = val.to(o.dtype)
        return o
    return f
for l, lay in enumerate(m.layers): lay.mlp.register_forward_hook(mhook(l))
# key / value hooks at layer 6 (capture, and patch one head's slice at one position)
KV = {"k": None, "v": None}; KVP = {"k": None, "v": None}
att6 = m.layers[6].self_attn
def kvhook(which):
    def f(mod, a, o):
        if m.capture: KV[which] = o[0].detach().clone()
        p = KVP[which]
        if p is None: return None
        pos, val = p; o = o.clone(); h = T6[1]
        o[:, pos, h * m.dh:(h + 1) * m.dh] = val.to(o.dtype); return o
    return f
att6.k_norm.register_forward_hook(kvhook("k")); att6.v_proj.register_forward_hook(kvhook("v"))

# all-position means of the o-proj inputs (as xm_conf), for D
acc = {l: [] for l in range(m.L)}
for q, p in enumerate(train_paragraphs(seed=6)[:30]):
    lines = wrap(p, (40, 60, 80)[q % 3])
    if lines is None: continue
    ids = tok("\n".join(lines), add_special_tokens=False)["input_ids"][:700]
    m.lp(ids, capture=True)
    for l in range(m.L): acc[l].append(m.cap[l].float().mean(0))
ZM = {l: torch.stack(acc[l]).mean(0) for l in range(m.L)}
sl = lambda h: slice(h * m.dh, (h + 1) * m.dh)
COMP = [("H", l, h) for l in range(Lw + 1, m.L) for h in range(m.H) if (l, h) not in READ] + [("M", l, None) for l in range(Lw + 1, m.L)]
cname = lambda c: f"L{c[1]}H{c[2]}" if c[0] == "H" else f"MLP{c[1]}"

def run(ids, B=1):
    return m.lp(ids)[-1] if B == 1 else m.lp_last_batch(ids, B)
def clear(): m.patch, m.rpatch = {}, {}; MPATCH.clear(); KVP.update(k=None, v=None)

rec = []
for it, (ib, is_, nb, ns) in enumerate(items):
    T = len(ib) - 1; clear()
    m.lp(is_, capture=True); rsn = m.rcap[Lw][ns].clone()
    lb = m.lp(ib, capture=True)[-1]
    zb = {l: m.cap[l][T].clone() for l in range(m.L)}; mb = {l: MCAP[l][T].clone() for l in range(m.L)}
    rb = {l: m.rcap[l].clone() for l in range(m.L)}
    m.rpatch = {Lw: [(nb, rsn)]}
    lp_ = m.lp(ib, capture=True)[-1]
    zp = {l: m.cap[l][T].clone() for l in range(m.L)}; mp = {l: MCAP[l][T].clone() for l in range(m.L)}
    kp, vp = KV["k"][nb, sl(T6[1])].clone(), KV["v"][nb, sl(T6[1])].clone()
    clear()
    r = dict(Pw=float(lp_ - lb))
    def frz(S, src=zb, row=slice(None)):
        P = {}
        for l, h in S: P.setdefault(l, []).append((h, T, src[l][sl(h)], row))
        return P
    # A
    m.rpatch = {Lw: [(nb, rsn)]}; m.patch = frz(READ); r["Fr"] = float(run(ib) - lb); clear()
    # B
    m.patch = frz(READ, zp); r["inj_readers"] = float(run(ib) - lb); clear()
    rowv = np.zeros(len(COMP)); rowc = np.zeros(len(COMP))
    for s in range(0, len(COMP), 32):
        part = COMP[s:s + 32]; B = len(part)
        for mode in ("inj", "frz"):
            clear()
            if mode == "frz": m.rpatch = {Lw: [(nb, rsn)]}; m.patch = frz(READ)
            for q, c in enumerate(part):
                if c[0] == "H":
                    m.patch.setdefault(c[1], []).append((c[2], T, (zp if mode == "inj" else zb)[c[1]][sl(c[2])], q))
                else:
                    MPATCH.setdefault(c[1], []).append((T, (mp if mode == "inj" else mb)[c[1]], q))
            v = run(ib, B) - lb
            (rowv if mode == "inj" else rowc)[s:s + B] = v
    clear(); r["inj_comp"] = rowv; r["frz_comp"] = rowc - r["Fr"]
    # D (mean and resample ablation of the readers at the decision token, in both runs)
    def abl_eff(src):
        clear(); m.patch = {}
        for l, h in READ: m.patch.setdefault(l, []).append((h, T, src[l][sl(h)]))
        a = run(ib); m.rpatch = {Lw: [(nb, rsn)]}; b_ = run(ib); clear(); return float(b_ - a)
    r["eff_mean"] = abl_eff(ZM)
    r["zb"] = {l: zb[l][[x for h in [h for (ll, h) in READ if ll == l] for x in range(h * m.dh, (h + 1) * m.dh)]].float().cpu() for l in {l for l, _ in READ}}
    # E
    for nm_, S in (("L6H2", [(6, 2)]), ("L7H10", [(7, 10)]), ("L11H4", [(11, 4)]), ("L7H10+L11H4", [(7, 10), (11, 4)])):
        m.rpatch = {Lw: [(nb, rsn)]}; m.patch = frz(S); r["Fr_" + nm_] = float(run(ib) - lb); clear()
    # F
    KVP["k"] = (nb, kp); r["inj_L6H2_key"] = float(run(ib) - lb); clear()
    KVP["v"] = (nb, vp); r["inj_L6H2_value"] = float(run(ib) - lb); clear()
    KVP["k"] = (nb, kp); KVP["v"] = (nb, vp); r["inj_L6H2_kv"] = float(run(ib) - lb); clear()
    # G
    mid = list(range(nb + 1, T))
    def gp():
        P = {Lw: [(nb, rsn)]}
        if mid:
            for l in range(Lw, m.L): P.setdefault(l, []).append((mid, rb[l][mid]))
        return P
    m.rpatch = gp(); r["Pw_direct"] = float(run(ib) - lb)
    m.patch = frz(READ); r["Fr_direct"] = float(run(ib) - lb); clear()
    rec.append((r, ib, nb, rsn, T, lb))
    if it % 20 == 0: print("item", it, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

# D resample: item i takes item (i+1)'s base-run reader outputs
eff_res = []
for i, (r, ib, nb, rsn, T, lb) in enumerate(rec):
    zo = rec[(i + 1) % len(rec)][0]["zb"]; clear()
    for l in zo:
        hs = [h for (ll, h) in READ if ll == l]
        for j, h in enumerate(hs): m.patch.setdefault(l, []).append((h, T, zo[l][j * m.dh:(j + 1) * m.dh].to(dev)))
    a = run(ib); m.rpatch = {Lw: [(nb, rsn)]}; b_ = run(ib); clear(); eff_res.append(float(b_ - a))

R = [x[0] for x in rec]; n = len(R)
arr = lambda k: np.array([x[k] for x in R])
Pw, Fr = arr("Pw"), arr("Fr")
br = np.random.default_rng(0); idx = [br.integers(0, n, n) for _ in range(NB)]
sh = lambda F, P=Pw: dict(share=float(1 - F.mean() / P.mean()), ci95=pct([1 - F[i].mean() / P[i].mean() for i in idx]))
frac = lambda F, P=Pw: dict(frac=float(F.mean() / P.mean()), ci95=pct([F[i].mean() / P[i].mean() for i in idx]))
IC = np.array([x["inj_comp"] for x in R]); FC = np.array([x["frz_comp"] for x in R])
names = [cname(c) for c in COMP]
out = dict(part=PART, smoke=SMOKE, n=n, Lw=Lw, readers="L6H2,L7H10,L11H4",
           Pw=float(Pw.mean()), Pw_ci95=pct([Pw[i].mean() for i in idx]), Fr=float(Fr.mean()), Fr_ci95=pct([Fr[i].mean() for i in idx]),
           A_share=sh(Fr), B_inject_readers=frac(arr("inj_readers")),
           B_inject_top_pos=[(names[i], float(IC[:, i].mean() / Pw.mean())) for i in np.argsort(-IC.mean(0))[:12]],
           B_inject_top_neg=[(names[i], float(IC[:, i].mean() / Pw.mean())) for i in np.argsort(IC.mean(0))[:12]],
           C_opposers=[(names[i], float(FC[:, i].mean() / Pw.mean()), float(FC[:, i].std() / np.sqrt(n) / Pw.mean())) for i in np.argsort(-FC.mean(0))[:15]],
           C_all=dict(zip(names, (FC.mean(0) / Pw.mean()).tolist())), B_all=dict(zip(names, (IC.mean(0) / Pw.mean()).tolist())),
           D_mean_ablation=sh(arr("eff_mean")), D_resample_ablation=sh(np.array(eff_res)),
           E={k: sh(arr("Fr_" + k)) for k in ("L6H2", "L7H10", "L11H4", "L7H10+L11H4")},
           F={k: frac(arr("inj_L6H2_" + k)) for k in ("key", "value", "kv")},
           G=dict(Pw_direct=frac(arr("Pw_direct")), share_direct=sh(arr("Fr_direct"), arr("Pw_direct"))))
for k in ("Pw", "Fr", "A_share", "B_inject_readers", "B_inject_top_pos", "B_inject_top_neg", "C_opposers", "D_mean_ablation", "D_resample_ablation", "E", "F", "G"):
    print(k, json.dumps(out[k]), flush=True)
save_json(out, f"xm_e06_olmo1b_{PART}{'_smoke' if SMOKE else ''}.json")
