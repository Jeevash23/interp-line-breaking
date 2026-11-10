"""D1-X exp e04 (EXPLORATORY, CHARTER.md): role tests for candidate head sets, each against 10
random sets matched layer by layer (each random head drawn from the same layer as the candidate
head it stands for, excluding all candidate heads; numpy seed 0).
Usage: xm_e04_roles.py <model> <part> --count A,B --store A,B --corr A,B --read A,B [--lw N] [--tag T] [--smoke]
Heads are written L<l>H<h>. Any role may be omitted.
 count: mean-ablation (all-position means, as e01) at every position; natural text width 60
        (40 paragraphs): separation kept and AUROC; and mean next-token CE increase on
        non-decision positions (10 paragraphs), as a general-damage control.
 store: mean-ablation at line-start marker positions only (marker-position means from the same 30
        train paragraphs); share of the width effect removed (width items as e02, up to 100, same
        ablation in both prompts) and natural-text separation kept (40 paragraphs, width 60).
 corr:  token coefficient (e01's measure, <= 30 left) at widths 40/60/80, 30 paragraphs each,
        intact vs ablated (all positions).
 read:  with the marker residual after layer Lw patched from the short run, the set frozen at the
        decision token to its base-run value; mediation share = 1 - frozen / patched effect.
        Lw = argmax of the residual-patch effect over the first half of the layers (as e02)
        unless given.
Item or paragraph bootstrap, 1,000 resamples, seed 0, for the candidate's numbers.
"""
import sys, json, numpy as np, torch
from xm_common import *

MODEL, PART = sys.argv[1], sys.argv[2]; SMOKE = "--smoke" in sys.argv
def arg(k, d=None):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
def parse(s): return [] if not s else [tuple(int(x) for x in t[1:].split("H")) for t in s.split(",")]
ROLES = {r: parse(arg("--" + r)) for r in ("count", "store", "corr", "read")}
TAG = arg("--tag", "a")
m = M(MODEL); tok = m.tok
paras = load_paras(PART, seed=0)
NN, NC, NW, ND, NB = (4, 3, 6, 3, 50) if SMOKE else (40, 30, 100, 10, 1000)
ALLC = {h for v in ROLES.values() for h in v}
rs = np.random.default_rng(0)
RLOG = []
def randsets(S, n=10):
    out = []
    for _ in range(n):  # layer-matched, without repeats within a set (changed 6 Nov 07:00 UTC after runs a/b drew repeats)
        r = []
        for l, _ in S: r.append((l, int(rs.choice([h for h in range(m.H) if (l, h) not in ALLC and (l, h) not in r]))))
        out.append(r)
    RLOG.append([",".join(f"L{l}H{h}" for l, h in r) for r in out]); return out
pct = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
name = lambda S: ",".join(f"L{l}H{h}" for l, h in S)

# means
acc = {l: [] for l in range(m.L)}; nacc = {l: [] for l in range(m.L)}
for q, p in enumerate(train_paragraphs(seed=6)[:30]):
    lines = wrap(p, (40, 60, 80)[q % 3])
    if lines is None: continue
    ids = tok("\n".join(lines), add_special_tokens=False)["input_ids"][:700]
    m.lp(ids, capture=True); pos = m.nlpos(ids)
    for l in range(m.L): acc[l].append(m.cap[l].float().mean(0)); nacc[l].append(m.cap[l][pos].float())
ZM = {l: torch.stack(acc[l]).mean(0) for l in range(m.L)}; ZN = {l: torch.cat(nacc[l]).mean(0) for l in range(m.L)}
def setpatch(S, where, ids=None, mean=None):
    mean = mean or ZM; P = {}
    pos = slice(None) if where == "all" else m.nlpos(ids)
    for l, h in S: P.setdefault(l, []).append((h, pos, mean[l][h * m.dh:(h + 1) * m.dh]))
    return P

nat = wrapped_items(paras, 60, NN, tok=tok)
def separation(S=(), where=None, mean=None):
    br, nb, sc, lab, per = [], [], [], [], []
    for ids, recs, _ in nat:
        m.patch = setpatch(S, where, ids, mean) if S else {}
        lp = m.lp(ids); m.patch = {}
        b_, n_ = [], []
        for r in recs:
            v = logodds(lp[r["i"]]); sc.append(v); lab.append(r["label"])
            if r["label"] == 1: br.append(v); b_.append(v)
            elif r["remaining"] <= 12: nb.append(v); n_.append(v)
        per.append((b_, n_))
    return float(np.mean(br) - np.mean(nb)), auroc(sc, lab), per
dm = wrapped_items(paras, 60, ND, tok=tok)
def ce(S=()):
    tot, n = 0.0, 0
    for ids, recs, _ in dm:
        m.patch = setpatch(S, "all") if S else {}
        dec = {r["i"] for r in recs}
        with torch.no_grad():
            lg = torch.log_softmax(m.model(torch.tensor([ids], device="cuda")).logits[0].float(), -1)
        m.patch = {}
        t = torch.tensor(ids[1:], device="cuda"); nll = -lg[:-1].gather(1, t[:, None])[:, 0].cpu().numpy()
        keep = np.array([i not in dec for i in range(len(ids) - 1)]); tot += nll[keep].sum(); n += keep.sum()
    return float(tot / n)
def boot_sep(per0, per1):
    br = np.random.default_rng(0); k = len(per0); out = []
    for _ in range(NB):
        ii = br.integers(0, k, k)
        s0 = np.mean(sum([per0[i][0] for i in ii], [])) - np.mean(sum([per0[i][1] for i in ii], []))
        s1 = np.mean(sum([per1[i][0] for i in ii], [])) - np.mean(sum([per1[i][1] for i in ii], []))
        out.append(s1 / s0)
    return pct(out)

out = dict(model=MODEL, part=PART, smoke=SMOKE, tag=TAG, roles={r: name(S) for r, S in ROLES.items()})
sep0, au0, per0 = separation(); ce0 = ce()
out["intact"] = dict(separation=sep0, auroc=au0, ce=ce0)
print("intact", out["intact"], flush=True)

if ROLES["count"]:
    S = ROLES["count"]; s, a, per = separation(S, "all")
    R = [separation(r, "all")[:2] + (ce(r) - ce0,) for r in randsets(S)]
    out["count"] = dict(set=name(S), kept=s / sep0, kept_ci95=boot_sep(per0, per), auroc=a, ce_increase=ce(S) - ce0,
                        random_kept=[x[0] / sep0 for x in R], random_auroc=[x[1] for x in R], random_ce_increase=[x[2] for x in R])
    print("count", json.dumps(out["count"]), flush=True)

if ROLES["corr"]:
    S = ROLES["corr"]; RS = randsets(S); res = {}
    for W in (40, 60, 80):
        its = wrapped_items(paras, W, NC, tok=tok)
        def tc(SS):
            X, y, g = [], [], []
            for k, (ids, recs, _) in enumerate(its):
                m.patch = setpatch(SS, "all") if SS else {}
                lp = m.lp(ids); m.patch = {}
                for r in recs:
                    if r["remaining"] <= 30: X.append((r["remaining"], r["ntok_line"])); y.append(logodds(lp[r["i"]])); g.append(k)
            return np.array(X, float), np.array(y), np.array(g)
        X0, y0, g = tc(()); X1, y1, _ = tc(S)
        groups = [np.where(g == k)[0] for k in range(len(its))]; br = np.random.default_rng(0); b0, b1 = [], []
        for _ in range(NB):
            ii = np.concatenate([groups[j] for j in br.integers(0, len(its), len(its))])
            b0.append(token_coef(X0[ii], y0[ii])); b1.append(token_coef(X1[ii], y1[ii]))
        res[W] = dict(n_points=len(y0), intact=token_coef(X0, y0), intact_ci95=pct(b0), ablated=token_coef(X1, y1), ablated_ci95=pct(b1),
                      diff_ci95=pct(np.array(b1) - np.array(b0)), random=[token_coef(*tc(r)[:2]) for r in RS])
        print("corr", W, json.dumps(res[W]), flush=True)
    out["corr"] = dict(set=name(S), widths=res)

items = width_items(paras, m, NW) if (ROLES["store"] or ROLES["read"]) else []
def width_eff(P_b=None, P_s=None, extra=None):
    """P_b, P_s: functions ids -> patch dict for base and short prompts."""
    e = []
    for ib, is_, nb, ns in items:
        m.patch = P_b(ib) if P_b else {}; lb = m.lp(ib)[-1]
        m.patch = P_s(is_) if P_s else {}; ls = m.lp(is_)[-1]; m.patch = {}
        e.append(ls - lb)
    return np.array(e)
if items: out["n_width_items"] = len(items)

if ROLES["store"]:
    S = ROLES["store"]; E0 = width_eff()
    f = lambda SS: (lambda ids: setpatch(SS, "nl", ids, ZN))
    E1 = width_eff(f(S), f(S))
    br = np.random.default_rng(0); n = len(E0); idx = [br.integers(0, n, n) for _ in range(NB)]
    Rsh = [float(1 - width_eff(f(r), f(r)).mean() / E0.mean()) for r in randsets(S)]
    s, a, per = separation(S, "nl", ZN)
    out["store"] = dict(set=name(S), width_effect=float(E0.mean()), width_effect_ci95=pct([E0[i].mean() for i in idx]),
                        share_removed=float(1 - E1.mean() / E0.mean()), share_ci95=pct([1 - E1[i].mean() / E0[i].mean() for i in idx]),
                        random_share=Rsh, nat_kept=s / sep0, nat_kept_ci95=boot_sep(per0, per), nat_auroc=a)
    print("store", json.dumps(out["store"]), flush=True)

if ROLES["read"]:
    S = ROLES["read"]; Lw = int(arg("--lw", -1)); cache = []
    for ib, is_, nb, ns in items:
        m.patch, m.rpatch = {}, {}
        lb = m.lp(ib, capture=True)[-1]; zb = {l: m.cap[l][-1].clone() for l in range(m.L)}
        m.lp(is_, capture=True); rsn = {l: m.rcap[l][ns].clone() for l in range(m.L)}
        cache.append((ib, nb, lb, zb, rsn))
    if Lw < 0:
        Bm = np.zeros(m.L)
        for ib, nb, lb, zb, rsn in cache:
            m.rpatch = {l: [(nb, rsn[l], l)] for l in range(m.L)}; Bm += m.lp_last_batch(ib, m.L) - lb; m.rpatch = {}
        Lw = int(np.argmax((Bm / len(cache))[: m.L // 2]))
    def med(SS):
        pw, fr = [], []
        for ib, nb, lb, zb, rsn in cache:
            m.rpatch = {Lw: [(nb, rsn[Lw])]}; m.patch = {}
            pw.append(m.lp(ib)[-1] - lb)
            m.patch = {}
            for l, h in SS: m.patch.setdefault(l, []).append((h, len(ib) - 1, zb[l][h * m.dh:(h + 1) * m.dh]))
            fr.append(m.lp(ib)[-1] - lb); m.patch, m.rpatch = {}, {}
        return np.array(pw), np.array(fr)
    Pw, Fr = med(S); n = len(Pw); br = np.random.default_rng(0); idx = [br.integers(0, n, n) for _ in range(NB)]
    out["read"] = dict(set=name(S), Lw=Lw, patch_effect=float(Pw.mean()), patch_effect_ci95=pct([Pw[i].mean() for i in idx]),
                       share=float(1 - Fr.mean() / Pw.mean()), share_ci95=pct([1 - Fr[i].mean() / Pw[i].mean() for i in idx]),
                       random_share=[float(1 - med(r)[1].mean() / Pw.mean()) for r in randsets(S)])
    print("read", json.dumps(out["read"]), flush=True)

out["random_sets_drawn"] = RLOG
save_json(out, f"xm_e04_{MODEL}_{PART}_{TAG}{'_smoke' if SMOKE else ''}.json")
