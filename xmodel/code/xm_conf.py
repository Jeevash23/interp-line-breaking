"""D1-X confirmation script (CONFIRMATORY when run on part conf-<model> under a frozen protocol).
Usage: xm_conf.py <spec.json> <part> [--smoke]
The spec (frozen with the protocol) names the model, the head sets for each role and the
thresholds. Parts allowed: "train" (pipeline check with --smoke only), explore-<model>, conf-<model>.
Paragraph order: seed 100. Ablation means: WikiText-2 train, 30 paragraphs (seed 6) wrapped at
40/60/80 in turn, first 700 tokens; all-position means, and line-start-marker means for the
newline-only ablation (as xm_e04). Random sets: 10 per test, layer-matched to the candidate set,
no repeats, never a head named in any role (numpy seed 0, drawn in the order G, C, T, S, R).
Bootstrap: 2,000 resamples (item or paragraph), numpy seed 0, identical across conditions.

 G  behaviour: AUROC and separation at width 60, letter-ending decision points, first 60 paragraphs
    with >= 5 lines and <= 700 tokens (as e01). Reported; the charter's gate (>= 0.90).
 C  counters, count interchange (as xm_e05; width 60, random seed 300, up to 150 items):
    share of the short-minus-base effect carried by the set's decision-token output.
 T  token correctors (as xm_e04 corr): token coefficient at widths 40, 60, 80 (first 60
    paragraphs per width), intact vs set mean-ablated at all positions; paired differences.
 S  stored length (as xm_e04 store): width items (xm_common.width_items, seed 200, up to 150);
    share of the width effect removed by newline-only ablation of the set (same ablation in both
    prompts); and the natural-text separation kept (G's paragraphs) under the same ablation.
 R  readers (as xm_e04 read): marker residual after layer Lw (from the spec) patched from the
    short run; mediation share of the set frozen at the decision token.
"""
import sys, json, random, numpy as np, torch
from xm_common import *

spec = json.loads(Path(sys.argv[1]).read_text()); PART = sys.argv[2]; SMOKE = "--smoke" in sys.argv
MODEL = spec["model"]
assert PART in ("train", f"explore-{spec['part_key']}", f"conf-{spec['part_key']}"), PART
assert SMOKE or PART != "train"
parse = lambda s: [tuple(int(x) for x in t[1:].split("H")) for t in s.split(",")] if s else []
SETS = {r: parse(spec["sets"].get(r, "")) for r in "CTSRX"}  # X: an optional second stored-length set
TH = spec["thresholds"]
NP, NI, NB = (5, 8, 50) if SMOKE else (60, 150, 2000)
m = M(MODEL); tok = m.tok
paras = load_paras(PART, seed=100)
pct = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
ALLC = {h for S in SETS.values() for h in S}
rs = np.random.default_rng(0)
def randsets(S, n=10):
    out = []
    for _ in range(n):
        r = []
        for l, _h in S: r.append((l, int(rs.choice([h for h in range(m.H) if (l, h) not in ALLC and (l, h) not in r]))))
        out.append(r)
    return out
nm = lambda S: ",".join(f"L{l}H{h}" for l, h in S)
RAND = {r: (randsets(SETS[r]) if SETS[r] else []) for r in "CTSRX"}
out = dict(model=MODEL, part=PART, smoke=SMOKE, spec=spec, random_sets={r: [nm(x) for x in v] for r, v in RAND.items()})

acc = {l: [] for l in range(m.L)}; nacc = {l: [] for l in range(m.L)}
for q, p in enumerate(train_paragraphs(seed=6)[:30]):
    lines = wrap(p, (40, 60, 80)[q % 3])
    if lines is None: continue
    ids = tok("\n".join(lines), add_special_tokens=False)["input_ids"][:700]
    m.lp(ids, capture=True); pos = m.nlpos(ids)
    for l in range(m.L): acc[l].append(m.cap[l].float().mean(0)); nacc[l].append(m.cap[l][pos].float())
ZM = {l: torch.stack(acc[l]).mean(0) for l in range(m.L)}; ZN = {l: torch.cat(nacc[l]).mean(0) for l in range(m.L)}
def setpatch(S, where, ids=None, mean=None):
    mean = mean or ZM; P = {}; pos = slice(None) if where == "all" else m.nlpos(ids)
    for l, h in S: P.setdefault(l, []).append((h, pos, mean[l][h * m.dh:(h + 1) * m.dh]))
    return P

# ---------------- G
nat = wrapped_items(paras, 60, NP, tok=tok)
def natural(S=(), where=None, mean=None):
    per, sc, lab = [], [], []
    for ids, recs, _ in nat:
        m.patch = setpatch(S, where, ids, mean) if S else {}
        lp = m.lp(ids); m.patch = {}; b_, n_ = [], []
        for r in recs:
            v = float(logodds(lp[r["i"]])); sc.append(v); lab.append(r["label"])
            if r["label"] == 1: b_.append(v)
            elif r["remaining"] <= 12: n_.append(v)
        per.append((b_, n_))
    return per, auroc(sc, lab)
def sepof(per, ii=None):
    ii = range(len(per)) if ii is None else ii
    return np.mean([v for i in ii for v in per[i][0]]) - np.mean([v for i in ii for v in per[i][1]])
per0, au0 = natural()
brp = np.random.default_rng(0); PIDX = [brp.integers(0, len(per0), len(per0)) for _ in range(NB)]
out["G"] = dict(n_paragraphs=len(nat), n_points=sum(len(x[0]) + len(x[1]) for x in per0), auroc=au0, separation=float(sepof(per0)),
                gate_pass=bool(au0 >= 0.90))
print("G", json.dumps(out["G"]), flush=True)

# ---------------- C
if SETS["C"]:
    W = 60; rng = random.Random(300); items = []
    for p in paras:
        if len(items) >= NI: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 5: continue
        k = rng.randrange(2, len(lines) - 1); words = lines[k].split(" ")
        js = [j for j in range(3, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
        if not js: continue
        cur = words[: js[0] + 1]; d = 0
        while len(cur) > 2 and d < 10: d += len(cur[0]) + 1; cur = cur[1:]
        if d < 10: continue
        ib = tok("\n".join(lines[:k]) + "\n" + " ".join(words[: js[0] + 1]), add_special_tokens=False)["input_ids"]
        is_ = tok("\n".join(lines[:k]) + "\n" + " ".join(cur), add_special_tokens=False)["input_ids"]
        if ib[-1] != is_[-1]: continue
        items.append((ib, is_))
    tot, cache = [], []
    for ib, is_ in items:
        m.patch = {}; lb = m.lp(ib)[-1]; ls = m.lp(is_, capture=True)[-1]
        cache.append((ib, lb, {l: m.cap[l][-1].clone() for l in range(m.L)})); tot.append(ls - lb)
    tot = np.array(tot)
    def cshare(S):
        e = []
        for ib, lb, zs in cache:
            m.patch = {}
            for l, h in S: m.patch.setdefault(l, []).append((h, len(ib) - 1, zs[l][h * m.dh:(h + 1) * m.dh]))
            e.append(m.lp(ib)[-1] - lb); m.patch = {}
        return np.array(e)
    e = cshare(SETS["C"]); n = len(tot); br = np.random.default_rng(0); idx = [br.integers(0, n, n) for _ in range(NB)]
    rsh = [float(cshare(r).mean() / tot.mean()) for r in RAND["C"]]
    out["C"] = dict(set=nm(SETS["C"]), n=n, total=float(tot.mean()), total_ci95=pct([tot[i].mean() for i in idx]),
                    share=float(e.mean() / tot.mean()), share_ci95=pct([e[i].mean() / tot[i].mean() for i in idx]),
                    random_share=rsh, random_mean=float(np.mean(rsh)))
    print("C", json.dumps(out["C"]), flush=True)

# ---------------- T
if SETS["T"]:
    res = {}
    for W in (40, 60, 80):
        its = wrapped_items(paras, W, NP, tok=tok)
        def tc(SS):
            X, y, g = [], [], []
            for k, (ids, recs, _) in enumerate(its):
                m.patch = setpatch(SS, "all") if SS else {}; lp = m.lp(ids); m.patch = {}
                for r in recs:
                    if r["remaining"] <= 30: X.append((r["remaining"], r["ntok_line"])); y.append(float(logodds(lp[r["i"]]))); g.append(k)
            return np.array(X, float), np.array(y), np.array(g)
        X0, y0, g = tc(()); X1, y1, _ = tc(SETS["T"])
        groups = [np.where(g == k)[0] for k in range(len(its))]; br = np.random.default_rng(0); b0, b1 = [], []
        for _ in range(NB):
            ii = np.concatenate([groups[j] for j in br.integers(0, len(its), len(its))])
            b0.append(token_coef(X0[ii], y0[ii])); b1.append(token_coef(X1[ii], y1[ii]))
        rnd = [token_coef(*tc(r)[:2]) for r in RAND["T"]]
        res[W] = dict(n_points=len(y0), n_paragraphs=len(its), intact=token_coef(X0, y0), intact_ci95=pct(b0),
                      ablated=token_coef(X1, y1), ablated_ci95=pct(b1), diff=token_coef(X1, y1) - token_coef(X0, y0),
                      diff_ci95=pct(np.array(b1) - np.array(b0)), random=rnd, random_mean=float(np.mean(rnd)))
        print("T", W, json.dumps(res[W]), flush=True)
    out["T"] = dict(set=nm(SETS["T"]), widths=res)

# ---------------- S and R
witems = width_items(paras, m, NI, seed=200) if (SETS["S"] or SETS["R"] or SETS["X"]) else []
E0 = None
for ROLE in ("S", "X"):
    if not SETS[ROLE]: continue
    def weff(S):
        e = []
        for ib, is_, nb, ns in witems:
            m.patch = setpatch(S, "nl", ib, ZN) if S else {}; lb = m.lp(ib)[-1]
            m.patch = setpatch(S, "nl", is_, ZN) if S else {}; ls = m.lp(is_)[-1]; m.patch = {}
            e.append(ls - lb)
        return np.array(e)
    E0 = weff(()) if E0 is None else E0; E1 = weff(SETS[ROLE]); n = len(E0); br = np.random.default_rng(0); idx = [br.integers(0, n, n) for _ in range(NB)]
    rsh = [float(1 - weff(r).mean() / E0.mean()) for r in RAND[ROLE]]
    perS, auS = natural(SETS[ROLE], "nl", ZN)
    keptb = [sepof(perS, ii) / sepof(per0, ii) for ii in PIDX]
    out[ROLE] = dict(set=nm(SETS[ROLE]), n=n, width_effect=float(E0.mean()), width_effect_ci95=pct([E0[i].mean() for i in idx]),
                    share=float(1 - E1.mean() / E0.mean()), share_ci95=pct([1 - E1[i].mean() / E0[i].mean() for i in idx]),
                    random_share=rsh, random_mean=float(np.mean(rsh)), nat_kept=float(sepof(perS) / sepof(per0)), nat_kept_ci95=pct(keptb), nat_auroc=auS)
    print(ROLE, json.dumps(out[ROLE]), flush=True)
if SETS["R"]:
    Lw = int(spec["Lw"]); cache = []
    for ib, is_, nb, ns in witems:
        m.patch, m.rpatch = {}, {}
        lb = m.lp(ib, capture=True)[-1]; zb = {l: m.cap[l][-1].clone() for l in range(m.L)}
        m.lp(is_, capture=True); cache.append((ib, nb, lb, zb, m.rcap[Lw][ns].clone()))
    def med(S):
        pw, fr = [], []
        for ib, nb, lb, zb, r in cache:
            m.rpatch = {Lw: [(nb, r)]}; m.patch = {}; pw.append(m.lp(ib)[-1] - lb)
            for l, h in S: m.patch.setdefault(l, []).append((h, len(ib) - 1, zb[l][h * m.dh:(h + 1) * m.dh]))
            fr.append(m.lp(ib)[-1] - lb); m.patch, m.rpatch = {}, {}
        return np.array(pw), np.array(fr)
    Pw, Fr = med(SETS["R"]); n = len(Pw); br = np.random.default_rng(0); idx = [br.integers(0, n, n) for _ in range(NB)]
    rsh = [float(1 - med(r)[1].mean() / Pw.mean()) for r in RAND["R"]]
    out["R"] = dict(set=nm(SETS["R"]), Lw=Lw, n=n, patch_effect=float(Pw.mean()), patch_effect_ci95=pct([Pw[i].mean() for i in idx]),
                    share=float(1 - Fr.mean() / Pw.mean()), share_ci95=pct([1 - Fr[i].mean() / Pw[i].mean() for i in idx]),
                    random_share=rsh, random_mean=float(np.mean(rsh)))
    print("R", json.dumps(out["R"]), flush=True)

# ---------------- verdicts (thresholds from the spec)
v = {}
if "C" in out:
    c = out["C"]; v["C"] = bool(c["total_ci95"][1] < 0 and c["share"] >= TH["C_share"] and c["share_ci95"][0] > 0 and c["random_mean"] < TH["C_random"])
if "T" in out:
    for W, r in out["T"]["widths"].items():
        v[f"T_w{W}"] = bool(r["ablated"] >= TH["T_coef"] and r["diff_ci95"][0] > 0 and r["random_mean"] < TH["T_random"])
    v["T"] = all(v[f"T_w{W}"] for W in (40, 60, 80) if W in TH["T_widths"])
if "S" in out:
    s = out["S"]; v["S"] = bool(s["width_effect_ci95"][0] > 0 and s["share"] >= TH["S_share"] and s["share_ci95"][0] > 0
                                and s["random_mean"] < TH["S_random"] and s["nat_kept"] >= TH["S_kept"])
if "X" in out:
    x = out["X"]; v["X"] = bool(x["width_effect_ci95"][0] > 0 and x["share"] >= TH["X_share"] and x["share_ci95"][0] > 0
                                and x["random_mean"] < TH["X_random"] and ("X_kept_max" not in TH or x["nat_kept"] <= TH["X_kept_max"]))
if "R" in out:
    r = out["R"]; v["R"] = bool(r["patch_effect_ci95"][0] > 0 and r["share"] >= TH["R_share"] and r["share_ci95"][0] > 0 and r["random_mean"] < TH["R_random"])
out["verdicts"] = v
print(json.dumps(v), flush=True)
save_json(out, f"xm_conf_{MODEL}_{PART}{'_smoke' if SMOKE else ''}.json")
