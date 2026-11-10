"""Every number the D1 paper reports, computed from the saved study outputs (read-only).

`load()` returns a flat dict: key -> float, or (point, lo, hi) for estimates with a 95% interval.
make_tables.py writes the tables from it; check_paper.py checks the tables and the text against it.
Round-4 intervals are recomputed with the frozen bootstrap of code/report4_tables.py (same seeds), and
check_paper.py also compares them with that script's saved table file.
Round-5 values come from the frozen script code/report5_tables.py itself: round5() runs it on links to the saved
outputs, writing its tables into a scratch folder, and reads its results from the script's variables;
check_paper.py requires the scratch table file to equal the saved one (outputs/report5_tables_conf.md).
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D1 = os.path.normpath(os.path.join(HERE, "..", ".."))
O = os.path.join(D1, "outputs")
XO = os.path.join(D1, "xmodel", "outputs")
J = lambda *p: json.load(open(os.path.join(*p)))
ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def token_round(V, rnd, f):
    d = J(O, f)
    for W, r in d["results"].items():
        for c, v in r.items():
            V[f"{rnd}.tok.{W}.{c}"] = (v["token_coef"], *v["ci95"])
            V[f"{rnd}.tok.{W}.{c}.n"] = v["n_points"]
        V[f"{rnd}.tok.{W}.ctlmax"] = max(abs(r[c]["token_coef"] - r["intact"]["token_coef"]) for c in ("abl_L5H3", "abl_L5H11", "abl_L5H14"))
        for k, p in d.get("paired", {}).get(W, {}).items():
            V[f"{rnd}.tok.{W}.{k}"] = (p["diff"], *p["ci95"])


def width_round(V, rnd, f):
    d = J(O, f)
    V[f"{rnd}.h2.n"] = d["n"]
    V[f"{rnd}.h2.width"] = (d["width_effect"], *d["width_effect_ci95"])
    for k in ("L8H15", "readers3"):
        V[f"{rnd}.h2.{k}"] = (d["shares"][k]["share"], *d["shares"][k]["ci95"])
    V[f"{rnd}.h2.random"] = d["random_mean_share"]


def newline_round(V, rnd, f):
    d = J(O, f); h = d["H4_width"]
    V[f"{rnd}.h4.n"] = h["n"]
    V[f"{rnd}.h4.width"] = (h["width_effect"], *h["width_effect_ci95"])
    for k in ("COUNT4_nl", "COUNT4_last"):
        V[f"{rnd}.h4.{k}"] = (h["shares"][k]["share"], *h["shares"][k]["ci95"])
    V[f"{rnd}.h4.random"] = h["random_mean_share"]
    dd = d["H4_dissociation"]
    V[f"{rnd}.h4.sep_intact"] = dd["intact"]["separation"]
    for k in ("COUNT4_nl", "COUNT4_all"):
        V[f"{rnd}.h4.kept_{k}"] = dd[k]["kept"]; V[f"{rnd}.h4.auroc_{k}"] = dd[k]["auroc"]
    for W, r in d["H6"].items():
        V[f"{rnd}.h6.{W}.intact"] = (r["intact"]["prev_ntok"], *r["intact"]["ci95"])
        V[f"{rnd}.h6.{W}.L5H7"] = (r["L5H7_nl_minus_intact"]["diff"], *r["L5H7_nl_minus_intact"]["ci95"])
        V[f"{rnd}.h6.{W}.L5H11"] = (r["L5H11_nl_minus_intact"]["diff"], *r["L5H11_nl_minus_intact"]["ci95"])


def paths_round(V, rnd, f):
    s = J(O, f)["summary"]
    V[f"{rnd}.h5.width"] = (s["width_effect"]["mean"], *s["width_effect"]["ci95"])
    for h in ("L8H15", "L17H1", "L8H11"):
        for kv in ("K", "V"):
            V[f"{rnd}.h5.{h}.{kv}"] = (s[h][kv]["mean"], *s[h][kv]["ci95"])
    V[f"{rnd}.h5.L8H15_VmK"] = (s["L8H15_V_minus_K"]["mean"], *s["L8H15_V_minus_K"]["ci95"])
    V[f"{rnd}.h5.L17H1_KmV"] = (s["L17H1_K_minus_V"]["mean"], *s["L17H1_K_minus_V"]["ci95"])


def round4(V):
    import torch
    NB = 2000
    sp = "conf4-pythia410m"
    d1 = J(O, f"p4_01_decision_{sp}.json"); S = d1["summary"]; rows = d1["rows"]
    pt = torch.load(os.path.join(O, f"p4_01_decision_{sp}_means.pt"))
    setn = [s for s in S["mediation_sets"]["C"] if s != "random_mlp3_mean"]
    n = len(rows); rng = np.random.default_rng(0); IDX = [rng.integers(0, n, n) for _ in range(NB)]
    E = {c: np.array([r[c] for r in rows]) for c in ("C", "Wd")}
    F = {c: np.array([r[f"frozen_{c}"] for r in rows]) for c in ("C", "Wd")}
    share = lambda c, s, i: 1 - F[c][i, setn.index(s)].mean() / E[c][i].mean()
    rsh = lambda c, i: np.mean([share(c, f"random_mlp3_{q}", i) for q in range(10)])
    full = np.arange(n)
    V["r4.d.n"] = n
    V["r4.d.C"] = float(E["C"].mean()); V["r4.d.W"] = float(E["Wd"].mean()); V["r4.d.CW"] = S["eff"]["CW"]; V["r4.d.inter"] = S["interaction"]
    V["r4.d1"] = (share("C", "MLP14_16_18", full), *ci(np.array([share("C", "MLP14_16_18", i) for i in IDX])))
    V["r4.d1.rand"] = rsh("C", full)
    V["r4.d1.diff"] = (share("C", "MLP14_16_18", full) - rsh("C", full), *ci(np.array([share("C", "MLP14_16_18", i) - rsh("C", i) for i in IDX])))
    for c in ("C", "Wd"):
        V[f"r4.d2.{c}"] = (share(c, "MLP21_23", full), *ci(np.array([share(c, "MLP21_23", i) for i in IDX])))
        for s in ("layers6-11", "layers12-17", "layers18-23", "all_heads", "all_mlps", "readers3", "L17H1", "MLP14_16_18", "MLP21_23", "random_mlp3_mean"):
            V[f"r4.med.{c}.{s}"] = S["mediation_sets"][c][s]
    dR = pt["dR"]
    def cosL(i, k):
        a = dR["C"][i, k].mean(0); b = dR["Wd"][i, k].mean(0)
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    for k, lab in enumerate(range(5, 24)):
        V[f"r4.cos.{lab}"] = S["cos_layer"][f"after L{lab}"]
    c6, c12 = cosL(torch.as_tensor(full), 1), cosL(torch.as_tensor(full), 7)
    V["r4.d3a"] = (c12 - c6, *ci(np.array([cosL(torch.as_tensor(i), 7) - cosL(torch.as_tensor(i), 1) for i in IDX])))
    names = d1["names"]; heads = [i for i, x in enumerate(names) if "H" in x]
    for c in ("C", "Wd"):
        dla = np.array(d1["dla"][c]); o = sorted(heads, key=lambda i: -dla[i])[:2]
        V[f"r4.d4.{c}.top"] = names[o[0]]; V[f"r4.d4.{c}.topv"] = float(dla[o[0]])
        V[f"r4.d4.{c}.next"] = names[o[1]]; V[f"r4.d4.{c}.nextv"] = float(dla[o[1]])
        for comp in ("L17H1", "MLP23", "MLP21", "MLP16", "MLP18", "MLP14"):
            V[f"r4.dla.{c}.{comp}"] = float(dla[names.index(comp)])
    # S
    raw = J(O, f"p4_02_sufficiency_{sp}.json")["raw"]
    NR = (len(raw["sep"]) - 1) // 3 - 2
    idx_of = lambda scope, k: 1 + {"DEC": 0, "EARLY": 1, "BOTH": 2}[scope] * (2 + NR) + k
    npar = max(max(e["psc"]) for e in raw["sep"]) + 1
    sep_of = lambda e: np.mean(e["br"]) - np.mean(e["nb"])
    def auc_of(e):
        sc = np.array(e["sc"]); lab = np.array(e["lab"]); pos = sc[lab == 1]; neg = sc[lab == 0]
        r = np.concatenate([pos, neg]).argsort().argsort() + 1.0
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    rngp = np.random.default_rng(0); PIDX = [rngp.integers(0, npar, npar) for _ in range(NB)]
    def sep_boot(e):
        br, nb = np.array(e["br"]), np.array(e["nb"]); pb, pn = np.array(e["pbr"]), np.array(e["pnb"])
        sb, cb = np.bincount(pb, br, npar), np.bincount(pb, minlength=npar); sn, cn = np.bincount(pn, nb, npar), np.bincount(pn, minlength=npar)
        return np.array([sb[i].sum() / cb[i].sum() - sn[i].sum() / cn[i].sum() for i in PIDX])
    W0 = np.array(raw["width"][0]); ni = len(W0); rngi = np.random.default_rng(0); IIDX = [rngi.integers(0, ni, ni) for _ in range(NB)]
    kept_w = lambda k: (np.array(raw["width"][k]).mean() / W0.mean(), np.array([np.array(raw["width"][k])[i].mean() / W0[i].mean() for i in IIDX]))
    sep0 = sep_of(raw["sep"][0]); V["r4.s.npar"] = npar; V["r4.s.nitems"] = ni
    V["r4.s.sep0"] = sep0; V["r4.s.auc0"] = auc_of(raw["sep"][0]); V["r4.s.width0"] = float(W0.mean())
    for scope in ("DEC", "EARLY", "BOTH"):
        for k, nm in ((0, "mean"), (1, "resample")):
            e = raw["sep"][idx_of(scope, k)]
            V[f"r4.s.{scope}.{nm}.sep"] = sep_of(e) / sep0; V[f"r4.s.{scope}.{nm}.auc"] = auc_of(e); V[f"r4.s.{scope}.{nm}.width"] = kept_w(idx_of(scope, k))[0]
        rs = [sep_of(raw["sep"][idx_of(scope, 2 + q)]) / sep0 for q in range(NR)]; rw = [kept_w(idx_of(scope, 2 + q))[0] for q in range(NR)]
        V[f"r4.s.{scope}.rand.sepmin"], V[f"r4.s.{scope}.rand.sepmax"] = min(rs), max(rs)
        V[f"r4.s.{scope}.rand.widthmin"], V[f"r4.s.{scope}.rand.widthmax"] = min(rw), max(rw)
    kw, bkw = kept_w(idx_of("DEC", 0)); V["r4.s1b"] = (kw, *ci(bkw))
    bks = sep_boot(raw["sep"][idx_of("BOTH", 0)]) / sep_boot(raw["sep"][0]); V["r4.s2"] = (V["r4.s.BOTH.mean.sep"], *ci(bks))
    # G
    g = J(O, f"p4_03_geometry_{sp}.json")
    for W in ("40", "60"):
        for l in ("5", "6"):
            r = g["split_half"][W][l]["real"]; s = g["split_half"][W][l]["shuffled"]
            V[f"r4.g.{W}.{l}.var1"] = r["var_share_B"][0]; V[f"r4.g.{W}.{l}.lin1"] = r["r2_linear_by_pc"][0]
            V[f"r4.g.{W}.{l}.lin2"] = r["r2_linear_by_pc"][1]; V[f"r4.g.{W}.{l}.spl2"] = r["r2_spline_by_pc"][1]
            V[f"r4.g.{W}.{l}.rel2"] = r["reliability_by_pc"][1]; V[f"r4.g.{W}.{l}.shuf"] = s["r2_linear"]
        for h in ("L6H2", "L8H15", "L17H1", "L5H6", "L5H7", "L8H11"):
            a = g["attention"][W][h]; V[f"r4.g.{W}.{h}.peak"] = a["peak_at_remaining"]; V[f"r4.g.{W}.{h}.mono"] = a["monotone_r2"]
        V[f"r4.g.{W}.npoints"] = g[f"n_points_{W}"]
    # X
    for m, key in (("gpt2", "gpt2"), ("pythia-410m", "p410")):
        x = J(O, f"p4_04_roles_{m}_{sp}.json")["summary"]
        V[f"r4.x.{key}.probe"] = max(x["probe_used"]); V[f"r4.x.{key}.probe_layer"] = int(np.argmax(x["probe_used"]))
        V[f"r4.x.{key}.probe_ntok"] = max(x["probe_ntok"])
        V[f"r4.x.{key}.count"] = max(v[0] for v in x["count_shift_by_layer"])
        V[f"r4.x.{key}.count_by_layer"] = [v[0] for v in x["count_shift_by_layer"]]
        V[f"r4.x.{key}.width"] = x["width_behaviour"][0]
        V[f"r4.x.{key}.dil"] = max([h["share"] for h in x["dilution_heads"] if h["r_used"] <= -0.5], default=0.0)
        V[f"r4.x.{key}.dilhead"] = max([h for h in x["dilution_heads"] if h["r_used"] <= -0.5], key=lambda h: h["share"])["head"]
        V[f"r4.x.{key}.patch"] = [v[0] for v in x["width_patch_by_layer"]]
        V[f"r4.x.{key}.auc40"] = x["behaviour"]["40"]["auroc"]; V[f"r4.x.{key}.auc60"] = x["behaviour"]["60"]["auroc"]
        V[f"r4.x.{key}.tok40"] = x["behaviour"]["40"]["token_coef"]; V[f"r4.x.{key}.tok60"] = x["behaviour"]["60"]["token_coef"]
        V[f"r4.x.{key}.nitems"] = x["n_items"]
    V["r4.x.gpt2.patch8max"] = max(abs(v) for v in V["r4.x.gpt2.patch"][8:12])
    V["r4.x.p410.patch8"] = V["r4.x.p410.patch"][8]


def exploratory(V):
    e = J(O, "exp01_summary.json")
    for m in ("pythia-410m", "pythia-1b", "gpt2"):
        for W, r in e[m]["by_width"].items():
            for k in ("break_acc", "false_break", "auroc_model", "auroc_remaining_only"):
                V[f"exp01.{m}.{W}.{k}"] = r[k]
    c = J(O, "exp02_cues.json")["pythia-410m"]
    V["exp02"] = c
    for s in ("1000", "2000", "4000", "8000", "16000", "48000", "final"):
        d = J(O, f"exp26_checkpoint_{s}.json"); i = d["intact"]
        for k in ("auroc", "sep", "token_coef", "probe_used_L5", "L5H6_share", "L5H6_r_used"):
            V[f"ck.{s}.{k}"] = i[k]
        for k in ("counters_sep_kept", "corrector_token_coef_change", "width_effect", "width_share_removed_COUNT4_nl",
                  "width_share_removed_readers3", "width_share_removed_MLP14_16_18", "width_share_removed_MLP21_23"):
            V[f"ck.{s}.{k}"] = d[k]
    h = J(O, "posthoc_holm.json")
    V["ph.ntests"] = h["n_tests"]; V["ph.alpha"] = h["alpha"]; V["ph.maxdev"] = h["max_reproduction_dev"]
    V["ph.hold"] = sum(t["holm"] < h["alpha"] for t in h["tests"])
    V["ph.tests"] = h["tests"]


def xmodel(V):
    for m, part in (("pythia-1b", "conf-pythia1b"), ("qwen05b", "conf-qwen05b"), ("olmo1b", "conf-olmo1b")):
        d = J(XO, f"xm_conf_{m}_{part}.json")
        V[f"x.{m}.auroc"] = d["G"]["auroc"]
        for t in ("C", "S", "R"):
            V[f"x.{m}.{t}"] = (d[t]["share"], *d[t]["share_ci95"]); V[f"x.{m}.{t}.rand"] = d[t]["random_mean"]
            V[f"x.{m}.{t}.set"] = d[t]["set"]
        V[f"x.{m}.S.kept"] = (d["S"]["nat_kept"], *d["S"]["nat_kept_ci95"])
        if "X" in d:
            V[f"x.{m}.X"] = (d["X"]["share"], *d["X"]["share_ci95"]); V[f"x.{m}.X.kept"] = (d["X"]["nat_kept"], *d["X"]["nat_kept_ci95"])
            V[f"x.{m}.X.set"] = d["X"]["set"]
        V[f"x.{m}.T.set"] = d["T"]["set"]
        for W, r in d["T"]["widths"].items():
            V[f"x.{m}.T.{W}"] = (r["ablated"], *r["ablated_ci95"]); V[f"x.{m}.T.{W}.rand"] = r["random_mean"]
        V[f"x.{m}.verdicts"] = d["verdicts"]
    o = J(XO, "xm_c2_overshoot_olmo1b_validation.json")
    V["x.ov.n"] = o["n"]; V["x.ov.share"] = (o["A_share"]["share"], *o["A_share"]["ci95"])
    V["x.ov.L5H11"] = (o["P2"]["dFr_L5H11"], *o["P2"]["ci95"]); V["x.ov.L5H11.randmean"] = float(np.mean(o["P2"]["random_dFr"]))
    V["x.ov.withL5H11"] = (o["P3"]["share"], *o["P3"]["ci95"]); V["x.ov.inject"] = (o["P4"]["inject_L5H11"], *o["P4"]["ci95"])
    V["x.ov.mean"] = (o["D_mean_ablation"]["share"], *o["D_mean_ablation"]["ci95"]); V["x.ov.resample"] = (o["D_resample_ablation"]["share"], *o["D_resample_ablation"]["ci95"])
    V["x.ov.value"] = (o["F"]["value"]["frac"], *o["F"]["value"]["ci95"]); V["x.ov.key"] = (o["F"]["key"]["frac"], *o["F"]["key"]["ci95"])
    V["x.ov.verdicts"] = o["verdicts"]


def xmodel3(V):
    """Qwen3-1.7B and Qwen3-0.6B (PROTOCOL-X3, confirmatory; head sets chosen in Qwen3-1.7B, used unchanged in 0.6B)."""
    for m, part in (("qwen3-17b", "conf-qwen3-17b"), ("qwen3-06b", "conf-qwen3-06b"),
                    ("gemma4-e2b", "conf-gemma4-e2b"), ("smollm2-17b", "conf-smollm2-17b")):
        d = J(XO, f"xm_conf_{m}_{part}.json")
        V[f"x.{m}.auroc"] = d["G"]["auroc"]; V[f"x.{m}.npara"] = d["G"]["n_paragraphs"]
        sx = "X" if "X" in d else "S"; V[f"x.{m}.sx"] = sx
        for t in ("C", sx, "R"):
            V[f"x.{m}.{t}"] = (d[t]["share"], *d[t]["share_ci95"]); V[f"x.{m}.{t}.rand"] = d[t]["random_mean"]
            V[f"x.{m}.{t}.set"] = d[t]["set"]; V[f"x.{m}.{t}.n"] = d[t]["n"]
        V[f"x.{m}.{sx}.kept"] = (d[sx]["nat_kept"], *d[sx]["nat_kept_ci95"])
        V[f"x.{m}.T.set"] = d["T"]["set"]
        for W, r in d["T"]["widths"].items():
            V[f"x.{m}.T.{W}"] = (r["ablated"], *r["ablated_ci95"]); V[f"x.{m}.T.{W}.rand"] = r["random_mean"]
            V[f"x.{m}.T.{W}.diff"] = (r["diff"], *r["diff_ci95"]); V[f"x.{m}.T.{W}.intact"] = (r["intact"], *r["intact_ci95"])
        V[f"x.{m}.verdicts"] = d["verdicts"]; V[f"x.{m}.spec"] = d["spec"]
    fs = J(os.path.join(D1, "xmodel", "data", "newtext3"), "fetch_summary.json")
    V["x3.n_articles"] = fs["n_articles"]; V["x3.parts"] = fs["parts"]


XM7 = (("pythia-1b", "conf-pythia1b"), ("qwen05b", "conf-qwen05b"), ("olmo1b", "conf-olmo1b"), ("qwen3-17b", "conf-qwen3-17b"),
       ("qwen3-06b", "conf-qwen3-06b"), ("gemma4-e2b", "conf-gemma4-e2b"), ("smollm2-17b", "conf-smollm2-17b"))
def xmodel_all(V):
    """All seven cross-model runs (PROTOCOL-X1, X3, X3b): test counts, ranges, role overlaps (from the frozen sets),
    and the token correctors' newline attention (exploratory, WikiText-2 train scans)."""
    ntest = npass = ngate = 0; C = []; R = []; rmax = []; rmean = []; auc = []; ov = {}; dil = []
    hs = lambda k, s: set(s.get(k, "").split(",")) - {""}
    for m, part in XM7:
        d = J(XO, f"xm_conf_{m}_{part}.json"); v = d["verdicts"]
        roles = [k for k in v if not k.startswith("T_w")]; ntest += len(roles); npass += sum(bool(v[k]) for k in roles)
        ngate += bool(d["G"]["gate_pass"]); auc.append(d["G"]["auroc"]); C.append(d["C"]["share"]); R.append(d["R"]["share"])
        for t in ("C", "S", "X", "R"):
            if t in d: rmax.append(max(d[t]["random_share"])); rmean.append(d[t]["random_mean"])
        s = d["spec"]["sets"]; c, tt, sx, r = hs("C", s), hs("T", s), hs("S", s) | hs("X", s), hs("R", s)
        ov[m] = {"CT": c & tt, "CS": c & sx, "CR": c & r, "CTS": c & tt & sx, "nC": len(c)}
        e = J(XO, f"xm_e01_{m}_train.json"); cr = e["attn"]["corr_used"]
        for h in s["T"].split(","):
            l, hh = int(h[1:].split("H")[0]), int(h.split("H")[1]); dil.append((m, h, cr[l][hh]))
    V["xa.ntest"], V["xa.npass"], V["xa.ngate"], V["xa.nmodel"] = ntest, npass, ngate, len(XM7)
    V["xa.auroc"] = (min(auc), max(auc)); V["xa.C"] = (min(C), max(C)); V["xa.R"] = (min(R), max(R))
    V["xa.randmax"] = max(rmax); V["xa.randmean"] = max(rmean); V["xa.ov"] = ov; V["xa.dil"] = dil
    # SmolLM2-1.7B float32 check (post hoc, PRECISION-smollm2-17b-posthoc.md): decision log-odds only
    c = J(XO, "xm3_e00_compare_smollm2-17b.json"); assert (c["a"], c["b"]) == ("fp32cpu", "bf16gpu")
    V["xp.smol.n"] = c["n"]; V["xp.smol.auroc"] = tuple(c["auroc"]); V["xp.smol.sep"] = tuple(c["separation"]); V["xp.smol.r"] = c["r"]


R5_TEXT = {}
def round5(V):
    import runpy, sys, tempfile
    tmp = tempfile.mkdtemp()
    for f in os.listdir(O): os.symlink(os.path.join(O, f), os.path.join(tmp, f))
    for f in ("report5_tables_conf.md", "report5_verdicts_conf.json"): os.remove(os.path.join(tmp, f))
    argv, env = sys.argv, os.environ.get("P5_OUT")
    sys.argv = ["report5_tables.py", "conf"]; os.environ["P5_OUT"] = tmp
    try:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            g = runpy.run_path(os.path.join(D1, "code", "report5_tables.py"))
    finally:
        sys.argv = argv
        if env is None: os.environ.pop("P5_OUT", None)
        else: os.environ["P5_OUT"] = env
    R5_TEXT["table"] = open(os.path.join(tmp, "report5_tables_conf.md")).read()
    V["r5.verdicts"] = g["V"]
    pct = g["pct"]
    nm = {g["ENG"]: "en", g["CODE"]: "code", g["DE"]: "de"}
    for (s_, w), t in g["TT"].items():
        k = f"r5.t.{nm[s_]}.{w}"
        V[k + ".intact"] = (t["intact"], *pct(t["intact_b"])); V[k + ".abl"] = (t["abl"], *pct(t["abl_b"]))
        V[k + ".rise"] = (t["abl"] - t["intact"], *pct(t["abl_b"] - t["intact_b"])); V[k + ".ntok"] = t["ntok"]; V[k + ".cpt"] = t["cpt"]
        V[k + ".att6"] = (t["att6"], *pct(t["att6_b"])); V[k + ".n"] = t["n"]
    V["r5.t1"] = (g["d1"], *pct(g["d1b"])); V["r5.t2"] = g["rho"]; V["r5.t3"] = (g["d3"], *pct(g["d3b"])); V["r5.t4"] = (g["d4"], *pct(g["d4b"]))
    for src, M in g["M"].items():
        for k in ("mlp", "mln", "top2", "top20"): V[f"r5.m.{src}.{k}"] = (M[k][0], *M[k][1])
        V[f"r5.m.{src}.E"] = M["E"]
    V["r5.m.n"] = g["n"]
    for (cond, isnl), x in g["NAT"].items():
        k = f"r5.m.nat.{cond}.{'nl' if isnl else 'other'}"
        V[k + ".n"] = x[0]; V[k + ".dlogp"] = (x[1], *x[2]); V[k + ".dent"] = (x[3], *x[4])
    for kk, v in g["MJ"]["corr"].items(): V[f"r5.m.corr.{kk}"] = v
    V["r5.m.ncorr"] = g["MJ"]["n_corr_points"]
    SJ = g["SJ"]; V["r5.s.npara"] = g["NPA"]; V["r5.s.nitems"] = len(g["W0"])
    V["r5.s.intact_sep"] = SJ["intact_sep"]; V["r5.s.intact_auroc"] = SJ["intact_auroc"]; V["r5.s.intact_width"] = SJ["intact_width"]
    for k, x in g["SS"].items():
        a = k.replace("|", ".")
        V[f"r5.s.{a}.fid"], V[f"r5.s.{a}.sep"], V[f"r5.s.{a}.width"] = x; V[f"r5.s.{a}.auroc"] = SJ[k]["auroc"]
    for k, b in g["SB"].items():
        a = k.replace("|", ".")
        for i, q in enumerate(("fid", "sep", "width")): V[f"r5.s.{a}.{q}.ci"] = tuple(pct(b[:, i]))
    for mode in ("mean", "resample"):
        r = np.array(g["rnd"][mode])
        for i, q in enumerate(("fid", "sep", "width")): V[f"r5.s.rand.{mode}.{q}"] = (float(r[:, i].min()), float(r[:, i].max()))
    V["r5.s.dsep"] = (g["dsep"], *pct(g["dsepb"]))
    for (s_, w) in g["GC"]:
        r = J(O, f"p5_04_general_{s_}_w{w}.json"); t = list(r["T"]["widths"].values())[0]; k = f"r5.g.{nm[s_]}.{w}"
        V[k + ".auroc"] = r["G"]["auroc"]; V[k + ".sep"] = r["G"]["separation"]
        V[k + ".C"] = (r["C"]["share"], *r["C"]["share_ci95"]); V[k + ".C.n"] = r["C"]["n"]; V[k + ".C.rand"] = r["C"]["random_mean"]
        V[k + ".T"] = (t["diff"], *t["diff_ci95"]); V[k + ".T.rand"] = t["random_mean"] - t["intact"]
        V[k + ".S"] = (r["S"]["share"], *r["S"]["share_ci95"]); V[k + ".S.n"] = r["S"]["n"]; V[k + ".S.rand"] = r["S"]["random_mean"]
        V[k + ".W"] = (r["S"]["width_effect"], *r["S"]["width_effect_ci95"])
        V[k + ".R"] = (r["R"]["share"], *r["R"]["share_ci95"]); V[k + ".R.n"] = r["R"]["n"]; V[k + ".R.rand"] = r["R"]["random_mean"]
        V[k + ".holds"] = g["GV"][f"{g['NAME'][s_]} {w}"]
    V["r5.g.pred_R"] = g["PRED_R"]; V["r5.g.pred_S"] = g["PRED_S"]; V["r5.g.match"] = sum(g["wid"]); V["r5.g.cnt"] = sum(g["cnt"])
    # exploration (round 5)
    e = J(O, "exp27_suff_search.json"); V["e27.final"] = e["curve"][-1]; V["e27.after2"] = e["curve"][-3]
    b = J(O, "exp27b_fidelity.json"); V["e27b.circuit"] = b["circuit"]["mean"]["fid"]
    u = J(O, "exp29b_unembed.json"); V["e29b.nl_null"] = u["nl_null_share"]; V["e29b.med_null"] = u["token_null_share_median"]
    for kk in ("MLP23.736", "MLP21.1410"):
        V[f"e29b.{kk}.nl_logit"] = u["neurons"][kk]["nl_logit"]; V[f"e29b.{kk}.sd"] = u["neurons"][kk]["logit_sd"]
        V[f"e29b.{kk}.rank"] = u["neurons"][kk]["nl_logit_rank_from_bottom"]
    x = J(O, "exp29_damping.json"); V["e29.conc5"] = x["concentration"]["5"]
    for t_ in x["top_neurons"][:2]: V[f"e29.{t_['neuron']}.norm_pct"] = t_["norm_pct"]


def load():
    V = {}
    token_round(V, "r1", "conf01_token_validation.json"); width_round(V, "r1", "conf02_width_validation.json")
    token_round(V, "r2", "c2_01_token_test.json"); width_round(V, "r2", "c2_02_width_test.json")
    newline_round(V, "r2", "c2_03_newline_test.json"); paths_round(V, "r2", "c2_04_paths_test.json")
    token_round(V, "r3", "c3_01_token_conf-pythia410m.json"); width_round(V, "r3", "c3_02_width_conf-pythia410m.json")
    newline_round(V, "r3", "c3_03_newline_conf-pythia410m.json"); paths_round(V, "r3", "c3_04_paths_conf-pythia410m.json")
    round4(V); exploratory(V); xmodel(V); xmodel3(V); xmodel_all(V); round5(V)
    return V


if __name__ == "__main__":
    V = load()
    for k, v in V.items():
        if not isinstance(v, (list, dict)): print(k, v)
