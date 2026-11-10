"""POST HOC (not confirmatory). Holm correction across every pass/fail part of rounds 1 and 2,
and the key results under resample ablation. Reads saved outputs only; changes no frozen verdict.

Each part becomes a one-sided bootstrap test (10,000 resamples, p = (1 + #null side) / (B + 1)):
 - parts whose frozen criterion is an interval bound use that bound as the null boundary
   (e.g. H1b: null coef <= 0.20; H3, H4a/b, H5, H6a: null difference/share on the wrong side of 0);
 - parts with only a point threshold use that threshold (H2 shares, H4c, H4d, R2);
 - "within a margin" parts are two one-sided tests, p = the larger (H1a, H1c, H6b).
Holm at family-wise alpha = 0.025, the one-sided level of the frozen 95% intervals.
Sources: outputs/posthoc_r12_{validation,test}.npz (re-run with mean ablation, as frozen) and the
item rows saved by conf02 (validation), c2_02 and c2_04 (test).
Run: python3 posthoc_holm.py [--check]   (--check: tables must appear verbatim in POSTHOC.md)
"""
import json, sys, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent.parent; O = R / "outputs"
NB = 10000; ALPHA = 0.025
P = lambda null_side: float((1 + np.sum(null_side)) / (NB + 1))
S = {s: dict(np.load(O / f"posthoc_r12_{s}.npz")) for s in ("validation", "test")}
PT = {s: json.loads((O / f"posthoc_r12_{s}.json").read_text())["points"] for s in ("validation", "test")}
FROZ = dict(r1=json.loads((O / "conf01_token_validation.json").read_text()), r2=json.loads((O / "c2_01_token_test.json").read_text()),
            c3=json.loads((O / "c2_03_newline_test.json").read_text()))

# reproduction check of mean-ablation re-runs against the frozen outputs
rep = []
for s, fz in (("validation", FROZ["r1"]), ("test", FROZ["r2"])):
    for W in ("40", "60", "80"):
        for fk, pk in (("intact", "intact"), ("abl_L5H7", "mean_all_L5H7"), ("abl_L5H3", "mean_all_L5H3"), ("abl_L5H11", "mean_all_L5H11"), ("abl_L5H14", "mean_all_L5H14")):
            rep.append(abs(fz["results"][W][fk]["token_coef"] - PT[s][f"tok_w{W}_{pk}"]))
c3 = FROZ["c3"]
rep += [abs(c3["H4_width"]["width_effect"] - PT["test"]["width_effect"]),
        abs(c3["H4_width"]["shares"]["COUNT4_nl"]["share"] - PT["test"]["share_mean_COUNT4_nl"]),
        abs(c3["H4_dissociation"]["COUNT4_nl"]["kept"] - PT["test"]["kept_mean_COUNT4_nl"])]
rep += [abs(c3["H6"][W]["L5H7_nl"]["prev_ntok"] - PT["test"][f"prevtok_w{W}_mean_nl_L5H7"]) for W in ("40", "60", "80")]
max_dev = float(max(rep))

def rows_boot(rows, f, seed=0):
    n = len(rows); br = np.random.default_rng(seed); idx = [br.integers(0, n, n) for _ in range(NB)]
    return f(np.arange(n)), np.array([f(i) for i in idx])

tests = []   # (round, part, p)
def tok_tests(s, rnd):
    b = S[s]
    for W in (40, 60, 80):
        i = b[f"tok_w{W}_intact"]
        tests.append((rnd, f"H1a, width {W}", max(P(i >= 0.10), P(i <= -0.10))))
        tests.append((rnd, f"H1b, width {W}", P(b[f"tok_w{W}_mean_all_L5H7"] <= 0.20)))
        for c in ("L5H3", "L5H11", "L5H14"):
            d = b[f"tok_w{W}_mean_all_{c}"] - i
            tests.append((rnd, f"H1c {c}, width {W}", max(P(d >= 0.10), P(d <= -0.10))))
def h2_tests(path, rnd):
    rows = json.loads((O / path).read_text())["rows"]
    E = np.array([r["eff"] for r in rows])
    _, be = rows_boot(rows, lambda i: E[i].mean())
    tests.append((rnd, "H2 positive control", P(be <= 0)))
    for g, t in (("L8H15", 0.25), ("readers3", 0.50)):
        F = np.array([r[g] for r in rows]); _, bs = rows_boot(rows, lambda i: 1 - F[i].mean() / E[i].mean())
        tests.append((rnd, f"H2 {g} >= {t:.2f}", P(bs <= t)))
    Fr = [np.array([r[f"random{q}"] for r in rows]) for q in range(10)]
    _, bm = rows_boot(rows, lambda i: np.mean([1 - F[i].mean() / E[i].mean() for F in Fr]))
    tests.append((rnd, "H2 random mean < 0.15", P(bm >= 0.15)))

tok_tests("validation", "1")
h2_tests("conf02_width_validation.json", "1")
tok_tests("test", "2 (R1)")
h2_tests("c2_02_width_test.json", "2 (R2)")
b = S["test"]
for W in (40, 60, 80):
    tests.append(("2", f"H3, width {W}", P(b[f"tok_w{W}_mean_all_L4H14"] - b[f"tok_w{W}_intact"] <= 0)))
tests += [("2", "H4 positive control", P(b["width_effect"] <= 0)), ("2", "H4a", P(b["share_mean_COUNT4_nl"] <= 0)),
          ("2", "H4b", P(b["share_mean_COUNT4_last"] <= 0)), ("2", "H4c", P(b["randmean_mean"] >= 0.15)),
          ("2", "H4d, newline-only kept >= 0.80", P(b["kept_mean_COUNT4_nl"] <= 0.80)), ("2", "H4d, all-position kept < 0.30", P(b["kept_mean_COUNT4_all"] >= 0.30))]
rows = json.loads((O / "c2_04_paths_test.json").read_text())["rows"]
_, be = rows_boot(rows, lambda i: np.mean([rows[j]["eff"] for j in i]))
_, b5a = rows_boot(rows, lambda i: np.mean([rows[j]["L8H15"]["V"] - rows[j]["L8H15"]["K"] for j in i]))
_, b5b = rows_boot(rows, lambda i: np.mean([rows[j]["L17H1"]["K"] - rows[j]["L17H1"]["V"] for j in i]))
tests += [("2", "H5 positive control", P(be <= 0)), ("2", "H5a", P(b5a <= 0)), ("2", "H5b", P(b5b <= 0))]
for W in (40, 60):
    d7 = b[f"prevtok_w{W}_mean_nl_L5H7"] - b[f"prevtok_w{W}_intact"]; d11 = b[f"prevtok_w{W}_mean_nl_L5H11"] - b[f"prevtok_w{W}_intact"]
    tests.append(("2", f"H6a, width {W}", P(d7 >= 0)))
    tests.append(("2", f"H6b, width {W}", max(P(d11 >= 0.02), P(d11 <= -0.02))))

m = len(tests); order = np.argsort([t[2] for t in tests]); adj = np.empty(m); run = 0.0
for k, j in enumerate(order):
    run = max(run, min(1.0, (m - k) * tests[j][2])); adj[j] = run
L = [f"Tests in the family: {m}. Holm at family-wise alpha {ALPHA} (one-sided). Largest difference between the mean-ablation re-run and the frozen outputs: {max_dev:.2e}.", "",
     "| Round | Part | One-sided p | Holm-adjusted p | Holds after Holm |", "|---|---|---|---|---|"]
for (rnd, part, p), a in zip(tests, adj):
    L.append(f"| {rnd} | {part} | {p:.4f} | {a:.4f} | {'yes' if a < ALPHA else 'no'} |")
n_hold = int(np.sum(adj < ALPHA)); L += ["", f"Parts holding after Holm: {n_hold} of {m}"]

# resample ablation, frozen criteria applied post hoc
ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
f3 = lambda x: f"{x:.3f}"; fci = lambda c: f"({c[0]:.3f}, {c[1]:.3f})"; yn = lambda x: "yes" if x else "no"
L += ["", "| Split | Measure | Mean ablation | Resample ablation | Frozen criterion met with resample |", "|---|---|---|---|---|"]
RS = {}
for s in ("validation", "test"):
    b, pt = S[s], PT[s]
    for W in (40, 60, 80):
        for c in ("L5H7", "L4H14"):
            mk, rk = f"tok_w{W}_mean_all_{c}", f"tok_w{W}_resample_all_{c}"
            if c == "L5H7": ok = pt[rk] >= 0.30 and ci(b[rk])[0] > 0.20
            else:
                d = b[rk] - b[f"tok_w{W}_intact"]; ok = pt[rk] - pt[f"tok_w{W}_intact"] >= 0.10 and ci(d)[0] > 0
            L.append(f"| {s} | token coefficient, {c} ablated, width {W} | {f3(pt[mk])} {fci(ci(b[mk]))} | {f3(pt[rk])} {fci(ci(b[rk]))} | {yn(ok)} |")
        ctl = max(abs(pt[f"tok_w{W}_resample_all_{c}"] - pt[f"tok_w{W}_intact"]) for c in ("L5H3", "L5H11", "L5H14"))
        L.append(f"| {s} | largest control change (L5H3, L5H11, L5H14), width {W} | {f3(max(abs(pt[f'tok_w{W}_mean_all_{c}'] - pt[f'tok_w{W}_intact']) for c in ('L5H3', 'L5H11', 'L5H14')))} | {f3(ctl)} | {yn(ctl < 0.10)} |")
    for c, nm in (("COUNT4_nl", "every newline"), ("COUNT4_last", "last newline")):
        rk = f"share_resample_{c}"
        L.append(f"| {s} | width effect removed, COUNT4 at {nm} | {f3(pt['share_mean_' + c])} {fci(ci(b['share_mean_' + c]))} | {f3(pt[rk])} {fci(ci(b[rk]))} | {yn(pt[rk] >= 0.25 and ci(b[rk])[0] > 0)} |")
    L.append(f"| {s} | width effect removed, random sets, mean | {f3(pt['randmean_mean'])} | {f3(pt['randmean_resample'])} | {yn(pt['randmean_resample'] < 0.15)} |")
    L.append(f"| {s} | separation kept, COUNT4 newline-only / every position | {f3(pt['kept_mean_COUNT4_nl'])} / {f3(pt['kept_mean_COUNT4_all'])} | {f3(pt['kept_resample_COUNT4_nl'])} / {f3(pt['kept_resample_COUNT4_all'])} | {yn(pt['kept_resample_COUNT4_nl'] >= 0.80 and pt['kept_resample_COUNT4_all'] < 0.30)} |")
    for W in (40, 60, 80):
        dm = b[f"prevtok_w{W}_mean_nl_L5H7"] - b[f"prevtok_w{W}_intact"]; dr = b[f"prevtok_w{W}_resample_nl_L5H7"] - b[f"prevtok_w{W}_intact"]
        pm = pt[f"prevtok_w{W}_mean_nl_L5H7"] - pt[f"prevtok_w{W}_intact"]; pr = pt[f"prevtok_w{W}_resample_nl_L5H7"] - pt[f"prevtok_w{W}_intact"]
        L.append(f"| {s} | previous-line token weight, L5H7 at newlines minus intact, width {W} | {f3(pm)} {fci(ci(dm))} | {f3(pr)} {fci(ci(dr))} | {yn(pr <= -0.03 and ci(dr)[1] < 0)} |")
txt = "\n".join(L)
(O / "posthoc_tables.md").write_text(txt + "\n")
(O / "posthoc_holm.json").write_text(json.dumps(dict(alpha=ALPHA, n_tests=m, max_reproduction_dev=max_dev,
    tests=[dict(round=r, part=p, p=pv, holm=float(a)) for (r, p, pv), a in zip(tests, adj)]), indent=1))
if "--check" in sys.argv:
    doc = (R / "POSTHOC.md").read_text(); missing = [l for l in L if l.strip() and l not in doc]
    print("missing lines:", missing if missing else "none"); sys.exit(1 if missing else 0)
print(txt)
