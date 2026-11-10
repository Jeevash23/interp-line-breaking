"""Print a compact summary of an xm_conf output (helper, no GPU)."""
import json, sys, numpy as np
d = json.load(open(sys.argv[1])); r2 = lambda x: round(x, 3)
print(d["model"], d["part"], "G", {k: r2(v) if isinstance(v, float) else v for k, v in d["G"].items()})
if "C" in d: c = d["C"]; print(" C", c["set"], "n", c["n"], "total", r2(c["total"]), "share", r2(c["share"]), [r2(x) for x in c["share_ci95"]], "rand mean", r2(c["random_mean"]), "max", r2(max(c["random_share"])))
if "T" in d:
    for W, t in d["T"]["widths"].items(): print(" T", W, "pts", t["n_points"], "intact", r2(t["intact"]), "abl", r2(t["ablated"]), [r2(x) for x in t["ablated_ci95"]], "diff", [r2(x) for x in t["diff_ci95"]], "rand mean", r2(t["random_mean"]), "max", r2(max(t["random"])))
for k in ("S", "X"):
    if k in d: s = d[k]; print(" ", k, s["set"], "n", s["n"], "eff", r2(s["width_effect"]), [r2(x) for x in s["width_effect_ci95"]], "share", r2(s["share"]), [r2(x) for x in s["share_ci95"]], "rand mean", r2(s["random_mean"]), "max", r2(max(s["random_share"])), "kept", r2(s["nat_kept"]), [r2(x) for x in s["nat_kept_ci95"]])
if "R" in d: s = d["R"]; print(" R", s["set"], "n", s["n"], "patch", r2(s["patch_effect"]), [r2(x) for x in s["patch_effect_ci95"]], "share", r2(s["share"]), [r2(x) for x in s["share_ci95"]], "rand mean", r2(s["random_mean"]), "max", r2(max(s["random_share"])))
print(" verdicts (as computed from the spec in the output)", d["verdicts"])
