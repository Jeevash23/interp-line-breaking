"""D1-X Amendment 3: check the numbers quoted in the prose of REPORT-<model>.md against the saved
confirmation outputs (as xm_report_prose_check.py). Each claim is a (text that must appear in the
report, value recomputed from the output) pair; the script fails if the formatted value is not in the
claimed text or the text is not in the report. Usage: xm3_report_prose_check.py"""
import json, sys
from pathlib import Path
X = Path(__file__).resolve().parent.parent
def load(m, part=None): return json.load(open(X / "outputs" / f"xm_conf_{m}_{part or 'conf-' + m}.json"))
P = lambda x: f"{100 * x:.0f}%"; P1 = lambda x: f"{100 * x:.1f}%"; D2 = lambda x: f"{x:.2f}"; D3 = lambda x: f"{x:.3f}"
bad = 0
def check(m, text, val):
    global bad
    p = X / f"REPORT-{m}.md"
    if not p.exists(): print("skip", m, "(no report yet)"); return
    rep = p.read_text(); ok = text in rep and val in text
    if not ok: bad += 1
    print("ok " if ok else "BAD", m, repr(val), "in", repr(text[:90]))

m = "qwen3-17b"
if (X / "outputs" / f"xm_conf_{m}_conf-{m}.json").exists():
    d = load(m); T = d["T"]["widths"]; ab = [T[w]["ablated"] for w in ("40", "60", "80")]
    check(m, "carry 70% of the current line's count to the decision", P(d["C"]["share"]))
    check(m, "each token adds 0.13–0.25 to the newline log-odds", D2(min(ab))); check(m, "each token adds 0.13–0.25 to the newline log-odds", D2(max(ab)))
    check(m, "carry 44% of the previous line's effect", P(d["X"]["share"]))
    check(m, "the decision keeps 52% of its separation", P(d["X"]["nat_kept"]))
    check(m, "carry 78% of the stored length's effect at the decision", P(d["R"]["share"]))
    check(m, "AUROC 0.988 at width 60 on post-2025 text", D3(d["G"]["auroc"]))
    check(m, "carries 70% of the count effect", P(d["C"]["share"]))
    check(m, "the largest is 0.0%", P1(max(d["C"]["random_share"])))
    for w, v in zip(("40", "60", "80"), ("0.25", "0.18", "0.13")): check(m, "0.25, 0.18 and 0.13 log-odds each", D2(T[w]["ablated"]))
    check(m, "(at most 0.02)", D2(max(max(T[w]["random"]) for w in T)))
    check(m, "removes 44% of the previous line's effect", P(d["X"]["share"]))
    check(m, "(52% kept)", P(d["X"]["nat_kept"]))
    check(m, "removes 78% of its effect; random sets remove at most 0.7%", P(d["R"]["share"])); check(m, "removes 78% of its effect; random sets remove at most 0.7%", P1(max(d["R"]["random_share"])))
    pt = load("pythia-1b", "conf-pythia1b")["T"]["widths"]; pa = [pt[w]["ablated"] for w in ("40", "60", "80")]
    check(m, "on its confirmation part (0.27–0.46 there)", D2(min(pa))); check(m, "on its confirmation part (0.27–0.46 there)", D2(max(pa)))
    ratio = [T[w]["ablated"] / pt[w]["ablated"] for w in ("40", "60", "80")]
    print("   ratio to Pythia-1B:", [round(r, 2) for r in ratio], "(text says about half)"); bad += 0 if all(0.4 <= r <= 0.6 for r in ratio) else 1

m = "qwen3-06b"
if (X / "outputs" / f"xm_conf_{m}_conf-{m}.json").exists():
    d = load(m); T = d["T"]["widths"]; ab = [T[w]["ablated"] for w in ("40", "60", "80")]
    q = load("qwen3-17b"); qT = q["T"]["widths"]; qab = [qT[w]["ablated"] for w in ("40", "60", "80")]
    check(m, "carry 76% of the current line's count to the decision | 70% |", P(d["C"]["share"])); check(m, "carry 76% of the current line's count to the decision | 70% |", P(q["C"]["share"]))
    check(m, "each token adds 0.11–0.21 to the newline log-odds | 0.13–0.25 |", D2(min(ab))); check(m, "each token adds 0.11–0.21 to the newline log-odds | 0.13–0.25 |", D2(max(ab)))
    check(m, "each token adds 0.11–0.21 to the newline log-odds | 0.13–0.25 |", D2(min(qab))); check(m, "each token adds 0.11–0.21 to the newline log-odds | 0.13–0.25 |", D2(max(qab)))
    check(m, "carry 43% of the previous line's effect; the decision keeps 42% of its separation | 44%; 52% |", P(d["X"]["share"]))
    check(m, "carry 43% of the previous line's effect; the decision keeps 42% of its separation | 44%; 52% |", P(d["X"]["nat_kept"]))
    check(m, "carry 43% of the previous line's effect; the decision keeps 42% of its separation | 44%; 52% |", P(q["X"]["share"]))
    check(m, "carry 43% of the previous line's effect; the decision keeps 42% of its separation | 44%; 52% |", P(q["X"]["nat_kept"]))
    check(m, "carry 78% of the stored length's effect at the decision | 78% |", P(d["R"]["share"])); check(m, "carry 78% of the stored length's effect at the decision | 78% |", P(q["R"]["share"]))
    check(m, "AUROC 0.987 at width 60", D3(d["G"]["auroc"]))
    check(m, "carries 76% of the count effect; random sets carry at most 0.6%", P(d["C"]["share"])); check(m, "carries 76% of the count effect; random sets carry at most 0.6%", P1(max(d["C"]["random_share"])))
    for w in ("40", "60", "80"): check(m, "0.21, 0.19 and 0.11 log-odds each", D2(T[w]["ablated"]))
    for w in ("40", "60", "80"): check(m, "intact level (0.05, 0.01 and 0.01)", D2(T[w]["intact"]))
    check(m, "remove 43% of the previous line's effect, and the decision keeps 42%", P(d["X"]["share"])); check(m, "remove 43% of the previous line's effect, and the decision keeps 42%", P(d["X"]["nat_kept"]))
    check(m, "(42% kept against 52%)", P(d["X"]["nat_kept"])); check(m, "(42% kept against 52%)", P(q["X"]["nat_kept"]))
    check(m, "removes 78% of the stored length's effect after layer 6; random sets remove at most 0.7%", P(d["R"]["share"]))
    check(m, "removes 78% of the stored length's effect after layer 6; random sets remove at most 0.7%", P1(max(d["R"]["random_share"])))
    t = T["40"]; check(m, "at 0.05 (0.02, 0.08)", D2(t["intact"])); check(m, "at 0.05 (0.02, 0.08)", D2(t["intact_ci95"][0])); check(m, "at 0.05 (0.02, 0.08)", D2(t["intact_ci95"][1]))
    e = load(m, "explore-qwen3-06b"); check(m, "on this model's exploration part (0.07)", D2(e["T"]["widths"]["40"]["intact"]))
    ok_rule = d["verdicts"]["C"] and d["verdicts"]["X"] and d["verdicts"]["R"]
    print("   cross-model rule (C, X, R pass in 0.6B):", ok_rule); bad += 0 if ok_rule else 1

m = "gemma4-e2b"
if (X / "outputs" / f"xm_conf_{m}_conf-{m}.json").exists():
    d = load(m); T = d["T"]["widths"]; ab = [T[w]["ablated"] for w in ("40", "60", "80")]
    check(m, "carry 68% of the current line's count to the decision", P(d["C"]["share"]))
    check(m, "each token adds 0.31–0.39 to the newline log-odds", D2(min(ab))); check(m, "each token adds 0.31–0.39 to the newline log-odds", D2(max(ab)))
    check(m, "carry 44% of the previous line's effect; the decision keeps 70% of its separation", P(d["S"]["share"]))
    check(m, "carry 44% of the previous line's effect; the decision keeps 70% of its separation", P(d["S"]["nat_kept"]))
    check(m, "carry 50% of the stored length's effect at the decision", P(d["R"]["share"]))
    check(m, "AUROC 0.987 at width 60", D3(d["G"]["auroc"]))
    for v in [P(d["C"]["share"])] + [P(x) for x in d["C"]["share_ci95"]] + [P1(max(d["C"]["random_share"]))]:
        check(m, "carries 68% (62%, 75%) of the count effect. Random pairs from the same layers carry at most 5.2%", v)
    for w in ("40", "60", "80"): check(m, "0.39, 0.38 and 0.31 log-odds each", D2(T[w]["ablated"]))
    for w in ("40", "60", "80"): check(m, "about 0 (0.00, 0.03, 0.01)", D2(T[w]["intact"]))
    check(m, "Random layer-6 heads reach at most 0.07", D2(max(max(T[w]["random"]) for w in T)))
    pt = load("pythia-1b", "conf-pythia1b")["T"]["widths"]
    for w in ("40", "60", "80"): check(m, "(0.46, 0.38 and 0.27 there)", D2(pt[w]["ablated"]))
    check(m, "remove 44% of the previous line's effect and leave 70%", P(d["S"]["share"])); check(m, "remove 44% of the previous line's effect and leave 70%", P(d["S"]["nat_kept"]))
    check(m, "(73% kept there)", P(load("olmo1b", "conf-olmo1b")["S"]["nat_kept"]))
    check(m, "removes 50% of its effect; random pairs at most 2.1%", P(d["R"]["share"])); check(m, "removes 50% of its effect; random pairs at most 2.1%", P1(max(d["R"]["random_share"])))
    check(m, "as in Pythia-1B (50%)", P(load("pythia-1b", "conf-pythia1b")["R"]["share"]))
    rs = [load(k, "conf-" + c)["R"]["share"] for k, c in (("qwen05b", "qwen05b"), ("olmo1b", "olmo1b"), ("qwen3-17b", "qwen3-17b"), ("qwen3-06b", "qwen3-06b"))]
    check(m, "against 78–113% in the Qwen and OLMo models", f"{100 * min(rs):.0f}–"); check(m, "against 78–113% in the Qwen and OLMo models", P(max(rs)))
    e5 = json.load(open(X / "outputs" / "xm_e05_gemma4-e2b_train_a.json"))["resid_share_by_layer"]
    print("   e05 residual share by layer 5, 6, 12:", [round(e5[i], 2) for i in (5, 6, 12)], "(text: only from layer 6, all of it from layer 12)")
    bad += 0 if (abs(e5[5]) < 0.05 and e5[6] > 0.3 and e5[12] >= 1.0) else 1
    e2 = json.load(open(X / "outputs" / "xm_e02_gemma4-e2b_train.json"))
    late = [v for k, v in e2["readers_top"] if int(k[1:].split("H")[0]) >= 14][:5]
    print("   late readers:", [round(v, 3) for v in late], "(text: 5–8% each)"); bad += 0 if all(0.045 <= v < 0.085 for v in late) else 1
    late_r = e2["resid_by_layer"][14:]
    check(m, "changes the decision by the same 0.003 (0.2% of the effect)", D3(late_r[0])); check(m, "changes the decision by the same 0.003 (0.2% of the effect)", P1(late_r[0] / e2["total"]))
    print("   layers 14-34 spread:", max(late_r) - min(late_r), "(text: no change from layer to layer)"); bad += 0 if max(late_r) - min(late_r) < 1e-6 else 1

m = "smollm2-17b"
if (X / "outputs" / f"xm_conf_{m}_conf-{m}.json").exists():
    d = load(m); T = d["T"]["widths"]; ab = [T[w]["ablated"] for w in ("40", "60", "80")]
    check(m, "carry 73% of the current line's count to the decision", P(d["C"]["share"]))
    check(m, "each token adds 0.18–0.23 to the newline log-odds", D2(min(ab))); check(m, "each token adds 0.18–0.23 to the newline log-odds", D2(max(ab)))
    check(m, "carry 36% of the previous line's effect; the decision keeps 86% of its separation", P(d["S"]["share"]))
    check(m, "carry 36% of the previous line's effect; the decision keeps 86% of its separation", P(d["S"]["nat_kept"]))
    check(m, "carries 48% of the stored length's effect at the decision", P(d["R"]["share"]))
    check(m, "AUROC 0.986 at width 60", D3(d["G"]["auroc"]))
    for v in [P(d["C"]["share"])] + [P(x) for x in d["C"]["share_ci95"]] + [P1(max(d["C"]["random_share"]))]:
        check(m, "carries 73% (69%, 77%) of the count effect; random sets carry at most 0.3%", v)
    e5 = json.load(open(X / "outputs" / "xm_e05_smollm2-17b_train_a.json"))["resid_share_by_layer"]
    for v in (P(e5[5]), P(e5[6]), P(e5[7])): check(m, "carried 29% and 30% of the count after layers 5 and 6, and 93% after layer 7", v)
    for w in ("40", "60", "80"): check(m, "0.23, 0.22 and 0.18 log-odds each", D2(T[w]["ablated"]))
    for w in ("40", "60", "80"): check(m, "intact, 0.04, 0.00 and 0.01", D2(T[w]["intact"]))
    check(m, "(at most 0.04)", D2(max(max(T[w]["random"]) for w in T)))
    check(m, "removes 36% of the previous line's effect, and the decision keeps 86%", P(d["S"]["share"])); check(m, "removes 36% of the previous line's effect, and the decision keeps 86%", P(d["S"]["nat_kept"]))
    check(m, "as in Qwen2.5-0.5B (85%)", P(load("qwen05b", "conf-qwen05b")["S"]["nat_kept"]))
    check(m, "removes 48% of its effect; random single heads from layer 11 remove −1.8% to 0.0%", P(d["R"]["share"]))
    check(m, "removes 48% of its effect; random single heads from layer 11 remove −1.8% to 0.0%", P1(min(d["R"]["random_share"])).replace("-", "−"))
    check(m, "removes 48% of its effect; random single heads from layer 11 remove −1.8% to 0.0%", P1(max(d["R"]["random_share"])))
    p0 = json.load(open(X / "outputs" / "xm_e00_smollm2-17b_bf16gpu.json")); check(m, "(AUROC 0.987 on the 1,264 check points)", D3(p0["auroc"])); check(m, "(AUROC 0.987 on the 1,264 check points)", f"{p0['n']:,}")

print("\nPROSE CHECK:", "all claims match" if not bad else f"{bad} problems"); sys.exit(1 if bad else 0)
