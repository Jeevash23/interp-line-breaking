"""Check the numbers quoted in the prose of REPORT-<model>.md against the saved confirmation outputs.
Each claim is a (text that must appear in the report, value recomputed from the output) pair; the script
fails if the formatted value does not appear in the claimed text or the text is missing."""
import json, sys
from pathlib import Path
X = Path(__file__).resolve().parent.parent
def load(m, k): return json.load(open(X / "outputs" / f"xm_conf_{m}_conf-{k}.json"))
P = lambda x: f"{100 * x:.0f}%"; D2 = lambda x: f"{x:.2f}"; D3 = lambda x: f"{x:.3f}"
bad = 0
def check(m, text, val):
    global bad
    rep = (X / f"REPORT-{m}.md").read_text()
    ok = text in rep and val in text
    if not ok: bad += 1
    print("ok " if ok else "BAD", m, repr(val), "in", repr(text[:90]))

d = load("pythia-1b", "pythia1b"); T = d["T"]["widths"]
check("pythia-1b", "carry 65% of the current line's count", P(d["C"]["share"]))
check("pythia-1b", "moves the newline log-probability 65% of the way", P(d["C"]["share"]))
check("pythia-1b", "move it 2% on average", P(d["C"]["random_mean"]))
check("pythia-1b", "0.46, 0.38 and 0.27 log-odds per token", ", ".join(D2(T[w]["ablated"]) for w in ("40", "60")) if isinstance(next(iter(T)), str) else ", ".join(D2(T[w]["ablated"]) for w in (40, 60)))
check("pythia-1b", "adds 0.27–0.46 to the newline log-odds", D2(T["80" if "80" in T else 80]["ablated"]))
check("pythia-1b", "carry 55% of the previous line's effect; the decision keeps 93%", P(d["S"]["share"]))
check("pythia-1b", "carry 55% of the previous line's effect; the decision keeps 93%", P(d["S"]["nat_kept"]))
check("pythia-1b", "removes 50% of its effect, against 1% for random sets", P(d["R"]["share"]))
check("pythia-1b", "removes 50% of its effect, against 1% for random sets", P(d["R"]["random_mean"]))
check("pythia-1b", "small positive weight at width 60, 0.074 (0.038, 0.105)", D3(T["60" if "60" in T else 60]["intact"]))
check("pythia-1b", "AUROC 0.984 at width 60", D3(d["G"]["auroc"]))

d = load("qwen05b", "qwen05b"); T = d["T"]["widths"]; g = lambda w: T[str(w)] if str(w) in T else T[w]
check("qwen05b", "carries 59% of the count effect", P(d["C"]["share"]))
check("qwen05b", "0.16, 0.21 and 0.13 log-odds each", ", ".join(D2(g(w)["ablated"]) for w in (40, 60)) + " and " + D2(g(80)["ablated"]))
check("qwen05b", "remove 37% of the previous line's effect and leave 85%", P(d["S"]["share"]))
check("qwen05b", "remove 37% of the previous line's effect and leave 85%", P(d["S"]["nat_kept"]))
check("qwen05b", "removes 96% of its effect", P(d["R"]["share"]))
check("qwen05b", "is small but above 0, at 0.035 (0.007, 0.063)", D3(g(40)["intact"]))
check("qwen05b", "AUROC 0.984 at width 60", D3(d["G"]["auroc"]))

d = load("olmo1b", "olmo1b"); T = d["T"]["widths"]; g = lambda w: T[str(w)] if str(w) in T else T[w]
check("olmo1b", "carries 74% of the count effect; random sets 1% on average", P(d["C"]["share"]))
check("olmo1b", "carries 74% of the count effect; random sets 1% on average", P(d["C"]["random_mean"]))
check("olmo1b", "removes 113% of the patched effect", P(d["R"]["share"]))
check("olmo1b", "0.36, 0.31 and 0.29 log-odds each", ", ".join(D2(g(w)["ablated"]) for w in (40, 60)) + " and " + D2(g(80)["ablated"]))
check("olmo1b", "remove 37% of the previous line's effect, and the decision keeps 73%", P(d["S"]["share"]))
check("olmo1b", "remove 37% of the previous line's effect, and the decision keeps 73%", P(d["S"]["nat_kept"]))
check("olmo1b", "removes 76% of the previous line's effect and costs the decision 43% of its separation (57% kept)", P(d["X"]["share"]))
check("olmo1b", "removes 76% of the previous line's effect and costs the decision 43% of its separation (57% kept)", P(d["X"]["nat_kept"]))
check("olmo1b", "removes 76% of the previous line's effect and costs the decision 43% of its separation (57% kept)", P(1 - d["X"]["nat_kept"]))
check("olmo1b", "AUROC 0.988 at width 60", D3(d["G"]["auroc"]))
print("\nCHECK:", "all prose numbers match" if not bad else f"{bad} mismatches")
sys.exit(1 if bad else 0)
