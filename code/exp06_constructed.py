"""exp06: constructed inputs that separate candidate cues. Exploratory.

Each test item is a prompt that ends at a word-end decision point on line k (k >= 3),
with 3..14 characters remaining. We compare log P(newline) at the end under:

 A. width source
    orig      : lines wrapped greedily at W
    prevshort : line k-1 loses its last words (now >= 10 characters shorter than before);
                earlier lines unchanged. If the model takes the width from the previous
                line, P(newline) should rise as if ~that many more characters were used.
    allshort  : every earlier line is rewrapped at W-12 (control for "most recent lines").
 C. what is counted
    accent    : in earlier words of line k, 4 letters (a/e/o/u/i) get an acute accent.
                Same number of characters, 4 more bytes, odd tokens.
    upper     : the same 4 letters upper-cased instead. Same characters and bytes, odd tokens.
    If the model counts characters, neither should move P(newline) much. If it counts
    bytes, 'accent' should act like 4 more characters used and 'upper' should not.

Reference: the clean hazard curve (mean P(newline) by characters remaining, same W)
gives the change predicted for "d more characters used": log h(rem-d) - log h(rem).
"""
import sys, json, random, numpy as np, torch
from common import *

MNAME = sys.argv[1] if len(sys.argv) > 1 else "pythia-410m"
W = 60
tok, model = load(MNAME)
nl = newline_id(tok)

def lp_last(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        lg = model(torch.tensor([ids], device="cuda")).logits[0, -1].float()
    return torch.log_softmax(lg, -1)[nl].item()

# clean hazard at W from separate paragraphs
haz = {}
for p in train_paragraphs(seed=4)[:400]:
    lines = wrap(p, W)
    if lines is None or len(lines) < 5: continue
    ids, recs = word_end_positions(tok, lines, W)
    if len(ids) > 700: continue
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0].float(), -1)
    for r in recs: haz.setdefault(r["remaining"], []).append(lp[r["i"], nl].exp().item())
    if sum(len(v) for v in haz.values()) > 6000: break
H = {r: float(np.mean(v)) for r, v in haz.items() if len(v) >= 10}

ACC = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}
rng = random.Random(0)
rows = []
for p in train_paragraphs(seed=5):
    if len(rows) >= 600: break
    lines = wrap(p, W)
    if lines is None or len(lines) < 6: continue
    for k in range(3, len(lines) - 1):
        line = lines[k]
        words = line.split(" ")
        # decision points: after word j (j>=2), with 3..14 remaining; line content before is words[:j+1]
        for j in range(2, len(words) - 0):
            prefix = " ".join(words[: j + 1])
            used = len(prefix); rem = W - used
            if not (3 <= rem <= 14): continue
            last = words[j]
            if not last[-1:].isalpha(): continue
            label = int(j == len(words) - 1)
            before = lines[:k]
            base = "\n".join(before) + "\n" + prefix
            # A: previous line shortened by dropping last words until >= 10 chars shorter
            pw = before[-1].split(" "); short = before[-1]
            while len(pw) > 2 and len(before[-1]) - len(short) < 10:
                pw = pw[:-1]; short = " ".join(pw)
            dshort = len(before[-1]) - len(short)
            prevshort = "\n".join(before[:-1] + [short]) + "\n" + prefix
            # A control: all earlier text rewrapped at W-12
            ew = wrap(" ".join(before), W - 12)
            allshort = "\n".join(ew) + "\n" + prefix if ew else None
            # C: modify 4 vowels in earlier words of line k (not the last word)
            cand = [(wi, ci) for wi, w_ in enumerate(words[:j]) for ci, ch in enumerate(w_) if ch in ACC]
            if len(cand) < 4: continue
            pick = rng.sample(cand, 4)
            wa = [list(w_) for w_ in words[: j + 1]]; wu = [list(w_) for w_ in words[: j + 1]]
            for wi, ci in pick:
                wa[wi][ci] = ACC[wa[wi][ci]]; wu[wi][ci] = wu[wi][ci].upper()
            acc = "\n".join(before) + "\n" + " ".join("".join(x) for x in wa)
            upp = "\n".join(before) + "\n" + " ".join("".join(x) for x in wu)
            assert len(" ".join("".join(x) for x in wa)) == used
            r0 = lp_last(base)
            row = dict(rem=rem, label=label, dshort=dshort, base=r0, prevshort=lp_last(prevshort),
                       allshort=lp_last(allshort) if allshort else None, accent=lp_last(acc), upper=lp_last(upp),
                       ntok_base=len(tok(prefix)["input_ids"]), ntok_acc=len(tok(" ".join("".join(x) for x in wa))["input_ids"]),
                       ntok_upp=len(tok(" ".join("".join(x) for x in wu))["input_ids"]))
            rows.append(row)
            break  # one decision point per line
    # end lines

def pred(rem, d):
    if rem in H and (rem - d) in H: return np.log(H[rem - d]) - np.log(H[rem])
    return np.nan

out = {"n": len(rows), "hazard": H}
R = rows
d_prev = np.array([r["prevshort"] - r["base"] for r in R]); p_prev = np.array([pred(r["rem"], r["dshort"]) for r in R])
d_all = np.array([r["allshort"] - r["base"] for r in R if r["allshort"] is not None])
d_acc = np.array([r["accent"] - r["base"] for r in R]); d_upp = np.array([r["upper"] - r["base"] for r in R])
p4 = np.array([pred(r["rem"], 4) for r in R]); p12 = np.array([pred(r["rem"], 12) for r in R if r["allshort"] is not None])
ok = ~np.isnan(p_prev)
out["A_prevshort"] = dict(mean_change=float(d_prev.mean()), mean_pred_if_prevline=float(np.nanmean(p_prev)), mean_dshort=float(np.mean([r["dshort"] for r in R])),
                          corr_with_pred=float(np.corrcoef(d_prev[ok], p_prev[ok])[0, 1]))
out["A_allshort"] = dict(mean_change=float(d_all.mean()), mean_pred_if_width_minus12=float(np.nanmean(p12)))
out["C_accent"] = dict(mean_change=float(d_acc.mean()), median=float(np.median(d_acc)), mean_pred_if_bytes=float(np.nanmean(p4)),
                       extra_tokens=float(np.mean([r["ntok_acc"] - r["ntok_base"] for r in R])))
out["C_upper"] = dict(mean_change=float(d_upp.mean()), median=float(np.median(d_upp)),
                      extra_tokens=float(np.mean([r["ntok_upp"] - r["ntok_base"] for r in R])))
save_json(dict(summary=out, rows=rows), f"exp06_constructed_{MNAME}.json")
print(json.dumps(out | {"hazard": None}, indent=1))
