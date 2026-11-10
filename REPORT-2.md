# D1 report, round 2: confirmation on the test split

6 November 2026. **Confirmation** under `PROTOCOL-2.md`, which was frozen at 05:24 UTC after a train-split pilot (exp21). Its digests are in `FROZEN.md`. The text is the WikiText-2 **test** split, read here for the first time; it was the last unread WikiText-2 material. All four scripts ran once, with no crash and no re-run. The tables below are generated from the saved outputs by `code/report2_tables.py`, and `--check` verifies that they appear here verbatim. Round 1 (validation split) is in `REPORT.md`.

## Summary
| Test | What it claims | Verdict |
|---|---|---|
| R1 | Replication of H1: L5H7 removes a token bias, and the intact model ignores tokens | **fail**, on the same part as round 1 (R1a at width 40) |
| R2 | Replication of H2: three reader heads carry the width | **pass** |
| H3 | L4H14 is a second, weaker token corrector | **pass** |
| H4 | The counting heads, run at the newline token, store the length of the line just ended | **pass** |
| H5 | L8H15 takes the width from the newline's value, L17H1 from its key | **pass** |
| H6 | L5H7 at the newline also removes the token bias from the stored length | **pass** |

## R1 and H3: token correction
**R1 and H3.** Token coefficient (95% paragraph-bootstrap interval).

| Width | Points | Intact | L5H7 ablated | L5H3 ablated | L5H11 ablated | L5H14 ablated | R1a | R1b | R1c |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 6654 | 0.057 (0.014, 0.103) | 0.620 (0.549, 0.695) | 0.058 (0.015, 0.104) | 0.053 (0.011, 0.099) | 0.055 (0.012, 0.101) | fail | pass | pass |
| 60 | 4114 | 0.028 (-0.008, 0.065) | 0.612 (0.540, 0.685) | 0.032 (-0.004, 0.068) | 0.025 (-0.010, 0.061) | 0.028 (-0.009, 0.065) | pass | pass | pass |
| 80 | 2828 | 0.005 (-0.022, 0.031) | 0.525 (0.471, 0.587) | 0.008 (-0.020, 0.033) | 0.005 (-0.021, 0.030) | 0.006 (-0.022, 0.032) | pass | pass | pass |

R1 overall: fail; R1a at width 40: fail

| Width | L4H14 ablated | L4H14 minus intact (paired) | H3 | L4H14 + L5H7 ablated | Joint minus L5H7 alone (paired) |
|---|---|---|---|---|---|
| 40 | 0.205 (0.159, 0.256) | 0.148 (0.126, 0.171) | pass | 0.766 (0.684, 0.855) | 0.146 (0.122, 0.172) |
| 60 | 0.257 (0.221, 0.291) | 0.228 (0.203, 0.250) | pass | 0.682 (0.610, 0.757) | 0.070 (0.049, 0.093) |
| 80 | 0.224 (0.193, 0.258) | 0.218 (0.189, 0.248) | pass | 0.601 (0.546, 0.664) | 0.076 (0.056, 0.098) |

H3 overall: pass

- **R1 fails as frozen, exactly as in round 1.**
  - The two mechanism parts replicate at every width: ablating L5H7 gives each token on the line 0.53–0.62 log-odds (R1b), and no control head moves the coefficient (R1c).
  - The behavioural part fails again at width 40. The intact model keeps a token weight of 0.057, interval (0.014, 0.103), whose upper bound crosses the frozen 0.10 limit. At widths 60 and 80 it is within the limit.
  - Two held-out splits now agree. At narrow widths the intact model is not fully token-blind: it keeps a small residual token weight, about a tenth of the weight seen with L5H7 removed.
- **H3 passes.** Ablating L4H14 raises the token coefficient by 0.15–0.23 at every width, with paired intervals well above 0. Ablating both L4H14 and L5H7 adds 0.07–0.15 more than L5H7 alone. So L4H14 is a second, weaker corrector, not the source of the bias that exp19 had expected.

## R2: width readers
**R2.** Width effect (n = 150): 0.753 (0.656, 0.860)

| Frozen at the decision token | Mediation share (95% CI) |
|---|---|
| L8H15 | 0.452 (0.419, 0.485) |
| L8H15, L17H1, L6H2 | 0.742 (0.704, 0.782) |
| 10 random triples, mean | 0.004 |

R2 overall: pass

**R2 passes, as in round 1.** On the test split, L8H15 alone carries 45% of the width effect and the three readers together carry 74%, against 49% and 82% on validation. Random triples carry nothing.

## H4: the counting heads re-used at the newline
COUNT4 is L2H7, L4H15, L5H6 and L5H7: the heads the decision depends on (exp04) that also write the previous line's length at the newline (exp20). Each is replaced by its newline-position mean at newline tokens only.

The width effect here is the total change in log P(newline) when the previous line is at least 10 characters shorter, with the same ablation in both prompts. It is larger than the R2 effect, which counts only the part that passes through the newline token's state after layer 5.

**H4.** Width effect, intact (n = 150): 1.616 (1.456, 1.773)

| Newline-only ablation | Share of the width effect removed (95% CI) |
|---|---|
| COUNT4, every newline | 46.4% (41.9%, 50.8%) |
| COUNT4, last newline only | 61.3% (57.4%, 65.1%) |
| 10 random sets of 4 heads, mean | -1.2% |

| Natural text, width 60 (60 paragraphs) | Separation | AUROC | Kept |
|---|---|---|---|
| Intact | 3.639 | 0.911 | |
| COUNT4 at newlines only | 3.139 | 0.891 | 86.3% |
| COUNT4 at every position | 0.279 | 0.540 | 7.7% |

H4a pass, H4b pass, H4c pass, H4d pass; H4 overall: pass

**H4 passes on all four parts.**
- Removing the counting heads' output at newline tokens only takes away 46% of the width effect, and 61% when only the newline ending the shortened line is touched. Ten random sets of four heads take away nothing (−1.2% on average).
- On natural text the same newline-only ablation leaves 86% of the break/no-break separation. Ablating the same heads at every position leaves 8%.
- So the same four heads do two jobs:
  - at word ends they count the characters on the current line;
  - at the newline token they record the length of the line that has just ended.
  The decision needs the first job almost entirely, and the width needs the second.

## H5: two routes for the width
**H5.** Width effect, eager attention (n = 150): 0.753 (0.656, 0.860)

| Head | Key only | Value only |
|---|---|---|
| L8H15 | 0.054 (0.020, 0.088) | 0.271 (0.239, 0.304) |
| L17H1 | 0.120 (0.098, 0.145) | 0.018 (0.014, 0.023) |
| L8H11 | -0.055 (-0.072, -0.041) | 0.003 (-0.001, 0.006) |

L8H15 value minus key: 0.217 (0.166, 0.270); L17H1 key minus value: 0.102 (0.079, 0.126)

H5a pass, H5b pass; H5 overall: pass

**H5 passes.**
- L8H15 takes up the width mainly through the newline's value: value-only 0.27 against key-only 0.05. It copies the stored length.
- L17H1 takes it up mainly through the newline's key: key-only 0.12 against value-only 0.02. It compares by attention.
- Control head L8H11 has no value route and a small negative key effect, also seen in exploration (exp17).

## H6: the stored length is token-corrected
**H6.** Coefficient of the previous line's token count (95% paragraph-bootstrap interval).

| Width | Points | Intact | L5H7 at newlines | L5H11 at newlines | L5H7 minus intact (paired) | L5H11 minus intact | H6a | H6b |
|---|---|---|---|---|---|---|---|---|
| 40 | 6654 | -0.014 (-0.052, 0.018) | -0.064 (-0.104, -0.028) | -0.015 (-0.052, 0.017) | -0.050 (-0.060, -0.040) | -0.001 | pass | pass |
| 60 | 4114 | -0.032 (-0.065, -0.000) | -0.107 (-0.145, -0.070) | -0.032 (-0.065, -0.000) | -0.075 (-0.092, -0.058) | 0.000 | pass | pass |
| 80 | 2828 | -0.005 (-0.042, 0.030) | -0.073 (-0.112, -0.035) | -0.005 (-0.042, 0.031) | -0.067 (-0.078, -0.056) | 0.000 | not assessed | not assessed |

H6 overall: pass

**H6 passes at widths 40 and 60, and the unassessed width 80 points the same way.**
- With L5H7 ablated at newline tokens only, a previous line made of more tokens counts as longer: each extra token lowers the newline log-odds by a further 0.05–0.08, given the line's character length.
- The control head changes nothing.
- This is the re-use account's own prediction. The token correction L5H7 applies to the running count at word ends is applied in the same way to the stored length at the newline.
- The intact model also has a small negative previous-line token weight at width 60 (−0.032, upper bound −0.000). It was not part of any prediction.

## What the two rounds now support
Confirmed on held-out text, in Pythia-410M on WikiText-2 prose wrapped greedily:

1. **The counting heads (L2H7, L4H15, L5H6, L5H7).**
   - At word ends they count characters on the current line.
   - At the newline token, the same heads record the finished line's length (H4).
2. **The token correctors.** L5H7, with L4H14 as a weaker second, remove a per-token bias from the count (R1b, R1c, H3). L5H7 does this both for the running count and for the stored length (H6). The correction is not complete at width 40 in either split (R1a).
3. **The readers.** At the decision token, L8H15, L17H1 and L6H2 bring the stored length to the decision (R2). L8H15 copies it from the newline's value and L17H1 compares through its key (H5).

## Limits
- One model and one corpus, with greedy wrapping of plain prose.
- Mean-ablation and patching only; no training-dynamics data.
- Thresholds for H3–H6 were set from train-split exploration on the same model, at about half the explored size.
- The test split has now been used, so WikiText-2 has no unread text left for D1. A further confirmation would need new text or a new model.
- The tests were run as separate hypotheses, with no correction for multiple tests (thresholds fixed in advance).
