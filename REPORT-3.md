# D1 report, round 3: confirmation on new text

6 November 2026. **Confirmation** under `PROTOCOL-3.md`, frozen at 10:41 UTC with its scripts and the new-text split (digests in `FROZEN.md`). The text is Pythia-410M's confirmation part of English Wikipedia articles created between January 2025 and September 2026 (`data/newtext/part-conf-pythia410m.jsonl`: 264 paragraphs from 146 articles; source, rule and licence in `data/newtext/MANIFEST.md`). Pythia was trained on text from before 2021, so it cannot have seen these articles. The text was read for results for the first time here. All four scripts ran once, between 16:11 and 10:50 UTC, with no crash and no re-run. The tables are generated from the saved outputs by `code/report3_tables.py`, and `--check` verifies that they appear here verbatim. Rounds 1 and 2 (WikiText-2 validation and test) are in `REPORT.md` and `REPORT-2.md`; a post hoc multiple-test and resample-ablation check of those rounds is in `POSTHOC.md`.

The scripts are copies of the round-2 scripts, so the table columns keep round 2's labels: R1a, R1b and R1c are the protocol's H1a, H1b and H1c, and "R2" is the width-reader test.

## Summary

| Prediction (`PROTOCOL-3.md`) | Result |
|---|---|
| T1, token correction by L5H7 (H1b and H1c at all widths, H1a at widths 60 and 80, width-40 residual above 0) | pass |
| Original H1, all parts at all widths under the 0.10 limit (reported for continuity) | fail, at width 40 only, as expected |
| R2, width readers | pass |
| H3, L4H14 as a second corrector | pass |
| H4, the counting heads re-used at the newline | pass |
| H5, two routes for the width | pass |
| H6, the stored length is token-corrected | pass |

Every prediction made for round 3 holds on text the model has not seen.

## T1 and H3: token correction
**R1 and H3.** Token coefficient (95% paragraph-bootstrap interval).

| Width | Points | Intact | L5H7 ablated | L5H3 ablated | L5H11 ablated | L5H14 ablated | R1a | R1b | R1c |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 5718 | 0.087 (0.046, 0.128) | 0.591 (0.522, 0.657) | 0.089 (0.049, 0.131) | 0.082 (0.042, 0.122) | 0.084 (0.043, 0.125) | fail | pass | pass |
| 60 | 3453 | 0.055 (0.024, 0.083) | 0.587 (0.539, 0.643) | 0.057 (0.026, 0.086) | 0.051 (0.021, 0.079) | 0.053 (0.022, 0.082) | pass | pass | pass |
| 80 | 2401 | 0.021 (-0.011, 0.052) | 0.474 (0.418, 0.536) | 0.021 (-0.010, 0.052) | 0.019 (-0.012, 0.050) | 0.019 (-0.012, 0.051) | pass | pass | pass |

Original H1 (all parts, all widths, frozen 0.10 limit): fail; H1a at width 40 under the 0.10 limit: fail
Round-3 prediction for width 40, intact lower bound above 0: pass
T1 (round-3 token-correction claim: H1b and H1c at all widths, H1a at widths 60 and 80, width-40 residual above 0): pass

| Width | L4H14 ablated | L4H14 minus intact (paired) | H3 | L4H14 + L5H7 ablated | Joint minus L5H7 alone (paired) |
|---|---|---|---|---|---|
| 40 | 0.198 (0.149, 0.252) | 0.111 (0.086, 0.135) | pass | 0.708 (0.633, 0.786) | 0.117 (0.094, 0.143) |
| 60 | 0.234 (0.199, 0.271) | 0.179 (0.154, 0.203) | pass | 0.658 (0.602, 0.723) | 0.071 (0.052, 0.092) |
| 80 | 0.217 (0.177, 0.261) | 0.197 (0.167, 0.227) | pass | 0.544 (0.485, 0.614) | 0.071 (0.054, 0.090) |

H3 overall: pass

- **T1 passes.**
  - Ablating L5H7 gives each token on the line 0.47–0.59 log-odds at every width, with lower bounds of 0.418 or more (H1b).
  - No control head moves the coefficient by more than 0.01 (H1c).
  - At widths 60 and 80 the intact interval lies inside (−0.10, +0.10) (H1a).
  - At width 40 the intact model keeps a token weight of 0.087, interval (0.046, 0.128). This is the residual predicted from rounds 1 and 2, whose width-40 values were 0.074 and 0.057.
- **The original H1 fails at width 40 for the third time**, because the upper bound crosses the 0.10 limit. This was expected and is why round 3 predicted the residual instead.
- Not predicted: at width 60 the intact interval (0.024, 0.083) also lies above 0 on this text. On the two WikiText splits the width-60 interval reached 0. The residual token weight therefore shrinks with width but does not vanish at width 60 here.
- **H3 passes.** Ablating L4H14 raises the token coefficient by 0.11–0.20, with paired intervals above 0 at every width. Ablating L4H14 and L5H7 together adds 0.07–0.12 more than L5H7 alone, as in round 2.

## R2: width readers
**R2.** Width effect (n = 150): 0.667 (0.565, 0.770)

| Frozen at the decision token | Mediation share (95% CI) |
|---|---|
| L8H15 | 0.431 (0.394, 0.469) |
| L8H15, L17H1, L6H2 | 0.750 (0.709, 0.790) |
| 10 random triples, mean | 0.005 |

R2 overall: pass

**R2 passes for the third time.** L8H15 alone carries 43% of the width effect and the three readers together carry 75%; on the WikiText splits the figures were 49% and 82% (validation) and 45% and 74% (test). Random triples carry 0.5% on average.

## H4: the counting heads re-used at the newline
COUNT4 is L2H7, L4H15, L5H6 and L5H7. Each is replaced by its newline-position mean at newline tokens only. As in round 2, the width effect here is the total change in log P(newline), which is larger than the R2 effect.

**H4.** Width effect, intact (n = 150): 1.431 (1.267, 1.601)

| Newline-only ablation | Share of the width effect removed (95% CI) |
|---|---|
| COUNT4, every newline | 45.8% (40.9%, 50.5%) |
| COUNT4, last newline only | 58.3% (54.1%, 62.7%) |
| 10 random sets of 4 heads, mean | -1.5% |

| Natural text, width 60 (60 paragraphs) | Separation | AUROC | Kept |
|---|---|---|---|
| Intact | 3.084 | 0.876 | |
| COUNT4 at newlines only | 2.580 | 0.856 | 83.7% |
| COUNT4 at every position | 0.076 | 0.513 | 2.5% |

H4a pass, H4b pass, H4c pass, H4d pass; H4 overall: pass

**H4 passes on all four parts.**
- The newline-only ablation removes 46% of the width effect, and 58% when only the newline ending the shortened line is touched (46% and 61% in round 2). Random sets of four heads remove nothing (−1.5% on average).
- On natural text the newline-only ablation keeps 84% of the separation, and ablating the same heads at every position keeps 2.5%. The 84% is closer to the frozen 80% limit than round 2's 86%.

## H5: two routes for the width
**H5.** Width effect, eager attention (n = 150): 0.667 (0.565, 0.770)

| Head | Key only | Value only |
|---|---|---|
| L8H15 | 0.037 (-0.000, 0.075) | 0.294 (0.259, 0.331) |
| L17H1 | 0.105 (0.083, 0.131) | 0.020 (0.015, 0.028) |
| L8H11 | -0.042 (-0.050, -0.035) | -0.002 (-0.005, 0.002) |

L8H15 value minus key: 0.257 (0.202, 0.315); L17H1 key minus value: 0.085 (0.063, 0.110)

H5a pass, H5b pass; H5 overall: pass

**H5 passes.** L8H15 takes up the width through the newline's value (0.29, against 0.04 through the key), and L17H1 through its key (0.11, against 0.02 through the value). Control head L8H11 again shows only a small negative key effect.

## H6: the stored length is token-corrected
**H6.** Coefficient of the previous line's token count (95% paragraph-bootstrap interval).

| Width | Points | Intact | L5H7 at newlines | L5H11 at newlines | L5H7 minus intact (paired) | L5H11 minus intact | H6a | H6b |
|---|---|---|---|---|---|---|---|---|
| 40 | 5718 | -0.013 (-0.045, 0.018) | -0.057 (-0.090, -0.026) | -0.012 (-0.043, 0.019) | -0.044 (-0.055, -0.032) | 0.001 | pass | pass |
| 60 | 3453 | -0.020 (-0.055, 0.018) | -0.089 (-0.121, -0.052) | -0.019 (-0.054, 0.019) | -0.069 (-0.087, -0.051) | 0.001 | pass | pass |
| 80 | 2401 | 0.009 (-0.018, 0.038) | -0.062 (-0.089, -0.033) | 0.009 (-0.018, 0.038) | -0.071 (-0.086, -0.057) | -0.000 | not assessed | not assessed |

H6 overall: pass

**H6 passes at widths 40 and 60.** With L5H7 ablated at newline tokens only, each extra token on the previous line lowers the newline log-odds by a further 0.04–0.07, given the line's character length. Width 80, reported without a verdict, shows the same (−0.071). The control head changes nothing. Round 2 noted a small negative intact weight at width 60 (−0.032); on this text the intact weight at width 60 is −0.020 with an interval that includes 0, so that side observation does not recur.

## What the three rounds support
In Pythia-410M, on greedily wrapped prose, and now on text from after its training:

1. **Counters.** L2H7, L4H15, L5H6 and L5H7 count the characters on the current line at word ends. At the newline token the same heads record the length of the line just ended (H4, rounds 2 and 3).
2. **Token correctors.** L5H7, with L4H14 as a weaker second, remove a per-token bias from the count (H1b, H1c, H3). L5H7 applies the same correction to the stored length (H6). The correction is incomplete at narrow widths: the intact model keeps a small token weight at width 40 in all three splits.
3. **Readers.** At the decision token L8H15, L17H1 and L6H2 bring the stored length to the decision (R2). L8H15 copies it from the newline's value and L17H1 compares through its key (H5).

## Limits
- One model, and greedy wrapping of plain prose at three widths.
- Mean ablation and patching only. `POSTHOC.md` repeats the key round-1 and round-2 results with resample ablation; round 3 was not repeated that way.
- The new text is recent Wikipedia, so it is new to the model but similar in style to its training data. A page created in 2025–26 can still contain older text that was moved without a summary saying so.
- Each prediction has its own fixed threshold, with no correction for multiple tests.
- Whether other models have the same roles is the cross-model study's question (`xmodel/CHARTER.md`), run as a separate part of the study.
