# D1-X report: Qwen2.5-0.5B

6 November 2026. **Exploration** under `CHARTER.md` (frozen 06:33 UTC) used the WikiText-2 train split and then the new-text exploration part (`NOTEBOOK.md`). **Confirmation** under `PROTOCOL-X1.md` (frozen 11:38 UTC; digests in `FROZEN.md`) used `part-conf-qwen05b.jsonl`: 269 paragraphs from English Wikipedia articles created in 2025–26, read here for the first time. The script `code/xm_conf.py` ran once (17:21–11:59 UTC), with no crash and no re-run. The tables below are generated from `outputs/xm_conf_qwen05b_conf-qwen05b.json` by `code/xm_report_tables.py`, and `--check` verifies that they appear here verbatim.

Qwen's tokeniser fuses sentence punctuation with a following newline, so decision points are word ends whose last character is a letter, and the line-start marker is any token that contains a newline (`CHARTER.md`). Qwen has 14 query heads per layer sharing 2 key/value heads; all tests here act on query heads' outputs.

## Summary
Qwen2.5-0.5B, from a different family, tokeniser and training set, has heads in all four roles, and every frozen test passed on the held-out new text.
- The arrangement is close to Pythia-410M's: the same early heads that count at word ends also store the finished line's length at the newline.
- As in Pythia-1B, removing those dilution heads lets the token count into the decision.
- Its readers carry almost all of the stored length's effect.

| Role | Heads | Result on the confirmation part |
|---|---|---|
| Counters | L2H1, L3H1, L3H2, L6H1 | carry 59% of the current line's count to the decision |
| Token correctors | L1H8, L1H9, L2H1, L3H1 | ablated, each token adds 0.13–0.21 to the newline log-odds |
| Stored length | L2H1, L3H1, L3H2 | at newline tokens only, carry 37% of the previous line's effect; the decision keeps 85% of its separation |
| Readers | L6H4, L4H8, L6H6 | carry 96% of the stored length's effect at the decision |

## Confirmation results

| Behaviour, width 60 | Value |
|---|---|
| Paragraphs | 60 |
| Decision points | 1354 |
| AUROC | 0.984 |
| Separation | 2.948 |
| Gate (AUROC ≥ 0.90) | pass |

| Test | Set | n | Effect (95% CI) | Share (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| C, counters | L2H1,L3H1,L3H2,L6H1 | 150 | -3.098 (-3.328, -2.855) | 59.1% (55.3%, 63.2%) | -0.0% (0.1%) | pass |
| S, stored length | L2H1,L3H1,L3H2 | 150 | 1.073 (0.903, 1.240) | 37.1% (30.8%, 42.7%) | 1.2% (9.4%) | pass |
| R, readers (Lw 3) | L6H4,L4H8,L6H6 | 150 | 0.669 (0.528, 0.806) | 96.4% (88.6%, 107.2%) | -0.1% (0.0%) | pass |

| Newline-only ablation | Natural separation kept (95% CI) | AUROC |
|---|---|---|
| S: L2H1,L3H1,L3H2 | 85.1% (79.4%, 90.7%) | 0.979 |

T, token correctors: L1H8,L1H9,L2H1,L3H1 ablated at every position.

| Width | Points | Intact (95% CI) | Ablated (95% CI) | Ablated minus intact (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| 40 | 5323 | 0.035 (0.007, 0.063) | 0.160 (0.125, 0.196) | 0.124 (0.085, 0.165) | 0.027 (0.047) | pass |
| 60 | 3213 | 0.025 (-0.006, 0.052) | 0.208 (0.168, 0.259) | 0.183 (0.133, 0.242) | 0.018 (0.037) | pass |
| 80 | 2209 | 0.010 (-0.011, 0.031) | 0.127 (0.095, 0.169) | 0.117 (0.075, 0.167) | 0.003 (0.012) | pass |

Verdicts: C pass, T pass, S pass, R pass

## Reading
- **Behaviour.** AUROC 0.984 at width 60 on post-2025 text (WikiText-2 train: 0.984).
- **C, counters: pass.** Patching four heads in layers 2–6 at the decision token carries 59% of the count effect. Random sets carry nothing (largest 0.1%).
- **T, token correctors: pass at all three widths.** Ablating L1H8, L1H9, L2H1 and L3H1 gives tokens on the line 0.16, 0.21 and 0.13 log-odds each at widths 40, 60 and 80; random sets ≤ 0.05. The effect is about a third of Pythia-410M's L5H7 and Pythia-1B's set. Two of these heads (L2H1, L3H1) are also counters.
- **S, stored length: pass.** The counters L2H1, L3H1 and L3H2, ablated at newline tokens only, remove 37% of the previous line's effect and leave 85% of the decision's separation. This is 410M's re-use pattern (H4: the same heads count at word ends and record the finished line at the newline).
- **R, readers: pass.** With the stored length patched into the newline after layer 3, freezing L6H4, L4H8 and L6H6 at the decision token removes 96% of its effect; random sets nothing. In exploration, L6H4 and L6H6 also carried part of the current count (e05). So Qwen's readers take in both the count and the stored width at the decision.
- **Not predicted, noted.** The intact token weight at width 40 is small but above 0, at 0.035 (0.007, 0.063), as in Pythia-410M.

## Limits
- One model checkpoint, Wikipedia prose and greedy wrapping only; letter-ending decision points only. Breaks after punctuation, where Qwen predicts punctuation and newline as one token, are not studied.
- Mean-ablation and activation patching of query-head outputs; key and value routes are shared within groups of 7 heads and were not tested.
- Head sets were chosen on WikiText-2 train and checked once on the new-text exploration part before the freeze. Thresholds are about half the explored sizes.
- The five tests were frozen as separate predictions, with no correction for multiple tests. All passed with intervals far from their thresholds.
