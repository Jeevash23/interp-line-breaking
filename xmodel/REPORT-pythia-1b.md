# D1-X report: Pythia-1B

6 November 2026. **Exploration** under `CHARTER.md` (frozen 06:33 UTC) used the WikiText-2 train split and then the new-text exploration part (`NOTEBOOK.md`). **Confirmation** under `PROTOCOL-X1.md` (frozen 11:38 UTC; digests in `FROZEN.md`) used `part-conf-pythia1b.jsonl`: 274 paragraphs from English Wikipedia articles created in 2025–26, read here for the first time. The script `code/xm_conf.py` ran once (17:08–11:51 UTC), with no crash and no re-run. The tables below are generated from `outputs/xm_conf_pythia-1b_conf-pythia1b.json` by `code/xm_report_tables.py`, and `--check` verifies that they appear here verbatim.

## Summary
Pythia-1B has heads in all four roles found in Pythia-410M, and every frozen test passed on the held-out new text. The arrangement differs from 410M in one respect. In 410M, the heads that count by line-start dilution (L5H6) and the head that removes the per-token bias (L5H7) are different heads. In Pythia-1B, ablating three early dilution heads (L0H0, L1H1, L1H6) is what lets the token count into the decision, and two of them also carry the count.

| Role | Heads | Result on the confirmation part |
|---|---|---|
| Counters | L1H1, L1H6, L2H5, L3H0, L3H5, L6H3 | carry 65% of the current line's count to the decision |
| Token correctors | L0H0, L1H1, L1H6 | ablated, each token on the line adds 0.27–0.46 to the newline log-odds |
| Stored length | L3H5, L6H3, L9H4, L10H1, L1H1, L2H6 | at newline tokens only, carry 55% of the previous line's effect; the decision keeps 93% of its separation |
| Readers | L12H6, L9H0, L7H3 | carry 50% of the stored length's effect at the decision |

## Confirmation results

| Behaviour, width 60 | Value |
|---|---|
| Paragraphs | 60 |
| Decision points | 1253 |
| AUROC | 0.984 |
| Separation | 3.092 |
| Gate (AUROC ≥ 0.90) | pass |

| Test | Set | n | Effect (95% CI) | Share (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| C, counters | L1H1,L1H6,L2H5,L3H0,L3H5,L6H3 | 150 | -3.431 (-3.650, -3.207) | 65.3% (61.9%, 68.8%) | 2.3% (8.1%) | pass |
| S, stored length | L3H5,L6H3,L9H4,L10H1,L1H1,L2H6 | 150 | 1.063 (0.922, 1.202) | 54.5% (46.3%, 62.6%) | -11.4% (4.9%) | pass |
| R, readers (Lw 6) | L12H6,L9H0,L7H3 | 150 | 0.588 (0.466, 0.708) | 49.9% (43.2%, 57.8%) | 0.6% (1.4%) | pass |

| Newline-only ablation | Natural separation kept (95% CI) | AUROC |
|---|---|---|
| S: L3H5,L6H3,L9H4,L10H1,L1H1,L2H6 | 92.7% (86.8%, 99.2%) | 0.979 |

T, token correctors: L0H0,L1H1,L1H6 ablated at every position.

| Width | Points | Intact (95% CI) | Ablated (95% CI) | Ablated minus intact (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| 40 | 4978 | 0.048 (-0.012, 0.109) | 0.463 (0.388, 0.541) | 0.415 (0.356, 0.481) | 0.075 (0.134) | pass |
| 60 | 3033 | 0.074 (0.038, 0.105) | 0.380 (0.314, 0.456) | 0.306 (0.221, 0.400) | 0.066 (0.105) | pass |
| 80 | 2106 | 0.036 (-0.000, 0.072) | 0.266 (0.219, 0.321) | 0.229 (0.177, 0.289) | 0.039 (0.067) | pass |

Verdicts: C pass, T pass, S pass, R pass

## Reading
- **Behaviour.** On post-2025 text, Pythia-1B predicts the break at letter-ending word ends with AUROC 0.984 at width 60, as on WikiText-2 (0.985).
- **C, counters: pass.** When the current line is made at least 10 characters shorter, patching only these six heads' outputs at the decision token moves the newline log-probability 65% of the way. Ten layer-matched random sets move it 2% on average. The heads sit in layers 1–6.
- **T, token correctors: pass at all three widths.**
  - Ablating L0H0, L1H1 and L1H6 gives tokens on the line a weight of 0.46, 0.38 and 0.27 log-odds per token at widths 40, 60 and 80, against 0.07–0.08 or less for random sets.
  - These are dilution-type heads: their attention to the line-start newline falls as the line fills (exploration, e01). In 410M the corrector was a separate head.
  - A reading consistent with this, though not tested directly: in 1B these heads supply the character-based part of the count, so removing them leaves a count that leans on tokens.
  - One random set reached 0.13 at width 40. The candidate's 0.46 is well clear of it.
- **S, stored length: pass.** Ablating the six writers at newline tokens only removes 55% of the effect of a shorter previous line, and the decision keeps 93% of its break/no-break separation. This is the same dissociation as 410M's H4 (46% removed, 86% kept). In 410M the same four heads counted and stored. In 1B only L1H1, L3H5 and L6H3 do both; L9H4, L10H1 and L2H6 are writers at the newline but not among the count carriers.
- **R, readers: pass.** With the stored length patched into the newline after layer 6, freezing L12H6, L9H0 and L7H3 at the decision token removes 50% of its effect, against 1% for random sets. In 410M three readers carried 74–82%; in 1B the share is lower, so more of the width reaches the decision by other routes.
- **Not predicted, noted.** The intact model gives tokens on the line a small positive weight at width 60, 0.074 (0.038, 0.105). Pythia-410M showed a similar residual token weight at width 40 on both WikiText splits.

## Limits
- One model checkpoint, Wikipedia prose and greedy wrapping only; letter-ending decision points only.
- Mean-ablation and activation patching only. Head sets were chosen in exploration on WikiText-2 train and checked once on the new-text exploration part before the freeze. Thresholds are about half the explored sizes, so the tests guard against the effects vanishing, not against their being somewhat smaller.
- The model runs in bfloat16; a float32 spot check agreed (AUROC 0.9865 against 0.9866).
- The five tests were frozen as separate predictions, with no correction for multiple tests. All passed with intervals far from their thresholds.
