# D1-X report: OLMo-2-1B

6 November 2026. OLMo-2-1B (`allenai/OLMo-2-0425-1B`) was added by Amendment 1 after the author approved the download. **Exploration** under `CHARTER.md` and Amendment 1 used the WikiText-2 train split and then the new-text exploration part (`NOTEBOOK.md`). **Confirmation** under `PROTOCOL-X1.md` (frozen 11:38 UTC; digests in `FROZEN.md`) used `part-conf-olmo1b.jsonl`: 238 paragraphs from English Wikipedia articles created in 2025–26, read here for the first time. The script `code/xm_conf.py` ran once (17:29–12:07 UTC), with no crash and no re-run. The tables below are generated from `outputs/xm_conf_olmo1b_conf-olmo1b.json` by `code/xm_report_tables.py`, and `--check` verifies that they appear here verbatim. The model runs in bfloat16; a float32 spot check agreed (AUROC 0.98660 against 0.98657).

## Summary
OLMo-2-1B has heads in all four roles, and every frozen test passed on the held-out new text, including one that predicted the absence of a dissociation. Its arrangement is the most different of the three models:
- **The count carriers and the width readers are the same heads.** L6H2, L7H10 and L11H4 carry most of the current line's count to the decision and also most of the stored length.
- **The token correctors are separate early heads** that carry little of the count themselves.
- **Most of the stored length passes through one head at the newline**, L5H0, and the decision also needs the newline state that L5H0 writes.

| Role | Heads | Result on the confirmation part |
|---|---|---|
| Counters | L6H2, L7H10, L11H4, L14H15 | carry 74% of the current line's count to the decision |
| Token correctors | L1H10, L0H5, L3H15 | ablated, each token adds 0.29–0.36 to the newline log-odds |
| Stored length, early writers | L0H9, L1H10, L1H12, L3H7 | at newline tokens only, carry 37% of the previous line's effect; the decision keeps 73% |
| Stored length, L5H0 | L5H0 | at newline tokens only, carries 76%; the decision keeps only 57% |
| Readers | L6H2, L7H10, L11H4 | freezing them removes 113% of the stored length's effect |

## Confirmation results

| Behaviour, width 60 | Value |
|---|---|
| Paragraphs | 60 |
| Decision points | 1251 |
| AUROC | 0.988 |
| Separation | 3.422 |
| Gate (AUROC ≥ 0.90) | pass |

| Test | Set | n | Effect (95% CI) | Share (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| C, counters | L6H2,L7H10,L11H4,L14H15 | 150 | -3.608 (-3.896, -3.331) | 73.5% (71.3%, 75.7%) | 0.7% (5.6%) | pass |
| S, stored length | L0H9,L1H10,L1H12,L3H7 | 150 | 1.395 (1.227, 1.568) | 36.7% (28.6%, 44.0%) | -2.0% (4.8%) | pass |
| X, stored length (second set) | L5H0 | 150 | 1.395 (1.227, 1.568) | 75.9% (71.4%, 80.8%) | 0.1% (1.2%) | pass |
| R, readers (Lw 3) | L6H2,L7H10,L11H4 | 150 | 0.836 (0.698, 0.978) | 112.7% (105.6%, 121.7%) | 0.4% (3.7%) | pass |

| Newline-only ablation | Natural separation kept (95% CI) | AUROC |
|---|---|---|
| S: L0H9,L1H10,L1H12,L3H7 | 73.2% (68.7%, 77.2%) | 0.977 |
| X: L5H0 | 57.2% (53.7%, 60.9%) | 0.979 |

T, token correctors: L1H10,L0H5,L3H15 ablated at every position.

| Width | Points | Intact (95% CI) | Ablated (95% CI) | Ablated minus intact (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| 40 | 4978 | -0.001 (-0.045, 0.045) | 0.360 (0.309, 0.413) | 0.361 (0.315, 0.411) | -0.002 (0.052) | pass |
| 60 | 3015 | -0.018 (-0.054, 0.021) | 0.309 (0.259, 0.360) | 0.327 (0.275, 0.373) | -0.021 (0.031) | pass |
| 80 | 2125 | -0.029 (-0.070, 0.007) | 0.289 (0.243, 0.339) | 0.318 (0.278, 0.366) | -0.038 (0.005) | pass |

Verdicts: C pass, T pass, S pass, X pass, R pass

## Reading
- **Behaviour.** AUROC 0.988 at width 60 on post-2025 text (WikiText-2 train: 0.986).
- **C, counters: pass.** Patching L6H2, L7H10, L11H4 and L14H15 at the decision token carries 74% of the count effect; random sets 1% on average. In exploration, the decision-token residual held only about 12% of the count effect up to layer 5 and 82% after layer 6 (e05), so the count reaches the decision late, mainly through L6H2.
- **R, readers: pass.** Freezing L6H2, L7H10 and L11H4 removes 113% of the patched effect. A share above 100% means that, with these heads frozen, the stored length moves the decision slightly the opposite way, so some other path partly offsets theirs.
- **C and R together.** Three of the four count carriers are the three readers. L6H2's attention to the line-start newline falls as the line fills (exploration, e01), so one head's attention does the counting and also brings in what the newline holds. That is what a comparison of the current line against the stored length through the newline would look like. It is a reading of the evidence, not a separate test.
- **T, token correctors: pass at all three widths.** Ablating L1H10, L0H5 and L3H15 gives tokens on the line 0.36, 0.31 and 0.29 log-odds each at widths 40, 60 and 80; the random-set means are at or below 0, and the largest single random set reaches 0.05. In exploration these heads, with L2H14, carried only 9% of the count to the decision (e05). So, unlike Pythia-1B and Qwen, OLMo's correction happens away from the main count carriers, as in Pythia-410M.
- **S, early writers: pass.** L0H9, L1H10, L1H12 and L3H7 at newline tokens remove 37% of the previous line's effect, and the decision keeps 73% of its separation. That is a partial dissociation, weaker than in 410M (86%) or Pythia-1B (93%).
- **X, L5H0: pass.** L5H0 alone, at newline tokens only, removes 76% of the previous line's effect and costs the decision 43% of its separation (57% kept), as frozen: a large width share with no clean dissociation. The newline state written by L5H0 is needed for more than the width. One reading is that L6H2 uses the newline as the anchor whose attention share it dilutes, as well as the source of the stored length. This was not tested separately.

## Limits
- One model checkpoint, Wikipedia prose and greedy wrapping only; letter-ending decision points only.
- Mean-ablation and activation patching only. Head sets were chosen on WikiText-2 train and checked once on the new-text exploration part before the freeze. Thresholds are about half the explored sizes.
- Mediation shares above 100% (R) show that other paths offset the readers' effect. Which paths do so was not studied.
- The six tests were frozen as separate predictions, with no correction for multiple tests. All passed with intervals far from their thresholds.
