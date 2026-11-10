# D1-X report: Qwen3-0.6B

7 November 2026.
- **Exploration** under `CHARTER.md` and Amendment 3 (`AMENDMENTS.md`) used the WikiText-2 train split and then this model's exploration part of the third English batch (`NOTEBOOK.md`).
- **Confirmation** under `PROTOCOL-X3.md` (frozen 03:30 UTC; digests in `FROZEN.md`) used `data/newtext3/part-conf-qwen3-06b.jsonl`: 194 paragraphs from English Wikipedia articles created in 2025–26, read here for the first time. Some of these articles may predate the model's training data, so "held out" means that no analysis had read them.
- The script `code/xm3_conf.py` ran once (09:49–04:29 UTC), with no crash and no re-run.
- The tables below are generated from `outputs/xm_conf_qwen3-06b_conf-qwen3-06b.json` by `code/xm3_report_tables.py`, and `--check` verifies that they appear here verbatim.

The model is `Qwen/Qwen3-0.6B-Base` in bfloat16, which matched float32 in a spot check (r = 0.9997). Like Qwen3-1.7B it has 28 layers of 16 query heads sharing 8 key/value heads, with half the width (hidden size 1,024 against 2,048).

**Every head set here was chosen in Qwen3-1.7B and used unchanged.** The 0.6B model's own exploration found the same heads (`NOTEBOOK.md`), but nothing was re-selected for it.

## Summary
Every frozen test passed on the held-out text, with Qwen3-1.7B's head sets. By the protocol's rule (C, X and R pass in Qwen3-0.6B), **the cross-model claim is confirmed: the two Qwen3 models use the same heads, by layer and index, for the same line-breaking roles.** The sizes are close to Qwen3-1.7B's (`REPORT-qwen3-17b.md`). The protocol does not test why the heads match across the two sizes (for example, a shared origin through distillation).

| Role | Heads (chosen in Qwen3-1.7B) | Result on the confirmation part | Qwen3-1.7B |
|---|---|---|---|
| Counters | L4H0, L4H1, L5H13, L6H2, L7H15 | carry 76% of the current line's count to the decision | 70% |
| Token corrector | L4H1 | ablated, each token adds 0.11–0.21 to the newline log-odds | 0.13–0.25 |
| Newline writers | L5H12, L5H13, L6H2 | at newline tokens only, carry 43% of the previous line's effect; the decision keeps 42% of its separation | 44%; 52% |
| Readers | L7H14, L15H3, L17H6, L22H3 | carry 78% of the stored length's effect at the decision | 78% |

## Confirmation results

| Behaviour, width 60 | Value |
|---|---|
| Paragraphs | 60 |
| Decision points | 1229 |
| AUROC | 0.987 |
| Separation | 3.349 |
| Gate (AUROC ≥ 0.90) | pass |

| Test | Set | n | Effect (95% CI) | Share (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| C, counters | L4H0,L4H1,L5H13,L6H2,L7H15 | 150 | -3.481 (-3.761, -3.224) | 75.5% (71.5%, 79.9%) | 0.0% (0.6%) | pass |
| X, newline writers (no dissociation predicted) | L5H12,L5H13,L6H2 | 143 | 1.431 (1.252, 1.622) | 43.2% (38.3%, 47.7%) | -1.0% (2.3%) | pass |
| R, readers (Lw 6) | L7H14,L15H3,L17H6,L22H3 | 143 | 0.833 (0.707, 0.965) | 77.6% (73.6%, 82.3%) | 0.1% (0.7%) | pass |

| Newline-only ablation | Natural separation kept (95% CI) | AUROC |
|---|---|---|
| X: L5H12,L5H13,L6H2 | 42.0% (39.7%, 44.2%) | 0.975 |

T, token correctors: L4H1 ablated at every position.

| Width | Points | Intact (95% CI) | Ablated (95% CI) | Ablated minus intact (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| 40 | 4884 | 0.052 (0.020, 0.085) | 0.211 (0.179, 0.251) | 0.159 (0.128, 0.195) | 0.053 (0.061) | pass |
| 60 | 2970 | 0.006 (-0.033, 0.045) | 0.190 (0.159, 0.222) | 0.184 (0.148, 0.218) | 0.008 (0.017) | pass |
| 80 | 2053 | 0.010 (-0.027, 0.044) | 0.108 (0.089, 0.129) | 0.098 (0.071, 0.130) | 0.012 (0.018) | pass |

Verdicts: C pass, T pass, X pass, R pass

## Reading
- **Behaviour.** AUROC 0.987 at width 60 on post-2025 text.
- **C, counters: pass.** Patching the five heads at the decision token carries 76% of the count effect; random sets carry at most 0.6%.
- **T, token corrector: pass at all three widths.** Ablating L4H1 gives tokens on the line 0.21, 0.19 and 0.11 log-odds each at widths 40, 60 and 80. Random heads from layer 4 leave the weight at its intact level (0.05, 0.01 and 0.01). This was the first test of L4H1 as token corrector in this model on held-out text. Its exploration used only the single-head scan and the new-text exploration part, so T was not part of the cross-model claim.
- **X, newline writers: pass.** At newline tokens only, the three heads remove 43% of the previous line's effect, and the decision keeps 42% of its natural separation. As in Qwen3-1.7B, and as predicted, the newline state these heads write is needed for the decision beyond the width. The loss is larger here than in 1.7B (42% kept against 52%).
- **R, readers: pass.** Freezing the four readers at the decision token removes 78% of the stored length's effect after layer 6; random sets remove at most 0.7%.
- **Not predicted, noted.** The intact token weight at width 40 is above 0, at 0.05 (0.02, 0.08). The same was seen on this model's exploration part (0.07), and in Pythia-410M and Qwen2.5-0.5B.

## Limits
- One model checkpoint, Wikipedia prose and greedy wrapping only, and letter-ending decision points only.
- Mean-ablation and activation patching of query-head outputs. Key and value routes are shared between pairs of heads and were not tested separately.
- The confirmation text is held out from these analyses, not necessarily from the model's training data.
- The head-for-head match is shown for the roles tested here. Whether the two models match elsewhere, and why, is not tested.
