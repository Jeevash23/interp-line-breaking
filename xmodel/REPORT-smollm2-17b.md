# D1-X report: SmolLM2-1.7B

7 November 2026.
- **Exploration** under `CHARTER.md` and Amendment 3 (`AMENDMENTS.md`) used the WikiText-2 train split and then this model's exploration part of the third English batch (`NOTEBOOK.md`).
- **Confirmation** under `PROTOCOL-X3b.md` and its SmolLM2 addendum `PROTOCOL-X3b-smollm2.md` (frozen 09:59 UTC; digests in `FROZEN.md`) used `data/newtext3/part-conf-smollm2-17b.jsonl`: 158 paragraphs from English Wikipedia articles created in 2025–26, read here for the first time. Whether they post-date the model's training data was not checked; "held out" means that no analysis had read them.
- The script `code/xm3_conf.py` ran once (15:29–10:07 UTC), with no crash and no re-run.
- The tables below are generated from `outputs/xm_conf_smollm2-17b_conf-smollm2-17b.json` by `code/xm3_report_tables.py`, and `--check` verifies that they appear here verbatim.

The model is `HuggingFaceTB/SmolLM2-1.7B` (base) in bfloat16, with 24 layers of 32 heads, each with its own key and value. **The float32 check could not be run**, because the local machine did not have enough free memory today. Only its bfloat16 half was run (AUROC 0.987 on the 1,264 check points).

## Summary
SmolLM2-1.7B has heads in all four roles, and every frozen test passed on the held-out text.
- **A single head reads the stored length.** L11H21, one of 32 heads in its layer, carries about half of the stored length's effect at the decision. In exploration it also carried the most of the current count.
- **L5H21 has three roles.** It is a counter, a token corrector and a newline writer. Unlike the corresponding heads in Qwen3 (L4H1) and Gemma (L6H3), its attention to the line-start newline rises as the line fills, not falls.
- **The newline writers can be removed with the decision mostly intact,** as in Qwen2.5-0.5B, unlike Qwen3.

| Role | Heads | Result on the confirmation part |
|---|---|---|
| Counters | L5H21, L7H22, L9H7 | carry 73% of the current line's count to the decision |
| Token correctors | L5H21, L3H2 | ablated, each token adds 0.18–0.23 to the newline log-odds |
| Stored length | L5H21, L9H7, L6H25, L2H17 | at newline tokens only, carry 36% of the previous line's effect; the decision keeps 86% of its separation |
| Reader | L11H21 | carries 48% of the stored length's effect at the decision |

## Confirmation results

| Behaviour, width 60 | Value |
|---|---|
| Paragraphs | 60 |
| Decision points | 1191 |
| AUROC | 0.986 |
| Separation | 3.344 |
| Gate (AUROC ≥ 0.90) | pass |

| Test | Set | n | Effect (95% CI) | Share (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| C, counters | L5H21,L7H22,L9H7 | 146 | -2.623 (-2.856, -2.402) | 72.9% (69.0%, 77.3%) | 0.1% (0.3%) | pass |
| S, stored length | L5H21,L9H7,L6H25,L2H17 | 149 | 1.232 (1.068, 1.395) | 36.2% (29.0%, 43.3%) | 0.5% (4.5%) | pass |
| R, readers (Lw 10) | L11H21 | 149 | 0.745 (0.634, 0.854) | 47.7% (42.8%, 52.2%) | -0.5% (0.0%) | pass |

| Newline-only ablation | Natural separation kept (95% CI) | AUROC |
|---|---|---|
| S: L5H21,L9H7,L6H25,L2H17 | 86.0% (82.2%, 89.9%) | 0.983 |

T, token correctors: L5H21,L3H2 ablated at every position.

| Width | Points | Intact (95% CI) | Ablated (95% CI) | Ablated minus intact (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| 40 | 4757 | 0.037 (0.005, 0.069) | 0.229 (0.193, 0.266) | 0.192 (0.157, 0.227) | 0.038 (0.044) | pass |
| 60 | 2840 | 0.004 (-0.023, 0.033) | 0.225 (0.192, 0.264) | 0.221 (0.192, 0.255) | 0.004 (0.006) | pass |
| 80 | 1931 | 0.008 (-0.014, 0.030) | 0.179 (0.142, 0.221) | 0.171 (0.137, 0.211) | 0.008 (0.009) | pass |

Verdicts: C pass, T pass, S pass, R pass

## Reading
- **Behaviour.** AUROC 0.986 at width 60 on post-2025 text.
- **C, counters: pass.** Patching L5H21, L7H22 and L9H7 at the decision token carries 73% (69%, 77%) of the count effect; random sets carry at most 0.3%. In exploration, the decision token's residual stream carried 29% and 30% of the count after layers 5 and 6, and 93% after layer 7, where L7H22 sits. L7H22 is the head with the largest attention to the line-start newline.
- **T, token correctors: pass at all three widths.** Ablating L5H21 and L3H2 gives tokens on the line 0.23, 0.22 and 0.18 log-odds each at widths 40, 60 and 80; intact, 0.04, 0.00 and 0.01. Random pairs leave the weight at the intact level (at most 0.04). Of the two heads, L3H2 is a dilution head (its attention to the newline falls as the line fills), while L5H21's attention rises.
- **S, stored length: pass.** Ablating the four writers at newline tokens only removes 36% of the previous line's effect, and the decision keeps 86% of its separation, as in Qwen2.5-0.5B (85%). In exploration the stored length built up over layers 1–10, and two heads (L8H1, L2H10) wrote against it, which may be why this set's share is moderate.
- **R, reader: pass.** With the stored length patched into the newline after layer 10, freezing L11H21 alone at the decision token removes 48% of its effect; random single heads from layer 11 remove −1.8% to 0.0%.

## Limits
- One model checkpoint, Wikipedia prose and greedy wrapping only, and letter-ending decision points only.
- bfloat16 only: the float32 check could not run today for lack of free memory.
- Mean-ablation and activation patching of head outputs; key and value routes were not tested separately.
- The confirmation text is held out from these analyses, not necessarily from the model's training data.
