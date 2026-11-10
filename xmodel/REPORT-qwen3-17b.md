# D1-X report: Qwen3-1.7B

7 November 2026.
- **Exploration** under `CHARTER.md` and Amendment 3 (`AMENDMENTS.md`) used the WikiText-2 train split and then this model's exploration part of the third English batch (`NOTEBOOK.md`).
- **Confirmation** under `PROTOCOL-X3.md` (frozen 03:30 UTC; digests in `FROZEN.md`) used `data/newtext3/part-conf-qwen3-17b.jsonl`: 186 paragraphs from English Wikipedia articles created in 2025–26, read here for the first time. Some of these articles may predate the model's training data, so "held out" means that no analysis had read them.
- The script `code/xm3_conf.py` ran once (09:20–04:09 UTC), with no crash and no re-run.
- The tables below are generated from `outputs/xm_conf_qwen3-17b_conf-qwen3-17b.json` by `code/xm3_report_tables.py`, and `--check` verifies that they appear here verbatim.

The model is `Qwen/Qwen3-1.7B-Base` in bfloat16, which matched float32 in a spot check (r = 0.9997). It has 28 layers of 16 query heads sharing 8 key/value heads; all tests act on query heads' outputs. Decision points are word ends whose last character is a letter, and the line-start marker is any token containing a newline (`CHARTER.md`).

## Summary
Qwen3-1.7B has heads in all four roles, and every frozen test passed on the held-out text.
- One head, **L4H1**, has three roles. It is a counter, it is the token corrector, and its attention to the line-start newline falls as the line fills.
- The heads that write the finished line's length at the newline cannot be removed there without damaging the decision itself, as for OLMo-2-1B's L5H0. The protocol predicted this.
- The same heads, by layer and index, play the same roles in Qwen3-0.6B (`REPORT-qwen3-06b.md`).

| Role | Heads | Result on the confirmation part |
|---|---|---|
| Counters | L4H0, L4H1, L5H13, L6H2, L7H15 | carry 70% of the current line's count to the decision |
| Token corrector | L4H1 | ablated, each token adds 0.13–0.25 to the newline log-odds |
| Newline writers | L5H12, L5H13, L6H2 | at newline tokens only, carry 44% of the previous line's effect; the decision keeps 52% of its separation |
| Readers | L7H14, L15H3, L17H6, L22H3 | carry 78% of the stored length's effect at the decision |

## Confirmation results

| Behaviour, width 60 | Value |
|---|---|
| Paragraphs | 60 |
| Decision points | 1306 |
| AUROC | 0.988 |
| Separation | 3.526 |
| Gate (AUROC ≥ 0.90) | pass |

| Test | Set | n | Effect (95% CI) | Share (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| C, counters | L4H0,L4H1,L5H13,L6H2,L7H15 | 150 | -2.763 (-3.005, -2.530) | 70.4% (65.3%, 75.6%) | -0.2% (0.0%) | pass |
| X, newline writers (no dissociation predicted) | L5H12,L5H13,L6H2 | 135 | 1.260 (1.091, 1.439) | 44.2% (38.7%, 49.7%) | -0.6% (2.2%) | pass |
| R, readers (Lw 6) | L7H14,L15H3,L17H6,L22H3 | 135 | 0.519 (0.410, 0.627) | 78.2% (71.3%, 87.1%) | -0.4% (0.7%) | pass |

| Newline-only ablation | Natural separation kept (95% CI) | AUROC |
|---|---|---|
| X: L5H12,L5H13,L6H2 | 52.3% (49.8%, 54.9%) | 0.980 |

T, token correctors: L4H1 ablated at every position.

| Width | Points | Intact (95% CI) | Ablated (95% CI) | Ablated minus intact (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| 40 | 5078 | 0.015 (-0.022, 0.052) | 0.254 (0.211, 0.300) | 0.238 (0.214, 0.268) | 0.015 (0.019) | pass |
| 60 | 3099 | -0.002 (-0.038, 0.041) | 0.184 (0.153, 0.219) | 0.186 (0.152, 0.216) | -0.002 (0.001) | pass |
| 80 | 2115 | -0.005 (-0.032, 0.026) | 0.128 (0.100, 0.161) | 0.133 (0.107, 0.163) | -0.005 (-0.003) | pass |

Verdicts: C pass, T pass, X pass, R pass

## Reading
- **Behaviour.** AUROC 0.988 at width 60 on post-2025 text, as on WikiText-2 train.
- **C, counters: pass.** Patching five heads in layers 4–7 at the decision token carries 70% of the count effect. Random sets carry nothing; the largest is 0.0%. L7H15 carried the most on its own in exploration (46%).
- **T, token corrector: pass at all three widths.** Ablating L4H1 alone gives tokens on the line 0.25, 0.18 and 0.13 log-odds each at widths 40, 60 and 80; intact, the weight is about 0. Random heads from layer 4 leave it at the intact level (at most 0.02). This single head has about half the effect of Pythia-1B's three-head set on its confirmation part (0.27–0.46 there).
- **X, newline writers: pass.** Ablating L5H12, L5H13 and L6H2 at newline tokens only removes 44% of the previous line's effect. It also costs the decision about half its natural separation (52% kept). So these heads carry a large share of the stored length, and the newline state they write is needed for the decision beyond the width, as the protocol predicted. L5H13 is also a counter whose attention to the newline falls as the line fills.
- **R, readers: pass.** With the stored length patched into the newline after layer 6, freezing L7H14, L15H3, L17H6 and L22H3 at the decision token removes 78% of its effect; random sets remove at most 0.7%.
- **The L7H14 and L7H15 pair.** These two heads share one key/value head. In exploration L7H14 was the strongest reader of the stored width, while L7H15 carried the most of the current count. When L7H14 was drawn into a random counter set, it pushed the decision against the count by about a third of the effect. The two heads read the same keys and values with different queries. This was noted in exploration and not tested by the protocol.

## Limits
- One model checkpoint, Wikipedia prose and greedy wrapping only, and letter-ending decision points only.
- Mean-ablation and activation patching of query-head outputs. Key and value routes are shared between pairs of heads and were not tested separately.
- The confirmation text is held out from these analyses, not necessarily from the model's training data.
- bfloat16 inference; float32 agreed in a spot check on train text.
