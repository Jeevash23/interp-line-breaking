# D1-X report: Gemma-4-E2B

7 November 2026.
- **Exploration** under `CHARTER.md` and Amendment 3 (`AMENDMENTS.md`) used the WikiText-2 train split and then this model's exploration part of the third English batch (`NOTEBOOK.md`).
- **Confirmation** under `PROTOCOL-X3b.md` (frozen 07:43 UTC; digests in `FROZEN.md`) used `data/newtext3/part-conf-gemma4-e2b.jsonl`: 193 paragraphs from English Wikipedia articles created in 2025–26, read here for the first time. Gemma 4 was released in 2026, so some of these articles may be in its training data; "held out" means that no analysis had read them.
- The script `code/xm3_conf.py` ran once (13:20–08:12 UTC), with no crash and no re-run.
- The tables below are generated from `outputs/xm_conf_gemma4-e2b_conf-gemma4-e2b.json` by `code/xm3_report_tables.py`, and `--check` verifies that they appear here verbatim.

The model is `google/gemma-4-E2B` (base), text decoder only, in bfloat16 on a 6 GB local GPU. A beginning-of-sequence token is prepended and excluded from every measure.
- It has 35 layers of 8 query heads, each layer with one key/value head. Sliding-window layers are interleaved with global layers, which have 512-wide heads against 256 elsewhere.
- Layers 15–34 reuse keys and values computed in earlier layers.
- The model has a per-layer token embedding, read here row by row from the weights file. The image and audio encoders are not loaded.
- Chunked outputs equal the model's own output exactly (`outputs/xm3_gemma_fit.json`).
- **No float32 check was possible** (too much memory), unlike the other six models.

## Summary
Gemma-4-E2B has heads in all four roles, and every frozen test passed on the held-out text. It is the first model in this series with mixed sliding-window and global attention and cross-layer key/value sharing, and it uses the fewest heads for the job.
- **Two heads carry the count.** L6H3 and L12H3 together carry 68% of the current line's count to the decision. Every earlier model needed four to six heads.
- **L6H3 has three roles.** It is a counter, the token corrector and a stored-length writer at the newline, and its attention to the line-start newline falls as the line fills.
- **L12H3 both carries the count and reads the stored length,** as OLMo-2-1B's L6H2 does.

| Role | Heads | Result on the confirmation part |
|---|---|---|
| Counters | L6H3, L12H3 | carry 68% of the current line's count to the decision |
| Token corrector | L6H3 | ablated, each token adds 0.31–0.39 to the newline log-odds |
| Stored length | L6H1, L6H3, L7H6 | at newline tokens only, carry 44% of the previous line's effect; the decision keeps 70% of its separation |
| Readers | L9H5, L12H3 | carry 50% of the stored length's effect at the decision |

## Confirmation results

| Behaviour, width 60 | Value |
|---|---|
| Paragraphs | 60 |
| Decision points | 1192 |
| AUROC | 0.987 |
| Separation | 3.663 |
| Gate (AUROC ≥ 0.90) | pass |

| Test | Set | n | Effect (95% CI) | Share (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| C, counters | L6H3,L12H3 | 150 | -3.065 (-3.364, -2.772) | 68.3% (61.7%, 75.5%) | 0.3% (5.2%) | pass |
| S, stored length | L6H1,L6H3,L7H6 | 150 | 1.578 (1.397, 1.753) | 43.8% (37.1%, 50.2%) | 1.9% (6.6%) | pass |
| R, readers (Lw 7) | L9H5,L12H3 | 150 | 0.996 (0.871, 1.129) | 50.5% (45.1%, 56.6%) | -0.1% (2.1%) | pass |

| Newline-only ablation | Natural separation kept (95% CI) | AUROC |
|---|---|---|
| S: L6H1,L6H3,L7H6 | 70.2% (63.9%, 75.8%) | 0.978 |

T, token correctors: L6H3 ablated at every position.

| Width | Points | Intact (95% CI) | Ablated (95% CI) | Ablated minus intact (95% CI) | Random sets, mean (max) | Verdict |
|---|---|---|---|---|---|---|
| 40 | 4919 | 0.001 (-0.037, 0.042) | 0.390 (0.341, 0.441) | 0.388 (0.333, 0.448) | 0.004 (0.058) | pass |
| 60 | 2975 | 0.032 (-0.009, 0.076) | 0.375 (0.329, 0.425) | 0.343 (0.290, 0.395) | 0.031 (0.068) | pass |
| 80 | 2039 | 0.006 (-0.025, 0.037) | 0.311 (0.264, 0.362) | 0.305 (0.254, 0.362) | 0.004 (0.030) | pass |

Verdicts: C pass, T pass, S pass, R pass

## Reading
- **Behaviour.** AUROC 0.987 at width 60 on post-2025 text.
- **C, counters: pass.** Patching L6H3 and L12H3 at the decision token carries 68% (62%, 75%) of the count effect. Random pairs from the same layers carry at most 5.2%. In exploration, the decision token's residual stream carried the count only from layer 6, and all of it from layer 12, the layers of these two heads.
- **T, token corrector: pass at all three widths.** Ablating L6H3 alone gives tokens on the line 0.39, 0.38 and 0.31 log-odds each at widths 40, 60 and 80; intact, the weight is about 0 (0.00, 0.03, 0.01). Random layer-6 heads reach at most 0.07. This single head matches Pythia-1B's three-head set (0.46, 0.38 and 0.27 there).
- **S, stored length: pass.** L6H1, L6H3 and L7H6, ablated at newline tokens only, remove 44% of the previous line's effect and leave 70% of the decision's separation. This partial dissociation is like OLMo-2-1B's early writers (73% kept there).
- **R, readers: pass.** With the stored length patched into the newline after layer 7, freezing L9H5 and L12H3 at the decision token removes 50% of its effect; random pairs at most 2.1%. The readers carry about half the effect, as in Pythia-1B (50%), against 78–113% in the Qwen and OLMo models.
- **Where the rest of the stored length goes (exploration, not tested here).** Patching the newline's state after any layer from 14 to 34 changes the decision by the same 0.003 (0.2% of the effect), with no change from layer to layer, because layers 15–34 attend with keys and values computed in earlier layers. The constant is probably the small difference between batched and single runs in bfloat16 (not tested). Many small heads in those later layers each read 5–8% of the effect (`NOTEBOOK.md`, xm_e02). Through the shared keys and values they read the newline as it stood at layers 13–14, which may account for the part the two main readers do not carry.

## Limits
- One model checkpoint, Wikipedia prose and greedy wrapping only, and letter-ending decision points only.
- bfloat16 only; no float32 check was possible.
- Mean-ablation and activation patching of query-head outputs. Each layer has a single key/value head shared by all 8 query heads, so key and value routes were not tested.
- The confirmation text is held out from these analyses, not necessarily from the model's training data, which is the most recent of the seven models.
