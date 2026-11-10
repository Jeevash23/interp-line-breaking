# D1-X confirmation protocol X3 (CONFIRMATORY): the line-breaking roles in Qwen3-0.6B and Qwen3-1.7B

Written 7 Nov 2026, morning, under `CHARTER.md` and Amendment 3 (`AMENDMENTS.md`), after exploration on WikiText-2 train and on each model's exploration part of the third English batch (`NOTEBOOK.md`). Frozen, with its spec files and scripts, before any script reads a confirmation part. Digests are in `FROZEN.md`. The loader in `code/xm3_common.py` refuses each model's confirmation part until this file is read-only. Gemma-4-E2B (and SmolLM2-1.7B, if run) will follow in a separate protocol, `PROTOCOL-X3b.md`. Later changes are dated entries in `AMENDMENTS.md`.

## Question
Pythia-410M, and then Pythia-1B, Qwen2.5-0.5B and OLMo-2-1B (`PROTOCOL-X1.md`, `REPORT-*.md`), break lines with four roles:
- counters;
- a stored length at the newline;
- token correctors;
- readers.

Exploration of two newer base models, Qwen3-1.7B and Qwen3-0.6B, found the same roles. **It also found them in the same heads, by layer and index, in both models.** This protocol tests, on text no script has read, whether the named heads do these jobs at stated sizes, beyond layer-matched random heads. Every head set was chosen in Qwen3-1.7B and is used unchanged in Qwen3-0.6B, so the 0.6B tests are also a test of that cross-model match.

## Material and models
- **Text:** `data/newtext3/part-conf-<model>.jsonl`.
  - Qwen3-1.7B: 186 paragraphs from 111 articles. Qwen3-0.6B: 194 paragraphs from 102 articles.
  - These are English Wikipedia articles created January 2025 to September 2026. None of them is in the first two batches.
  - The articles were split by article hash before any reading (`data/newtext3/MANIFEST.md`; digests in `FROZEN.md`). Paragraph order seed 100.
  - Some of these articles may predate the models' training data (Amendment 3). Here "held out" means that no analysis has read the part.
- **Models:** `Qwen/Qwen3-1.7B-Base` and `Qwen/Qwen3-0.6B-Base`, both in bfloat16 on the local GPU. For both, bfloat16 agreed with float32 in the spot checks (r = 0.9997).
- **Script:** `code/xm3_conf.py specs/conf-<model>.json conf-<model>`, one run per model.
  - Measures, samples, ablation means (WikiText-2 train), decision points (letter-ending word ends), line-start marker (any token containing a newline), random sets and bootstrap are exactly as in `PROTOCOL-X1.md`.
  - Random sets: 10 per test, layer-matched, never a head named in any role of the model, numpy seed 0.
  - Bootstrap: 2,000 resamples, seed 0.
  - Samples: the first 60 eligible paragraphs per width; up to 150 count-interchange and 150 width items.
  - Lw = 6 for both models.

## Tests
- **G, behaviour (gate).** As in `PROTOCOL-X1.md`.
- **C, counters (count interchange).** As in `PROTOCOL-X1.md`.
- **T, token corrector.** As in `PROTOCOL-X1.md`.
  - The random-set limit is set at 0.10 for both models. A random head leaves the token weight at its intact level, and on new text that level was up to 0.07 (Qwen3-0.6B, width 40).
  - The paired test (ablated minus intact, interval above 0) is what makes T specific.
- **X, newline writers, no dissociation predicted.** Scored as in `PROTOCOL-X1.md` for OLMo's L5H0. Ablating the set at line-start markers only (same ablation in both prompts) should remove at least X_share of the width effect. It should *also* cost the decision at least a quarter of its natural separation (kept ≤ 0.75).
  - **Pass:** width-effect interval above 0; share ≥ X_share with lower bound > 0; random-set mean < 0.15; kept ≤ 0.75.
  - In exploration these heads removed 37–42% of the width effect but kept only 40–51% of the natural separation. So the clean dissociation seen for Pythia-1B's and Qwen2.5's stored-length heads is not predicted here.
- **R, readers.** As in `PROTOCOL-X1.md`.

Thresholds are about half the smaller of the two explored sizes (WikiText-2 train and the new-text exploration part), rounded down, as in `PROTOCOL-X1.md`.

## Qwen3-1.7B (`specs/conf-qwen3-17b.json`; Lw = 6; sets chosen in this model)
| Test | Set | Explored (train / new text) | Threshold | Random limit |
|---|---|---|---|---|
| C | L4H0, L4H1, L5H13, L6H2, L7H15 | 71% / 69% | share ≥ 0.30 | mean < 0.15 |
| T | L4H1 | ablated 0.30, 0.20, 0.14 / 0.27, 0.20, 0.13 at widths 40, 60, 80 | ≥ 0.06 at 40, 60 and 80 | mean < 0.10 |
| X | L5H12, L5H13, L6H2 | 37%, kept 51% / 39%, kept 51% | share ≥ 0.18; kept ≤ 0.75 | mean < 0.15 |
| R | L7H14, L15H3, L17H6, L22H3 | 79% / 78% | share ≥ 0.35 | mean < 0.15 |

## Qwen3-0.6B (`specs/conf-qwen3-06b.json`; Lw = 6; the same sets, unchanged)
| Test | Set | Explored (train / new text) | Threshold | Random limit |
|---|---|---|---|---|
| C | L4H0, L4H1, L5H13, L6H2, L7H15 | 76% / 76% | share ≥ 0.35 | mean < 0.15 |
| T | L4H1 | single-head scan 0.19 at width 60 / ablated 0.20, 0.20, 0.11 at widths 40, 60, 80 | ≥ 0.05 at 40, 60 and 80 | mean < 0.10 |
| X | L5H12, L5H13, L6H2 | 42%, kept 41% / 40%, kept 40% | share ≥ 0.20; kept ≤ 0.75 | mean < 0.15 |
| R | L7H14, L15H3, L17H6, L22H3 | 76% / 74% | share ≥ 0.35 | mean < 0.15 |

## Predictions that would not be made from the earlier models alone
- **The same heads in both models.** Qwen3-0.6B and Qwen3-1.7B differ in width (hidden size 1,024 vs 2,048) but share 28 layers × 16 query heads. Every set above was chosen in Qwen3-1.7B. If C, X and R pass in Qwen3-0.6B with these sets, the head-for-head match holds on held-out text, not only in exploration. Why it holds (a shared origin such as distillation) is not tested here.
- **One head in three roles.** L4H1 is a counter (in C), the token corrector (T) and a dilution head (its newline attention falls as the line fills). This is the arrangement seen in Pythia-1B and Qwen2.5-0.5B, now in a single head.
- **Newline writers without a dissociation (X).** As OLMo's L5H0, and unlike Pythia-1B's and Qwen2.5's stored-length heads.

## What counts
For each model, the roles that pass are confirmed for that model. **The cross-model claim (same heads, same roles) counts as confirmed if C, X and R all pass in Qwen3-0.6B.** T in Qwen3-0.6B is reported but not part of that claim, because L4H1's corrector role was not tested in 0.6B on train. If the gate fails for a model, its role tests are still run and reported, but no role claim is made for it.

## What is reported
For each model, `REPORT-<model>.md` gives every test's numbers and verdict as frozen, whichever way they go. Its tables are generated from the saved output by `code/xm3_report_tables.py` and checked with `--check`. Nulls and failures are stated plainly. No re-runs and no threshold changes after a confirmation part has been read.

## Order and compute
- One GPU job per model, Qwen3-1.7B first, then Qwen3-0.6B, through `code/run.sh` (memory cage, shared GPU lock, at most one D1-X job waiting).
- Each run is expected to take 15–35 minutes, which is above the charter's 20-minute guide for 1.7B; the confirmation script is not split, so that it matches the transfer checks exactly.
- A crash before any output is written may be re-run once, unchanged, and logged.
