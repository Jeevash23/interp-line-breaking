# D1-X confirmation protocol (CONFIRMATORY): the four roles in three models

Written 6 Nov 2026 under `CHARTER.md` and Amendment 1, after exploration on WikiText-2 train and on each model's new-text exploration part (`NOTEBOOK.md`). Frozen, with its spec files and scripts, before any script reads a confirmation part. Digests are in `FROZEN.md`. Later changes are dated entries in `AMENDMENTS.md`. This round is covered by the author's "New text, then models".

## Question
Pythia-410M breaks lines with four roles (`../REPORT.md`, `../REPORT-2.md`): counters, a stored length at the newline, token correctors, and readers. Exploration found heads in each role in Pythia-1B, Qwen2.5-0.5B and OLMo-2-1B, arranged differently in each model. This protocol tests, on text no script has read, whether those named heads do the jobs exploration found, at stated sizes, beyond layer-matched random heads.

## Material and models
- **Text:** `../data/newtext/part-conf-<model>.jsonl` (Pythia-1B 274 paragraphs, Qwen2.5-0.5B 269, OLMo-2-1B 238). These are English Wikipedia articles created between January 2025 and September 2026, split by article before any reading (`../data/newtext/MANIFEST.md`, digests in `../FROZEN.md`). Paragraph order seed 100.
- **Models:** Pythia-1B (bfloat16), Qwen2.5-0.5B (float32), OLMo-2-1B (bfloat16), on the local GPU. bfloat16 agreed with float32 in the spot checks (`NOTEBOOK.md`).
- **Ablation means:** WikiText-2 train, 30 paragraphs (seed 6) wrapped at 40, 60 and 80 in turn, first 700 tokens; all-position means, and line-start-marker means for newline-only ablation.
- **Decision points:** word ends whose last character is a letter (charter). The line-start marker is any token containing a newline.
- **Script:** `code/xm_conf.py <spec> conf-<model>`, one run per model, with `specs/conf-<model>.json`. It reads only that model's confirmation part. Samples: the first 60 eligible paragraphs per width; up to 150 count-interchange items (seed 300) and up to 150 width items (seed 200), at width 60.
- **Random sets:** 10 per test, layer-matched to the candidate set, no repeats, never a head named in any role of that model (numpy seed 0). Bootstrap: 2,000 resamples (items or paragraphs), seed 0.

## Tests (the same five kinds for every model)
- **G, behaviour (reported; the charter's gate).** AUROC of the newline log-odds at width 60. If AUROC < 0.90, the model's role tests are still run and reported, but no role claim is made for it.
- **C, counters (count interchange).** Base prompt: wrapped text up to a letter-ending word with 4–14 characters left on the line. Short prompt: the same, with leading words of the current line dropped until at least 10 characters are gone (earlier lines unchanged). Share = mean change in log P(newline) when only the set's decision-token outputs are patched from the short run, over the mean total change. **Pass:** total effect interval below 0; share ≥ C_share with lower bound > 0; random-set mean < C_random.
- **T, token correctors.** Coefficient of tokens-on-the-line in the newline log-odds (≤ 30 characters left), given a spline in characters remaining, intact and with the set mean-ablated at every position. **Pass at a width:** ablated ≥ T_coef; paired (ablated − intact) interval above 0; random-set mean < T_random. **T passes** if it passes at every width listed in T_widths.
- **S, stored length (newline-only ablation).** Width items: the previous line shortened by ≥ 10 characters. Share of the width effect removed when the set is ablated at line-start markers only (same ablation in both prompts); and the natural-text separation kept under the same ablation. **Pass:** width-effect interval above 0; share ≥ S_share with lower bound > 0; random-set mean < S_random; separation kept ≥ S_kept.
- **R, readers.** The marker's residual after layer Lw is patched from the short-previous-line run; the set is frozen at the decision token to its base-run value. Mediation share = 1 − frozen effect / patched effect. **Pass:** patched-effect interval above 0; share ≥ R_share with lower bound > 0; random-set mean < R_random.

Thresholds are about half the smaller of the two explored sizes (WikiText-2 train and the new-text exploration part), rounded down, as in D1's rounds. Random-set limits are set per model from the random sets seen in exploration.

## Pythia-1B (`specs/conf-pythia1b.json`; Lw = 6)
| Test | Set | Explored (train / new text) | Threshold | Random limit |
|---|---|---|---|---|
| C | L1H1, L1H6, L2H5, L3H0, L3H5, L6H3 | 68% / 64% | share ≥ 0.30 | mean < 0.15 |
| T | L0H0, L1H1, L1H6 | 0.47, 0.43, 0.28 / 0.49, 0.36, 0.24 at widths 40, 60, 80 | ≥ 0.12 at 40, 60 and 80 | mean < 0.15 |
| S | L3H5, L6H3, L9H4, L10H1, L1H1, L2H6 | 54%, kept 89% / 52%, kept 92% | share ≥ 0.25; kept ≥ 0.75 | mean < 0.15 |
| R | L12H6, L9H0, L7H3 | 54% / 46% | share ≥ 0.20 | mean < 0.15 |

Prediction that would not be made from 410M alone: the heads whose removal lets tokens in (T) are dilution heads that also carry the count (C shares L1H1 and L1H6), rather than a separate corrector.

## Qwen2.5-0.5B (`specs/conf-qwen05b.json`; Lw = 3)
| Test | Set | Explored (train / new text) | Threshold | Random limit |
|---|---|---|---|---|
| C | L2H1, L3H1, L3H2, L6H1 | 59% / 63% | share ≥ 0.30 | mean < 0.15 |
| T | L1H8, L1H9, L2H1, L3H1 | 0.16, 0.21, 0.13 / 0.14, 0.21, 0.15 | ≥ 0.07 at 40, 60 and 80 | mean < 0.05 |
| S | L2H1, L3H1, L3H2 | 33%, kept 79% / 36%, kept 82% | share ≥ 0.15; kept ≥ 0.65 | mean < 0.15 |
| R | L6H4, L4H8, L6H6 | 98% / 100% | share ≥ 0.50 | mean < 0.15 |

As in 410M, the same heads count (C) and store the finished line's length at the newline (S: L2H1, L3H1, L3H2).

## OLMo-2-1B (`specs/conf-olmo1b.json`; Lw = 3)
| Test | Set | Explored (train / new text) | Threshold | Random limit |
|---|---|---|---|---|
| C | L6H2, L7H10, L11H4, L14H15 | 71% / 74% | share ≥ 0.35 | mean < 0.15 |
| T | L1H10, L0H5, L3H15 | 0.40, 0.37, 0.31 / 0.35, 0.30, 0.28 | ≥ 0.14 at 40, 60 and 80 | mean < 0.10 |
| S | L0H9, L1H10, L1H12, L3H7 | 29%, kept 71% / 34%, kept 69% | share ≥ 0.14; kept ≥ 0.55 | mean < 0.15 |
| X | L5H0 (second stored-length set) | 74%, kept 55% / 76%, kept 55% | share ≥ 0.35; kept ≤ 0.75 | mean < 0.15 |
| R | L6H2, L7H10, L11H4 | 115% / 111% | share ≥ 0.50 | mean < 0.15 |

**X, the second stored-length test.** It is scored like S, except that it predicts the *absence* of a full dissociation. Ablating L5H0 at newline tokens alone should remove at least 35% of the width effect *and* cost the decision at least a quarter of its natural separation (kept ≤ 0.75). **Pass:** width-effect interval above 0; share ≥ 0.35 with lower bound > 0; random-set mean < 0.15; kept ≤ 0.75.

Predictions that would not be made from 410M alone:
- the count carriers and the width readers are the same heads (C and R share L6H2, L7H10 and L11H4);
- the token correctors (T) carry little of the count themselves;
- most of the stored length passes through one head at the newline (L5H0), whose newline state the decision also needs (X).

## What is reported
For each model, `REPORT-<model>.md`: every test's numbers and verdict as frozen, whichever way they go, with tables generated from the saved output by `code/xm_report_tables.py` and checked with `--check`. Nulls and failures are stated plainly. No re-runs and no threshold changes after reading a confirmation part.

## Order and compute
One GPU job per model, in the order Pythia-1B, Qwen2.5-0.5B, OLMo-2-1B, through `code/run.sh` (memory cage, shared GPU lock, pause while `~/.cache/d1-gpu.round3` exists). Each run is expected to take 10–20 minutes. A crash before any output is written may be re-run once, unchanged, and logged.
