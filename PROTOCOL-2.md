# D1 second confirmation protocol (CONFIRMATORY)

Written 6 Nov 2026, after the first confirmation (`PROTOCOL.md`, validation split) and two exploratory runs on the train split (exp20, exp21). The author chose a second confirmation round on the test split ("Second confirmation round"). This protocol, its four scripts and the test data were digested in `FROZEN.md` before any script read the test split. Later changes are dated entries in `AMENDMENTS.md`. No threshold from `PROTOCOL.md` is changed.

The test split is the last unread WikiText-2 material, so everything still open that can be tested on held-out text is tested here.

## Background (exploratory, from `NOTEBOOK.md`)
- **Round 1 (validation).**
  - H2 passed: heads L8H15, L17H1 and L6H2 at the decision token carry most of the previous line's influence.
  - H1 failed on one of its three parts: ablating L5H7 gave tokens a large weight, as predicted (H1b, H1c). But at width 40 the intact model kept a small token weight whose interval crossed the 0.10 bound (H1a).
- **exp19 (train).** Ablating L4H14 *raised* the token weight to 0.21–0.25 at every width. The prediction had been that L4H14 is the source of the bias, so this failed, and suggests a second, weaker corrector.
- **exp17 (train).** Two ways of taking up the width. With the width patch applied only to a head's key or only to its value at the newline:
  - L8H15: key +0.04, value +0.29;
  - L17H1: key +0.12, value +0.02.
- **exp20 (train).** The previous line's length is written into the line-start newline token mostly by heads the decision itself depends on. This suggests the counting heads are re-used at the newline to measure the line just ended.
- **exp21 (train, pilot for this protocol).**
  - COUNT4 (L2H7, L4H15, L5H6, L5H7) are the heads in both exp04's critical list and exp20's writers. Replacing them with their newline-position mean at newline tokens only removed:
    - 46% of the width effect (every newline);
    - 61% (the last newline only).
  - Random sets of four heads removed −8% to +3%.
  - The same ablation left 93% of the break/no-break separation on natural text, while ablating COUNT4 everywhere left 5%.
  - With L5H7 ablated at newlines only, the previous line's token count gained a negative weight: −0.077 at width 40 and −0.103 at width 60, against −0.033 and −0.028 intact. A control head (L5H11) changed nothing.
  - **Dropped after the pilot:** a probe-transfer test of the re-use idea. A ridge readout trained at word ends predicted line length at newlines with r = 0.83 from COUNT4, but just as well from some random head sets (up to 0.82), and with a large offset. So it cannot tell the accounts apart.

## Material and model
- **Model:** Pythia-410M (on disk), float32, local GPU.
- **Text:** the WikiText-2 **test** split, read for the first time here. Same paragraph filter, detokenisation, seed-100 order and greedy wrap as round 1 (`code/common.py`, `split_paragraphs`).
- **Samples** (exact rules in each script's docstring):
  - **Token regressions (R1, H3, H6):** the first 60 paragraphs per width that wrap into at least 5 lines and tokenise to at most 700 tokens, at widths 40, 60 and 80. Points are at word ends with at most 30 characters remaining, from line 3 on.
  - **Width items (R2, H4, H5):** up to 150 items at width 60, built exactly as in round 1. The previous line is shortened by at least 10 characters; random choices use seed 200.
  - **Dissociation (H4d):** the first 60 paragraphs at width 60; points that are breaks or have at most 12 characters remaining.
- **Ablation means:** from 30 train paragraphs (seed 6).
  - For head ablations at all positions: the mean over positions, as in round 1.
  - For newline-only ablations: the mean over newline positions in those paragraphs wrapped at widths 40, 60 and 80.

## Hypotheses and predictions (each could fail)

**R1, replication of H1** (`code/c2_01_token.py`). Same measure, conditions and thresholds as `PROTOCOL.md`:
- R1a: intact 95% interval within (−0.10, +0.10);
- R1b: L5H7 ablated ≥ 0.30, with lower bound > 0.20;
- R1c: controls L5H3, L5H11 and L5H14 each within 0.10 of intact.

All three must hold at widths 40, 60 and 80. **What counts against:** any part failing at any width. R1a at width 40 is reported separately, since it is the part that failed in round 1.

**R2, replication of H2** (`code/c2_02_width.py`, a copy of `conf02_width.py` with only the split and file name changed). The thresholds:
- the positive control, a width-effect interval above 0;
- L8H15 mediation share ≥ 25%;
- L8H15, L17H1 and L6H2 together ≥ 50%;
- the mean of 10 random triples < 15%.

**H3, L4H14 as a second token corrector** (`code/c2_01_token.py`).
- **Prediction:** at each width, the token coefficient with L4H14 mean-ablated minus the intact coefficient is ≥ 0.10, and the paired 95% paragraph-bootstrap interval of the difference lies above 0.
- **What counts against:** a difference below 0.10, or an interval reaching 0, at any width.
- **Reported without a threshold:** L4H14 + L5H7 ablated minus L5H7 alone.

**H4, newline-only ablation of the counting heads** (`code/c2_03_newline.py`). COUNT4 heads are replaced by their newline-position mean at newline tokens only. The width effect is measured with the same ablation in both prompts.
- **H4a:** COUNT4 at every newline removes ≥ 25% of the width effect (point estimate), with the 95% item-bootstrap lower bound above 0.
- **H4b:** COUNT4 at the last newline only (the end of the shortened line) removes ≥ 25%, with lower bound above 0.
- **H4c, specificity:** 10 random sets of 4 heads from layers 1–5, outside exp04's eight critical heads, each at every newline, remove < 15% on average.
- **H4d, dissociation:** on natural text, COUNT4 ablated at newlines only keeps ≥ 80% of the intact separation, while COUNT4 ablated at every position keeps < 30%.
- **Positive control:** the intact width effect has a 95% interval above 0.
- **What counts against:** any of H4a–d failing. In particular, the re-use account fails if the newline-only ablation removes little width effect, if random heads do as much, or if the newline-only ablation also destroys the count.

**H5, two ways of taking up the width** (`code/c2_04_paths.py`, eager attention). The width patch is applied only to a head's key, or only to its value, at the newline position.
- **H5a:** for L8H15, value-only minus key-only > 0, with the 95% item-bootstrap interval above 0.
- **H5b:** for L17H1, key-only minus value-only > 0, with the interval above 0.
- **Positive control:** the width effect, re-measured with eager attention, has an interval above 0.
- **Reported without a threshold:** control head L8H11.
- **What counts against:** either interval reaching 0.

**H6, the stored length is token-corrected by L5H7 at the newline** (`code/c2_03_newline.py`).
- **Model:** the newline log-odds regressed on the spline in characters remaining, tokens on the current line, the previous line's character length, and the previous line's token count.
- **H6a:** at widths 40 and 60, the previous-line token coefficient with L5H7 ablated at newlines only, minus intact, is ≤ −0.03, and the paired 95% paragraph-bootstrap interval of the difference lies below 0.
- **H6b, specificity:** at widths 40 and 60, L5H11 ablated at newlines only changes that coefficient by less than 0.02.
- **Width 80** is reported without a verdict (not piloted).
- **What counts against:** H6a or H6b failing at width 40 or 60.

## Analysis
- **Bootstrap:** 2,000 resamples with seed 0, over paragraphs (R1, H3, H6) or items (R2, H4, H5). Paired differences use identical resamples across conditions.
- **Verdicts:** each part is reported as passed or failed against its threshold, with no correction for multiple tests, because the thresholds are fixed in advance. A hypothesis passes only if all its parts pass.
- **Thresholds for the new predictions** (H3–H6) come from the exploratory estimates (exp17, exp19, exp21), set at about half the explored size, or as a sign test.
- **Pipeline checks:** the scripts were checked on the train split with `--smoke` before the freeze. Those outputs (`*_train_smoke.json`) are not results.
- **Tables:** `REPORT-2.md` reports the results, with tables generated from the saved outputs by `code/report2_tables.py` and checked verbatim with `--check`.

## Stopping rules
- One run per script.
- A crash is fixed and re-run, with a dated note in `AMENDMENTS.md`.
- No re-runs to improve a result.
- Results are written up as stated, whichever way they go.
- After this round, D1 stops for the author.

## Compute
Local machine only, about 1–1.5 hours of GPU time, one job at a time, in the memory cage. No downloads, no Modal.

## Order of runs
1. Freeze: digests of this file, the four scripts, `code/common.py` and the test parquet in `FROZEN.md`; frozen files made read-only.
2. `run.sh c2_01_token.py test` (R1, H3).
3. `run.sh c2_02_width.py test` (R2).
4. `run.sh c2_03_newline.py test` (H4, H6).
5. `run.sh c2_04_paths.py test` (H5).
6. Outputs: `outputs/c2_0*_test.json`, written up in `REPORT-2.md`.
