# D1 confirmation protocol (CONFIRMATORY)

Drafted 6 Nov 2026 (day 2) after exploration on the train split. The author approved freezing it on 6 Nov 2026. Its SHA-256 digest, with those of the scripts, the shared helper and the validation data, is in `FROZEN.md`; all were recorded before any run on the validation split. Later changes are dated entries in `AMENDMENTS.md`.

## Background (exploratory, from `NOTEBOOK.md`)
In Pythia-410M, the newline decision in fixed-width wrapped prose depends on characters remaining, not on tokens used. Exploration on the WikiText-2 train split (exp18, exp19) found the following:
- With attention head L5H7 mean-ablated, the token count gains a large positive weight.
- Ablating control heads in layer 5 changes nothing.
- Exploration also suggested that the width reaches the decision through heads L8H15, L17H1 and L6H2 at the decision token.

## Material and model
- **Model:** Pythia-410M (on disk), float32, local GPU.
- **Text:** the WikiText-2 **validation** split, read for the first time here. Paragraphs of 600–1,800 characters, using the same filters and detokenisation as `code/common.py`. Greedy wrap at widths 40, 60 and 80.
- **H1 sample:** the first 60 paragraphs per width, in seed-100 order, that wrap into at least 5 lines and tokenise to at most 700 tokens. Points are at word ends with at most 30 characters remaining, from line 3 on.
- **H2 sample:** 150 items at width 60, one per paragraph in seed-100 order, built as described in the docstring of `code/conf02_width.py`.
- **Ablation means:** taken from 30 train paragraphs (seed 6), as in exploration.
- **The test split stays unread.**

## Hypotheses and predictions (each could fail)
- **H1, token correction.** Measure: the coefficient of tokens-on-the-line in the newline log-odds, given a linear spline in characters remaining (knots 2, 5, 8, 12, 16, 24, 32). Ablation is by the mean over positions, as in exp18.
  - **H1a.** Intact model: the 95% paragraph-bootstrap interval for the coefficient lies within (−0.10, +0.10), at each width.
  - **H1b.** L5H7 ablated: the coefficient is at least 0.30, and the lower 95% bound is above 0.20, at each width.
  - **H1c, specificity.** For each of three control heads from layer 5 (L5H3, L5H11, L5H14), the coefficient differs from the intact one by less than 0.10.
- **H2, width readers.** Same construction as exp14: the previous line shortened by at least 10 characters, and the newline state after layer 5 patched; 150 items at width 60.
  - Freezing L8H15 at the decision token (point estimate of the mediation share) removes at least 25% of the width effect.
  - Freezing L8H15, L17H1 and L6H2 together removes at least 50%.
  - Freezing three random heads from layers 6–23 together (10 random draws, seeded) removes less than 15% on average.
- **Positive control.** The total width effect is positive, with a 95% interval above 0. Otherwise H2 is not assessable and is reported as such.

## Analysis
- **Bootstrap:** over paragraphs (H1) or items (H2), with 2,000 resamples and seed 0.
- **Verdicts:** each prediction is reported as passed or failed against its threshold, at each width, with no correction because the thresholds are fixed in advance. H1 passes only if H1a, H1b and H1c all pass at all three widths.
- **Code:** `code/conf01_token.py` (H1) and `code/conf02_width.py` (H2), with `code/common.py` (`split_paragraphs`), written and digested before the run. Their docstrings give the exact construction. They were pipeline-checked on the train split with `--smoke` (5 paragraphs or 8 items); those runs are not results.
- **Interval for H2:** shares are point estimates against the thresholds; 95% item-bootstrap intervals are reported alongside.

## Stopping rules
One run per script. A crash is fixed and re-run with a dated note in `AMENDMENTS.md`. No re-runs to improve a result. Whichever way the results go, they are written up as stated.

## Compute
Local machine, about 1 hour of GPU time. No downloads.

## Order of runs
1. Freeze (digests in `FROZEN.md`).
2. `run.sh conf01_token.py validation`.
3. `run.sh conf02_width.py validation`.
4. Results go to `outputs/conf01_token_validation.json` and `outputs/conf02_width_validation.json`, written up as stated, whichever way they come out.
