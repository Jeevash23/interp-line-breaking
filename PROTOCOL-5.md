# D1 protocol, round 5: robustness on held-out text

7 November 2026. Written after the round-5 exploration (`AMENDMENTS.md`, Amendment 2; `NOTEBOOK.md`, exp27–exp29 and the train runs of the p5 scripts), before any round-5 confirmation text is read. The digests of this file and of the scripts it names are in `FROZEN.md`. Every test runs once; verdicts are computed by `code/report5_tables.py conf` from the saved outputs, with the thresholds below.

## Why this round

An outside review of the paper draft named four limits, all stated in the draft itself: the circuit is only partly sufficient; a token-count bias remains at width 40; the damping by MLPs 21 and 23 is described but not explained; and only Wikipedia prose at widths 40, 60 and 80 was tested. Round 5 explored each (Amendment 2). This protocol tests what the exploration proposed, on text the exploration did not read.

## Material

- **English:** `reserve-pythia410m`, the unread part of the second new-text batch (200 paragraphs from 116 Wikipedia articles created in 2025–26; `data/newtext2/`, frozen 6 Nov 14:23 UTC).
- **Code:** `conf5-code` (374 paragraphs from 154 files of the Python 3.14 standard library, whitespace collapsed; `data/code/`, frozen 6 Nov 16:40 UTC).
- **German:** `conf5-de` (114 paragraphs from 60 German Wikipedia articles created in 2025–26; `data/de/`, frozen 6 Nov 17:03 UTC).

Ablation means, count-shift class means and resample donors come from WikiText-2 train, as in earlier rounds.

## T: where the token bias comes from

**Account (exploration, exp28).** L5H6 counts by dilution, so each extra token on a line lowers its line-start attention, most of all while the line holds few tokens. The uncorrected per-token bias in the decision is therefore largest on short lines. L5H7 cancels most of it but not all, so a leftover weight remains that grows as lines hold fewer tokens. Width 40 shows it in English because its lines hold about six tokens; code and German, with fewer characters per token, hold more tokens per line and show less.

**Script:** `code/p5_02_tokenbias.py conf`. 60 paragraphs (seed 100) per condition; English at widths 20, 30, 40, 60, 80, 100 and 120; code and German at widths 40 and 60. The token coefficient is that of rounds 1–3. Paragraph bootstraps: 2,000 resamples, numpy seed 0, independent per condition.

| Part | Prediction | Threshold | Explored value (train) |
|---|---|---|---|
| T1 | The leftover weight is larger at width 20 than at width 60 (English) | difference >= 0.04 with lower bound > 0; width-20 lower bound > 0 | 0.101; width 20: 0.158 (0.107, 0.210) |
| T2 | Across the 11 conditions, the leftover weight falls as tokens on the line rise | rank correlation <= -0.50 | -0.87 |
| T3 | The uncorrected bias (L5H7 ablated) is larger at width 20 than at width 100 (English) | difference >= 0.15 with lower bound > 0 | 0.357 |
| T4 | L5H6's line-start attention falls more per token at width 20 than at width 80 (English; spline in characters used) | difference <= -0.003 with upper bound < 0 | -0.0086 |
| T5 | L5H7 corrects the token weight in every condition, code and German included | rise >= 0.10 with lower bound > 0, in all 11 | smallest 0.198 |

## M: the late damping

**Account (exploration, exp29, exp29b).** The damping by MLPs 21 and 23 is not the entropy-neuron route through the final LayerNorm. Two large neurons, MLP23.736 and MLP21.1410, whose output weights push down the newline tokens above all others, fire as a break becomes likely and write directly against the newline logit. This is negative feedback specific to the newline: removing them makes the model more confident about the newline and leaves other predictions almost unchanged.

**Script:** `code/p5_03_damping.py reserve-pythia410m`. Up to 100 items (seed 14, width 60); 30 natural paragraphs (seed 25, width 60). Item bootstraps (paired across conditions) and paragraph bootstraps: 2,000 resamples, numpy seed 0.

| Part | Prediction | Threshold | Explored value (train) |
|---|---|---|---|
| M1 | With the final LayerNorm's statistics frozen, freezing MLPs 21 and 23 still damps both sources about as much | share <= -0.15 with upper bound < 0, and within 0.10 of the share with the LayerNorm free, for COUNT and WIDTH | -0.336 and -0.364 (free: -0.357, -0.373) |
| M2 | Freezing only MLP23.736 and MLP21.1410 gives most of the damping | share <= -0.15 with upper bound < 0 and >= 0.5 of the whole MLPs' share, for both sources | -0.355 and -0.387 (0.99, 1.04 of the whole) |
| M3 | Mean-ablating the two neurons everywhere raises the newline's log-probability where it is the likely next token, and leaves other likely tokens nearly unchanged (top-token probability 0.2–0.9) | newline >= 0.15 with lower bound > 0; other tokens within 0.10 and their interval within ±0.15 | train pilot, by bin: newline +0.15 to +0.53; other -0.003 to -0.054 |
| M4 | Each neuron's activation tracks the newline log-odds at decision points | correlation >= 0.50 for each | 0.837 and 0.780 |

## S: a fuller circuit

**Account (exploration, exp27, exp27b).** The round-4 circuit keeps the ranking of the decision but not its size: with everything else ablated, its log-odds match the intact model's no better than a constant does. A greedy search for fidelity to the intact log-odds found 24 further components that bring most of the size back. They are 13 heads of layers 1–4, mostly attending to earlier lines, plus MLPs 9, 13 and 19 and eight later heads. In order: MLP9, L4H8, L3H4, MLP13, L14H8, L1H13, L20H12, L4H5, L1H14, MLP19, L1H12, L1H9, L21H2, L4H9, L2H5, L9H10, L3H12, L2H8, L3H11, L13H12, L7H15, L11H1, L21H12, L1H1. Random additions of the same composition do almost nothing. The circuit is fuller, not complete; it keeps less under resample ablation.

**Fidelity** is 1 − Σ(circuit − intact)² / Σ(intact − mean)² over the newline log-odds at every decision point.

**Script:** `code/p5_01_sufficiency.py reserve-pythia410m`. Scope as in round 4 (heads of layers 1–5 outside the set ablated at every position; heads and MLPs of layers 6–23 outside the set ablated at decision points); mean and resample ablation. Material: 40 paragraphs (seed 27, width 60) and up to 80 width items (seed 14). Sets: the round-4 circuit; it plus the first 4, 8, 16 and 24 components of the path; 5 random sets of the round-4 circuit plus 24 components of the same composition. Bootstraps over paragraphs and items, 2,000, numpy seed 0, paired across sets.

| Part | Prediction | Threshold | Explored value (train) |
|---|---|---|---|
| S1 | The fuller circuit reproduces most of the decision under mean ablation | fidelity >= 0.50, lower bound > the largest random set's | 0.785 (0.752, 0.810); random at most 0.219 |
| S2 | It keeps clearly more of the separation than the round-4 circuit | difference >= 0.20, lower bound > 0 | 0.376 (0.346, 0.403) |
| S3 | It keeps most of the width effect | >= 0.40, lower bound > the largest random set's | 0.604 (0.523, 0.702); random at most 0.309 |
| S4 | Under resample ablation it still beats the round-4 circuit | fidelity >= 0.15 and separation kept >= 0.25, each lower bound above the round-4 circuit's value | 0.271 and 0.391 (round-4 circuit: -0.088 and 0.202) |

## G: other widths and text types

**Account (exploration, p5_04 runs).** Counting and token correction are general. The counters and L5H7 play their roles at every width from 20 to 120, in Python code and in German. Storing the width at the newline and reading it later is detectable only where the previous line's length clearly moves the decision. The readers hold at English widths 30–120 and German 60, but not at English width 20, in code, or in German at 40. The counters' stored width at the newline is detectable only at width 60 (English, and German). At widths 100 and 120 the newline-only ablation is not specific, because random sets also remove much of the effect, so that test fails there by its own criterion.

**Script:** `code/p5_04_general.py <part> <width>`, on `reserve-pythia410m` at widths 20, 30, 100 and 120, and on `conf5-code` and `conf5-de` at widths 40 and 60 (8 runs). The battery is D1-X's (60 paragraphs; up to 100 items; 5 random sets per test; 2,000 bootstraps). Role sets: counters C = L2H7, L4H15, L5H6, L5H7; corrector T = L5H7; stored width S = the same four heads, ablated at newlines only; readers R = L6H2, L8H15, L17H1, with the newline's residual after layer 5 patched from the short-line prompt.

A role **holds** in a run when:
- **C:** share >= 0.30, lower bound > 0, random mean < 0.15.
- **T:** rise >= 0.10, lower bound > 0, random sets' rise < 0.10.
- **S:** width effect lower bound > 0, share >= 0.15 with lower bound > 0, random mean < 0.10.
- **R:** patch effect lower bound > 0, share >= 0.25 with lower bound > 0, random mean < 0.15.

| Part | Prediction | Threshold |
|---|---|---|
| G1 | Line-breaking works in every condition | AUROC >= 0.80 in all 8 (explored: 0.865–0.987) |
| G2 | C and T hold in every condition | all 16 cells hold (explored: 16 of 16) |
| G3 | The width roles follow the explored pattern. R holds at English 30, 100 and 120 and German 60, and not at English 20, code 40 and 60, or German 40. S holds only in German at 60. | at least 13 of the 16 cells as predicted (explored: 16 of 16) |

## Runs, order and stopping

1. `p5_02_tokenbias.py conf`
2. `p5_03_damping.py reserve-pythia410m`
3. `p5_01_sufficiency.py reserve-pythia410m`
4. The 8 `p5_04_general.py` runs, in the order English 20, 30, 100, 120; code 40, 60; German 40, 60.

Each runs once, under the shared GPU lock and the memory gate. A run that crashes before writing its output may be restarted once with no change. A crash that needs a code change stops that part, and the change is recorded in `AMENDMENTS.md` before the run is repeated. Then `report5_tables.py conf` writes `outputs/report5_tables_conf.md` and `outputs/report5_verdicts_conf.json`, and `REPORT-5.md` reports every part, whichever way it goes. No threshold is changed after this file is frozen.

## Labels and limits

Everything here is **confirmatory** for the account stated in each section. The explored values come from WikiText-2 train and the exploration parts. Thresholds are about half the explored sizes, or explored minimum margins, with no correction for multiple tests across the 16 parts. G3 tests a pattern, so a pass shows that the explored pattern recurs, not that each cell is certain. S tests a circuit chosen by a greedy search on train; its held-out values show how far that choice generalises.
