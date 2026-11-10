# D1 post hoc checks on rounds 1 and 2 (POST HOC, not confirmatory)

6 November 2026, at the author's request. These checks reuse only the splits already read: WikiText-2 validation (round 1) and test (round 2). **They do not change any frozen verdict.** The verdicts in `REPORT.md` and `REPORT-2.md` stand as written.

- **Code:**
  - `code/posthoc_r12_rerun.py` re-runs the ablation measurements with per-paragraph and per-item data kept, and adds resample ablation.
  - `code/posthoc_holm.py` computes the p-values, the Holm correction and the resample table.
- **Checks:** the tables below are generated from the saved outputs, and `posthoc_holm.py --check` verifies they appear here verbatim. The mean-ablation re-run reproduces the frozen point estimates to within 2 × 10⁻⁸.

## 1. Holm correction across all pass/fail parts of rounds 1 and 2
- **The tests.** Each frozen part is turned into a one-sided bootstrap test, with 10,000 resamples:
  - where the frozen criterion is an interval bound, that bound is the null boundary (for example H1b: the coefficient ≤ 0.20);
  - where the criterion is only a point threshold, the threshold is the null boundary (H2 shares, H4c, H4d);
  - "within a margin" parts (H1a, H1c, H6b) are two one-sided tests, taking the larger p.
- **The correction.** Holm, at a family-wise α of 0.025, the one-sided level of the frozen 95% intervals. With 10,000 resamples the smallest possible p is about 0.0001. This makes the smallest adjusted p 0.0054 for 54 tests, which is a floor rather than an estimate.

Tests in the family: 54. Holm at family-wise alpha 0.025 (one-sided). Largest difference between the mean-ablation re-run and the frozen outputs: 1.93e-08.

| Round | Part | One-sided p | Holm-adjusted p | Holds after Holm |
|---|---|---|---|---|
| 1 | H1a, width 40 | 0.1081 | 0.1081 | no |
| 1 | H1b, width 40 | 0.0001 | 0.0054 | yes |
| 1 | H1c L5H3, width 40 | 0.0001 | 0.0054 | yes |
| 1 | H1c L5H11, width 40 | 0.0001 | 0.0054 | yes |
| 1 | H1c L5H14, width 40 | 0.0001 | 0.0054 | yes |
| 1 | H1a, width 60 | 0.0013 | 0.0054 | yes |
| 1 | H1b, width 60 | 0.0001 | 0.0054 | yes |
| 1 | H1c L5H3, width 60 | 0.0001 | 0.0054 | yes |
| 1 | H1c L5H11, width 60 | 0.0001 | 0.0054 | yes |
| 1 | H1c L5H14, width 60 | 0.0001 | 0.0054 | yes |
| 1 | H1a, width 80 | 0.0001 | 0.0054 | yes |
| 1 | H1b, width 80 | 0.0001 | 0.0054 | yes |
| 1 | H1c L5H3, width 80 | 0.0001 | 0.0054 | yes |
| 1 | H1c L5H11, width 80 | 0.0001 | 0.0054 | yes |
| 1 | H1c L5H14, width 80 | 0.0001 | 0.0054 | yes |
| 1 | H2 positive control | 0.0001 | 0.0054 | yes |
| 1 | H2 L8H15 >= 0.25 | 0.0001 | 0.0054 | yes |
| 1 | H2 readers3 >= 0.50 | 0.0001 | 0.0054 | yes |
| 1 | H2 random mean < 0.15 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1a, width 40 | 0.0324 | 0.0648 | no |
| 2 (R1) | H1b, width 40 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1c L5H3, width 40 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1c L5H11, width 40 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1c L5H14, width 40 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1a, width 60 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1b, width 60 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1c L5H3, width 60 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1c L5H11, width 60 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1c L5H14, width 60 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1a, width 80 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1b, width 80 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1c L5H3, width 80 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1c L5H11, width 80 | 0.0001 | 0.0054 | yes |
| 2 (R1) | H1c L5H14, width 80 | 0.0001 | 0.0054 | yes |
| 2 (R2) | H2 positive control | 0.0001 | 0.0054 | yes |
| 2 (R2) | H2 L8H15 >= 0.25 | 0.0001 | 0.0054 | yes |
| 2 (R2) | H2 readers3 >= 0.50 | 0.0001 | 0.0054 | yes |
| 2 (R2) | H2 random mean < 0.15 | 0.0001 | 0.0054 | yes |
| 2 | H3, width 40 | 0.0001 | 0.0054 | yes |
| 2 | H3, width 60 | 0.0001 | 0.0054 | yes |
| 2 | H3, width 80 | 0.0001 | 0.0054 | yes |
| 2 | H4 positive control | 0.0001 | 0.0054 | yes |
| 2 | H4a | 0.0001 | 0.0054 | yes |
| 2 | H4b | 0.0001 | 0.0054 | yes |
| 2 | H4c | 0.0001 | 0.0054 | yes |
| 2 | H4d, newline-only kept >= 0.80 | 0.0004 | 0.0054 | yes |
| 2 | H4d, all-position kept < 0.30 | 0.0001 | 0.0054 | yes |
| 2 | H5 positive control | 0.0001 | 0.0054 | yes |
| 2 | H5a | 0.0001 | 0.0054 | yes |
| 2 | H5b | 0.0001 | 0.0054 | yes |
| 2 | H6a, width 40 | 0.0001 | 0.0054 | yes |
| 2 | H6b, width 40 | 0.0001 | 0.0054 | yes |
| 2 | H6a, width 60 | 0.0001 | 0.0054 | yes |
| 2 | H6b, width 60 | 0.0001 | 0.0054 | yes |

Parts holding after Holm: 52 of 54

**Reading.**
- 52 of the 54 parts hold after correction.
- The two that do not are H1a at width 40 in both rounds, which had already failed as frozen. That part says the intact model gives tokens no weight.
- Every mechanism-level part holds after correction. This includes the parts that had only point thresholds when frozen (the H2 shares, H1c, H4c, H4d and H6b), now tested as hypotheses.

## 2. Resample ablation in place of mean ablation
- **What changes.** A head is ablated by copying its output from a forward pass on a *different* paragraph, a WikiText-2 train paragraph wrapped at the same width:
  - for all-position ablation, position by position;
  - for newline-only ablation, newline by newline.
- **The same donor** is used for every condition of a target paragraph, and for both prompts of a width item.
- **The criteria** are the frozen ones, applied post hoc.

| Split | Measure | Mean ablation | Resample ablation | Frozen criterion met with resample |
|---|---|---|---|---|
| validation | token coefficient, L5H7 ablated, width 40 | 0.584 (0.516, 0.664) | 0.485 (0.422, 0.558) | yes |
| validation | token coefficient, L4H14 ablated, width 40 | 0.201 (0.165, 0.242) | 0.175 (0.139, 0.215) | yes |
| validation | largest control change (L5H3, L5H11, L5H14), width 40 | 0.006 | 0.005 | yes |
| validation | token coefficient, L5H7 ablated, width 60 | 0.613 (0.542, 0.692) | 0.433 (0.376, 0.501) | yes |
| validation | token coefficient, L4H14 ablated, width 60 | 0.241 (0.204, 0.280) | 0.186 (0.145, 0.229) | yes |
| validation | largest control change (L5H3, L5H11, L5H14), width 60 | 0.002 | 0.001 | yes |
| validation | token coefficient, L5H7 ablated, width 80 | 0.468 (0.408, 0.547) | 0.342 (0.291, 0.404) | yes |
| validation | token coefficient, L4H14 ablated, width 80 | 0.217 (0.180, 0.257) | 0.153 (0.114, 0.194) | yes |
| validation | largest control change (L5H3, L5H11, L5H14), width 80 | 0.001 | 0.002 | yes |
| validation | width effect removed, COUNT4 at every newline | 0.500 (0.450, 0.548) | 0.385 (0.324, 0.440) | yes |
| validation | width effect removed, COUNT4 at last newline | 0.616 (0.572, 0.660) | 0.376 (0.319, 0.430) | yes |
| validation | width effect removed, random sets, mean | -0.021 | -0.013 | yes |
| validation | separation kept, COUNT4 newline-only / every position | 0.854 / 0.042 | 0.913 / 0.044 | yes |
| validation | previous-line token weight, L5H7 at newlines minus intact, width 40 | -0.043 (-0.055, -0.031) | -0.033 (-0.050, -0.016) | yes |
| validation | previous-line token weight, L5H7 at newlines minus intact, width 60 | -0.059 (-0.072, -0.045) | -0.048 (-0.061, -0.035) | yes |
| validation | previous-line token weight, L5H7 at newlines minus intact, width 80 | -0.058 (-0.070, -0.047) | -0.045 (-0.058, -0.033) | yes |
| test | token coefficient, L5H7 ablated, width 40 | 0.620 (0.549, 0.696) | 0.481 (0.419, 0.550) | yes |
| test | token coefficient, L4H14 ablated, width 40 | 0.205 (0.158, 0.255) | 0.195 (0.149, 0.245) | yes |
| test | largest control change (L5H3, L5H11, L5H14), width 40 | 0.004 | 0.002 | yes |
| test | token coefficient, L5H7 ablated, width 60 | 0.612 (0.542, 0.686) | 0.415 (0.353, 0.479) | yes |
| test | token coefficient, L4H14 ablated, width 60 | 0.257 (0.220, 0.292) | 0.188 (0.148, 0.229) | yes |
| test | largest control change (L5H3, L5H11, L5H14), width 60 | 0.004 | 0.004 | yes |
| test | token coefficient, L5H7 ablated, width 80 | 0.525 (0.471, 0.586) | 0.368 (0.323, 0.420) | yes |
| test | token coefficient, L4H14 ablated, width 80 | 0.224 (0.194, 0.257) | 0.154 (0.120, 0.188) | yes |
| test | largest control change (L5H3, L5H11, L5H14), width 80 | 0.002 | 0.001 | yes |
| test | width effect removed, COUNT4 at every newline | 0.464 (0.420, 0.510) | 0.356 (0.306, 0.405) | yes |
| test | width effect removed, COUNT4 at last newline | 0.613 (0.574, 0.653) | 0.422 (0.372, 0.469) | yes |
| test | width effect removed, random sets, mean | -0.012 | -0.008 | yes |
| test | separation kept, COUNT4 newline-only / every position | 0.863 / 0.078 | 0.902 / 0.036 | yes |
| test | previous-line token weight, L5H7 at newlines minus intact, width 40 | -0.050 (-0.060, -0.040) | -0.041 (-0.051, -0.032) | yes |
| test | previous-line token weight, L5H7 at newlines minus intact, width 60 | -0.075 (-0.091, -0.058) | -0.060 (-0.076, -0.041) | yes |
| test | previous-line token weight, L5H7 at newlines minus intact, width 80 | -0.067 (-0.078, -0.056) | -0.065 (-0.078, -0.050) | yes |

**Reading.**
- Every key result keeps its frozen criterion under resample ablation, on both splits:
  - L5H7's token correction;
  - the specificity of the controls;
  - L4H14 as a second corrector;
  - the newline-only ablation of the counting heads;
  - the dissociation;
  - L5H7's correction of the stored length.
- The effects are mostly smaller than with mean ablation:
  - the L5H7 token coefficient falls from 0.47–0.62 to 0.34–0.49;
  - the share of the width effect removed at the last newline falls from 0.61–0.62 to 0.38–0.42.
- So part of the mean-ablation size comes from moving the newline token's state to an average that no real newline has. The direction and specificity of every result do not depend on the ablation method.
