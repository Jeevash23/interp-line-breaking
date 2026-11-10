# D1 report, round 4: the decision step, sufficiency, geometry and GPT-2

6 November 2026. **Confirmation** under `PROTOCOL-4.md`, frozen at 14:23 UTC after the round-4 exploration (`AMENDMENTS.md`, Amendment 1; `NOTEBOOK.md`, exp22–exp26). The text is a second batch of English Wikipedia articles created between January 2025 and September 2026. It shares no article with the first batch, was fetched for this round, and was read here for the first time. This round uses its `conf4-pythia410m` part: 208 paragraphs from 108 articles (`data/newtext2/MANIFEST.md`). The `reserve-pythia410m` part is still unread. Each script ran once, between 19:53 and 14:30 UTC, with no crash and no re-run. Tables and verdicts come from `code/report4_tables.py` (frozen), and `--check` verifies that they appear here verbatim. The last section, on training, is **exploratory** and was not part of the protocol.

## Summary

| Group | Claim tested on new text | Result |
|---|---|---|
| D | The decision is assembled across layers 6–20: the width is written onto the count's direction, L17H1 is the main direct writer, MLPs 14, 16 and 18 add to the result and MLPs 21 and 23 damp it | pass (D1–D4) |
| S | The circuit is partly sufficient: the late part keeps the ranking and most of the width effect; the whole circuit keeps far more than random circuits of the same size | pass (S1, S2) |
| G | The count lies on a low-dimensional curve with one dominant linear axis; the readers ramp towards the edge, with no head-specific offsets | pass (G1–G3) |
| X | GPT-2 Small has a decodable count but uses it weakly, has no strong counting head, and loses the stored width early | pass (X1–X5) |

## D: the decision step
**D, the decision step.** Items: 100. Effects on log P(newline): COUNT 1.852, WIDTH 0.655, both 1.967 (interaction -0.540).

| Part | Measure | Value (95% CI) | Threshold | Verdict |
|---|---|---|---|---|
| D1 | COUNT share lost, MLPs 14, 16, 18 frozen | 0.215 (0.203, 0.228) | >= 0.11, lower bound > 0 | pass |
| D1 | minus the mean of 10 random MLP triples (0.033) | 0.182 (0.172, 0.192) | lower bound > 0 | pass |
| D2 | COUNT share lost, MLPs 21, 23 frozen | -0.358 (-0.404, -0.315) | <= -0.10, upper bound < 0 | pass |
| D2 | WIDTH share lost, MLPs 21, 23 frozen | -0.442 (-0.552, -0.344) | <= -0.10, upper bound < 0 | pass |
| D3a | cosine after layer 12 minus after layer 6 (0.801 - 0.322) | 0.479 (0.407, 0.514) | >= 0.20, lower bound > 0 | pass |
| D3b | cosine after layer 12 | 0.801 | >= 0.65 | pass |
| D4 | largest head direct effect, COUNT: L17H1 0.917 (next L15H10 0.233) | | L17H1 | pass |
| D4 | largest head direct effect, WIDTH: L17H1 0.411 (next L8H15 0.125) | | L17H1 | pass |

D overall: pass

Cosine between the COUNT and WIDTH residual changes at the decision token, by layer: L5 0.00, L6 0.32, L7 0.35, L8 0.49, L9 0.62, L10 0.68, L11 0.76, L12 0.80, L13 0.81, L14 0.89, L15 0.86, L16 0.88, L17 0.82, L18 0.85, L19 0.86, L20 0.87, L21 0.76, L22 0.75, L23 0.76.

| Frozen at the decision token | COUNT share lost | WIDTH share lost |
|---|---|---|
| layers6-11 | 0.375 | 0.569 |
| layers12-17 | 0.435 | 0.481 |
| layers18-23 | -0.259 | -0.129 |
| all_heads | 0.437 | 1.000 |
| all_mlps | 0.074 | -0.240 |
| readers3 | 0.339 | 0.789 |
| L17H1 | 0.106 | 0.281 |
| MLP14_16_18 | 0.215 | 0.119 |
| MLP21_23 | -0.358 | -0.442 |
| random_mlp3_mean | 0.033 | -0.005 |

- **No single comparison site.** Freezing each block of layers at the decision token removes part of both effects: layers 6–11 take 37% of the COUNT effect and 57% of the WIDTH effect, and layers 12–17 take 43% and 48%. Freezing layers 18–23 makes both effects larger.
- **The width is written onto the count's direction (D3).** At the decision token, the residual changes caused by the two inputs are nearly orthogonal after layer 6 (cosine 0.32). They align steadily through the readers' layers and beyond, to 0.80 after layer 12 and 0.89 after layer 14. In effect, by the middle of the network the two inputs are one quantity, which behaves like characters remaining.
- **Who writes the decision (D1, D2, D4).**
  - L17H1 is the largest direct writer of both inputs to the newline logit.
  - MLPs 14, 16 and 18 carry 21% of the COUNT effect, against 3% for random MLP triples.
  - MLPs 21 and 23 push against both inputs: freezing them makes the COUNT effect 36% larger and the WIDTH effect 44% larger. This late damping fits the model's close calibration (`NOTEBOOK.md`, exp02).
- The two inputs combine sub-additively (−0.54 against their sum), partly because log P(newline) saturates near the edge. That was reported without a prediction.

## S: is the circuit enough?
The circuit, from exploration:
- early heads L2H7, L2H11, L2H13, L4H10, L4H14, L4H15, L5H6 and L5H7;
- late heads L6H2, L6H11, L8H15, L15H10, L17H1, L19H0 and L20H6;
- all MLPs of layers 0–5, and MLPs 6, 14, 15, 16, 18, 21 and 23.

Everything else is ablated:
- DEC: everything outside the circuit in layers 6–23, at decision points only;
- EARLY: every head of layers 1–5 outside the circuit, at all positions;
- BOTH: the two together.

**S, sufficiency.** Paragraphs: 40; width items: 80. Intact separation 3.067, AUROC 0.978, width effect 1.651.

| Part | Measure | Value (95% CI) | Threshold | Verdict |
|---|---|---|---|---|
| S1a | AUROC, late circuit only at the decision token (mean ablation) | 0.976 | >= 0.958 | pass |
| S1b | width effect kept, same | 0.727 (0.663, 0.804) | >= 0.33, lower bound > 0.280 (largest random circuit) | pass |
| S2 | separation kept, whole circuit (mean ablation) | 0.304 (0.279, 0.328) | >= 0.15, lower bound > 0.163 (largest random circuit) | pass |

S overall: pass

| Scope | Ablation | Separation kept | AUROC | Width effect kept |
|---|---|---|---|---|
| DEC | mean | 0.529 | 0.976 | 0.727 |
| DEC | resample | 0.520 | 0.966 | 0.674 |
| DEC | 5 random circuits, mean ablation | 0.186 to 0.514 | | 0.169 to 0.280 |
| EARLY | mean | 0.312 | 0.941 | 0.304 |
| EARLY | resample | 0.133 | 0.862 | 0.285 |
| EARLY | 5 random circuits, mean ablation | -0.021 to 0.020 | | 0.070 to 0.119 |
| BOTH | mean | 0.304 | 0.946 | 0.248 |
| BOTH | resample | 0.172 | 0.845 | 0.409 |
| BOTH | 5 random circuits, mean ablation | -0.016 to 0.163 | | -0.002 to 0.047 |

- **The late circuit keeps the ranking (S1).**
  - With 7 of 288 heads and 7 of 18 MLPs left in layers 6–23 at the decision token, AUROC is unchanged and 73% of the width effect survives, against at most 28% for random circuits of the same size.
  - About half of the separation survives, a level random circuits can also reach (19–51%). So the circuit does not explain the full scale of the decision: amplification is spread more widely.
- **The early part needs other early heads (S2).**
  - Keeping only the eight early circuit heads of 80 leaves 31% of the separation. Random sets of the same size leave about 0.
  - The rest presumably comes from generic machinery the account takes for granted (heads that assemble words and track previous tokens). The whole circuit keeps 30% of the separation and AUROC 0.95.
  - Resample ablation gives lower separation for the early scopes (13–17%). For the late scope it agrees with mean ablation.

## G: the shape of the count
**G, the shape of the count** (split-half; half B's class means in half A's top directions).

| Width | After layer | Top direction: variance share | its linear R^2 | Second direction: linear / spline R^2 | its reliability | Overall linear R^2, labels shuffled |
|---|---|---|---|---|---|---|
| 40 | 5 | 0.738 | 0.990 | 0.009 / 0.568 | 0.858 | 0.010 |
| 40 | 6 | 0.755 | 0.913 | 0.074 / 0.935 | 0.966 | 0.007 |
| 60 | 5 | 0.689 | 0.945 | 0.001 / 0.619 | 0.857 | 0.016 |
| 60 | 6 | 0.707 | 0.811 | 0.100 / 0.896 | 0.954 | 0.026 |

| Width | Head | Peak at characters remaining | Monotone fit R^2 |
|---|---|---|---|
| 40 | L6H2 | 0 | 0.997 |
| 40 | L8H15 | 2 | 0.993 |
| 40 | L17H1 | 0 | 0.997 |
| 40 | L5H6 | 37 | 1.000 |
| 40 | L5H7 | 27 | 0.973 |
| 40 | L8H11 | 0 | 0.976 |
| 60 | L6H2 | 1 | 0.998 |
| 60 | L8H15 | 1 | 0.999 |
| 60 | L17H1 | 1 | 0.996 |
| 60 | L5H6 | 49 | 1.000 |
| 60 | L5H7 | 40 | 0.913 |
| 60 | L8H11 | 0 | 0.681 |

G1 pass, G2 pass, G3 pass; G overall: pass

- **G1, G2.** After layer 5 the count is mostly one direction:
  - it carries 69–74% of the held-out between-class variance, and a straight line in characters used explains 94–99% of it;
  - a reliable second direction is curved, and after layer 6 a spline explains 90–94% of it against 7–10% for a line.
  - So the count lies on a low-dimensional bent curve. It is not the multi-turn, helix-like manifold reported for Claude 3.5 Haiku.
- **G3.** The three readers' attention to the line start rises monotonically as the line fills and peaks at 0–2 characters remaining, at both widths. They do not tile the distance to the edge with different offsets, as the Haiku paper's boundary heads do. They behave as ramps aimed at the edge. The counting heads L5H6 and L5H7 do the opposite: their attention to the line start falls as the line fills, the dilution count.

## X: what GPT-2 Small lacks
**X, GPT-2 Small against Pythia-410M.**

| Part | Measure | GPT-2 Small | Pythia-410M | Threshold | Verdict |
|---|---|---|---|---|---|
| X1 | best-layer probe R^2, characters used | 0.809 | 0.915 | GPT-2 >= 0.70 | pass |
| X2 | largest count-shift effect over layers | 0.630 | 2.267 | GPT-2 <= 0.5 x Pythia | pass |
| X3 | behavioural width effect | 0.317 | 1.465 | GPT-2 <= 0.5 x Pythia | pass |
| X4 | largest line-start share among counting heads | 0.277 | 0.792 | GPT-2 <= 0.5 x Pythia; Pythia >= 0.60 | pass |
| X5 | width patch at the newline: GPT-2 after layers 8-11 (largest absolute), Pythia after layer 8 | 0.047 | 0.352 | GPT-2 < 0.10; Pythia >= 0.25 | pass |

X overall: pass

GPT-2 Small: AUROC width 40 0.807, width 60 0.852; token coefficient -0.022 / 0.015; width items 100.
Pythia-410M: AUROC width 40 0.968, width 60 0.978; token coefficient 0.039 / 0.021; width items 100.

- **GPT-2 has the count but uses it weakly (X1, X2).** Characters used are almost as decodable as in Pythia (R² 0.81 against 0.92), but shifting the count by six characters moves the decision by 0.63 against 2.27.
- **It has no strong counting head (X4).** Its best counting head puts 28% of its attention on the line start; Pythia's L5H6 puts 79% there.
- **It loses the stored width early (X5) and responds weakly to it (X3).** The newline token carries the previous line's length only in GPT-2's early layers; after layer 8 a patch there does nothing. The behavioural width effect is a fifth of Pythia's.
- The tests do not separate the possible causes: model size, GPT-2's absolute position embeddings, or its training data.

## Training dynamics (EXPLORATORY; not part of the protocol)
Six intermediate checkpoints of Pythia-410M (911 MB each, about 5.5 GB, downloaded with the author's approval), measured with the component indices of the final model on WikiText-2 train text at width 60 (`code/exp26_checkpoints.py`; table generated by `code/report4_training_table.py`).

| Step | AUROC | Separation | L5H6 share to line start | Separation kept, COUNT4 ablated | L5H7 correction | Width share removed: COUNT4 at newlines | readers | MLPs 14, 16, 18 | MLPs 21, 23 |
|---|---|---|---|---|---|---|---|---|---|
| 1k | 0.936 | 1.46 | 0.73 | 0.14 | -0.003 | 0.22 | 0.15 | 0.03 | -0.05 |
| 2k | 0.957 | 1.96 | 0.81 | 0.11 | 0.020 | 0.34 | 0.23 | 0.01 | -0.02 |
| 4k | 0.964 | 2.65 | 0.86 | 0.08 | 0.108 | 0.38 | 0.31 | 0.01 | -0.00 |
| 8k | 0.971 | 2.64 | 0.83 | 0.05 | 0.176 | 0.20 | 0.34 | 0.03 | -0.01 |
| 16k | 0.964 | 2.75 | 0.88 | 0.03 | 0.291 | 0.28 | 0.37 | 0.07 | -0.01 |
| 48k | 0.975 | 2.92 | 0.82 | 0.06 | 0.423 | 0.12 | 0.42 | 0.14 | -0.33 |
| 143k (final) | 0.978 | 3.26 | 0.81 | 0.08 | 0.546 | 0.25 | 0.54 | 0.11 | -0.40 |

- **Order of appearance** (one training run, a coarse grid of steps):
  1. The dilution counter L5H6, the dependence on the counting heads and the stored width are all present at step 1,000, under 1% of training. Line-breaking already works then (AUROC 0.94).
  2. The readers then strengthen steadily, carrying 15% of the width effect at step 1,000 and 54% at the end.
  3. The L5H7 corrector grows steadily from nothing at step 2,000 to 0.55 at the end, while the intact token weight stays near 0. So the token bias it cancels grows during training, and the correction keeps pace.
  4. The late MLP roles appear last, between steps 16,000 and 48,000: MLPs 14, 16 and 18 as writers, and MLPs 21 and 23 as dampers.
- AUROC levels off early, while the separation keeps growing (2.6 at step 4,000, 3.3 at the end). That fits late components that sharpen the scale of the decision rather than its ranking.

## Limits
- One model in depth, with greedy wrapping of plain prose at widths 40 and 60.
- Ablation and patching only. The sufficiency test uses a hand-built circuit and three fixed scopes, and its separation result is not specific at the late scope.
- Thresholds were set at about half the explored sizes on WikiText-2 train, with no correction for multiple tests.
- The geometry comparison with Claude 3.5 Haiku rests on class-mean curves and attention profiles, not on the Haiku paper's own feature analysis.
- The training-dynamics results are exploratory, from one run and seven checkpoints, and label components by their final-model roles.

## Corrections
- 6 Nov 2026, late evening: four percentages in the prose were rounded by eye and were one point off. Corrected from the saved outputs, as found by the paper's number check (`paper/v1-2026-11-06/check_paper.py`): layers 6–11 take 37% (not 38%) and layers 12–17 43% (not 44%) of the COUNT effect; MLPs 14, 16 and 18 carry 21% (not 22%); the straight line explains 94–99% (not 95–99%). The tables, which a script checks, were already right, and no verdict changes.
