# D1 report: how Pythia-410M decides where to break a line

6 November 2026. **Exploration** under `CHARTER.md` (frozen 19:33 UTC the previous day) used the WikiText-2 train split, in experiments exp01–exp19 (`NOTEBOOK.md`). **Confirmation** under `PROTOCOL.md` (frozen 02:26 UTC; digests in `FROZEN.md`) used the validation split, read for the first time. The test split is unread. The tables in the confirmation section are generated from saved outputs by `code/report_tables.py`, and `--check` verifies that they appear here verbatim.

## Behaviour (exploratory)
- Text is greedily wrapped to a fixed width. At word ends, Pythia-410M predicts the line break with 87–90% accuracy at widths 40–80 (AUROC 0.97–0.985).
- Its newline probability follows a calibrated hazard on characters remaining.
- It counts characters, not tokens or bytes.
- It takes the width from several recent lines; the previous line carries about 40% of the full shift.
- GPT-2 Small does this poorly (AUROC 0.85 at width 60), and Pythia-1B is no better than 410M.

## Candidate circuit (exploratory)
1. **Counting at layers 4–5.**
   - The count that drives the decision is first written at the decision token by layer 5. Shifting the count before layer 5 has no effect; from layer 5 on, it moves the decision as predicted.
   - L5H6 counts by line-start dilution: each token draws attention roughly in proportion to its character length, so the attention left on the line-start newline falls as the line fills. Editing that attention in L5H6 or L5H7 moves the decision smoothly; a control head does nothing.
2. **Token correction.**
   - L5H6's count carries a bias with tokens: more, shorter tokens read as more characters.
   - L5H7 removes the bias; L4H14 does so more weakly.
   - This explains why a dotted re-spelling fools L5H6 and L5H7 in opposite directions while the behaviour barely moves.
3. **Width readers.**
   - The previous line's length is held in the line-start newline token's state after layer 5.
   - Three heads at the decision token, L8H15, L17H1 and L6H2, attend to that newline more as the line fills; the count enters through their queries.
   - L17H1 picks up the width through the newline's key (a comparison, like Haiku's "boundary heads"); L8H15 copies it from the newline's value.
   - MLPs 6, 14 and 16 also read the count. MLP23 damps both the count and the width.

## Confirmation on held-out text (`PROTOCOL.md`)

**H1, token correction by L5H7.** The measure is the coefficient of tokens-on-the-line in the newline log-odds, given a spline in characters remaining, with 95% paragraph-bootstrap intervals and 60 paragraphs per width. The thresholds:
- H1a: intact interval within ±0.10;
- H1b: L5H7 ablated ≥ 0.30, with lower bound > 0.20;
- H1c: each control within 0.10 of intact.

| Width | Points | Intact | L5H7 ablated | L5H3 ablated | L5H11 ablated | L5H14 ablated | H1a | H1b | H1c |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 6819 | 0.074 (0.035, 0.113) | 0.584 (0.516, 0.658) | 0.077 (0.038, 0.115) | 0.068 (0.028, 0.108) | 0.072 (0.033, 0.112) | fail | pass | pass |
| 60 | 4162 | 0.034 (-0.006, 0.078) | 0.613 (0.545, 0.693) | 0.036 (-0.005, 0.080) | 0.034 (-0.007, 0.078) | 0.033 (-0.007, 0.077) | pass | pass | pass |
| 80 | 2950 | 0.011 (-0.029, 0.053) | 0.468 (0.409, 0.546) | 0.012 (-0.027, 0.052) | 0.011 (-0.030, 0.053) | 0.010 (-0.029, 0.052) | pass | pass | pass |

H1 overall: fail

**H1 fails as frozen, on one of its three parts.**
- The specific prediction holds at every width: ablating L5H7, and no control head, makes token count weigh 0.47–0.61 log-odds per token (H1b, H1c).
- But at width 40 the intact model keeps a small token weight of its own, whose interval crosses the frozen ±0.10 bound (H1a).
- So the correction is strong but not complete at narrow widths.

**H2, width readers.** Items are 150 at width 60. The previous line is at least 10 characters shorter, and the newline state after layer 5 is patched. The thresholds: L8H15 ≥ 25%, the three readers ≥ 50%, random triples < 15%.

Width effect (n = 150): 0.631 (0.527, 0.729)

| Frozen at the decision token | Mediation share (95% CI) |
|---|---|
| L8H15 | 0.494 (0.462, 0.527) |
| L8H15, L17H1, L6H2 | 0.816 (0.788, 0.848) |
| 10 random triples, mean | 0.005 |

H2 overall: pass

**H2 passes.** Most of the previous line's influence on the break passes through three heads at the decision token, half of it through L8H15 alone.

## What failed during exploration (kept on record)
- A "punctuation illusion" predicted from the dilution reading (exp10). It failed because L5H7 cancels it.
- A reading of L5H7 as a ratio of the current line to the previous line (exp12).
- Re-spellings becoming token-sensitive once L5H7 is ablated (exp18 P1): they over-count by a fixed amount instead.
- L4H14 as the source of the token bias (exp19 P3).
- Pythia-1B's head-edit test was not specific (exp15).

## Limits
- One model, one corpus (WikiText-2), and greedy wrapping of plain prose.
- Mean-ablation and patching only; no training-dynamics data.
- The confirmation is held-out text, not a new model.
- Thresholds were set from exploration on the same model.

## Next
See `NOTEBOOK.md` and `../STATUS.md`. The decisions taken while the author was away are in `../DECISIONS.md`.
