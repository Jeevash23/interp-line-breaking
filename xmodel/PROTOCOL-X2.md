# D1-X follow-up protocol (CONFIRMATORY): why OLMo-2-1B's reader share exceeds 100%

Written 6 Nov 2026, evening, after the author approved one more D1-X item: explain why freezing OLMo-2-1B's readers removed more than 100% of the stored length's effect (R: 115% on WikiText-2 train, 113% on the confirmation part, `REPORT-olmo1b.md`). Added by Amendment 2 (`AMENDMENTS.md`). Frozen with its script before the held-out text is read; digests in `FROZEN.md`.

## What exploration found (`NOTEBOOK.md`, xm_e06; WikiText-2 train and `part-explore-olmo1b`, 100 items each)
- **Not a baseline artefact.** The reader test freezes the readers to their own base-run values, so no ablation mean is involved. Replacing the freeze with mean-ablation or resample-ablation of the readers (in both prompts) gives 122% and 122% on train (freeze: 118%), and 101% and 107% on the exploration part (freeze: 108%).
- **Not double counting by L6H2.** The three readers are close to additive (L6H2 80% / 70%, L7H10 + L11H4 43% / 41%). L6H2 takes the width through the newline's value, not through its attention share: injecting only L6H2's value at the newline from the patched run gives 85% / 89% of the effect; injecting only its key gives −5% / −4%.
- **The overshoot comes from paths that push the other way.** With the readers frozen, the rest of the network moves the decision against the width (−18% / −8% of the patched effect). Freezing each other head or MLP above layer 3 in addition shows where this comes from. L5H11 is the largest such component in both samples (+26% / +21% of the effect when it is also frozen). MLP11, MLP7, MLP15, L14H15 and L8H7 add 5–12% each. Other heads average 0% (95th percentile ≤ 1%).
- **L5H11 changes sign with context.** Injected alone into the base run, L5H11's patched output moves the decision *with* the width (+14% / +13%). With the readers frozen, its change works *against* the width. L5H11 sits below L6H2, L7H10 and L11H4, so a reading is that L5H11 passes width information that the readers amplify, while its own direct effect is opposite.
- With the current-line tokens held at their base-run states, the patch still reaches the decision (42% / 45% of the effect) and the frozen-reader remainder is just as negative. So the opposing path acts at the decision token itself, not through the tokens of the current line.

## Material
WikiText-2 **validation** split (paragraph order seed 100), up to 150 width items (seed 200), width 60. D1 used this split for Pythia-410M, but no OLMo script and no D1-X script has read it. OLMo-2-1B in bfloat16; readers L6H2, L7H10, L11H4; Lw = 3; marker residual patch, freezes and bootstraps as in `PROTOCOL-X1.md` R and `code/xm_e06_overshoot.py`. Script: `code/xm_c2_overshoot.py validation`, run once. Bootstrap: 2,000 item resamples, seed 0.

## Predictions
- **P1, the overshoot replicates.** Reader mediation share ≥ 1.03, with the lower 95% bound above 1.00.
- **P2, L5H11 is the main opposing path.** With the readers frozen, also freezing L5H11 raises the frozen effect by at least 0.10 of the patched effect, with the lower bound above 0. L5H11 ranks first among all heads and MLPs above layer 3. Ten random heads from layers 4–15 (not readers, not L5H11; numpy seed 0), each frozen in addition, average below 0.05.
- **P3, freezing L5H11 with the readers removes the overshoot.** Mediation share of the readers plus L5H11 ≤ 1.00 (point estimate).
- **P4, the sign change.** L5H11's patched output, injected alone into the base run, moves the decision with the width by at least 0.05 of the patched effect, with the lower bound above 0.
- **P5, not a baseline artefact.** Mean-ablation and resample-ablation versions of the reader share are each within 0.15 of the freeze share.
- **P6, L6H2 reads through the value.** Injecting only L6H2's value at the marker gives ≥ 0.50 of the patched effect (lower bound > 0.30); injecting only its key gives a point estimate within ±0.25.

The account counts as confirmed if P1–P4 pass. P5 and P6 test the two alternative explanations the author raised. Every verdict is reported as frozen, whichever way it goes.

## Limits stated in advance
WikiText-2 is pre-2017 Wikipedia and may be in OLMo's training data; it is held out only in the sense that no OLMo analysis has used it. The account names the components on the opposing path; it does not say what L5H11 computes.
