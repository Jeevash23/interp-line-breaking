# D1 fourth confirmation protocol (CONFIRMATORY): the decision step, sufficiency, geometry and GPT-2

Written 6 Nov 2026, after the round-4 exploration under `AMENDMENTS.md` (Amendment 1; `NOTEBOOK.md`, exp22–exp26). This protocol, its scripts, the loader and the second new-text batch were digested in `FROZEN.md` before any script read the round-4 text for results. Later changes are dated entries in `AMENDMENTS.md`.

## Material and models
- **Text:** `data/newtext2/part-conf4-pythia410m.jsonl`: English Wikipedia articles created January 2025 to September 2026, a second batch that shares no article with the first (`data/newtext2/MANIFEST.md`). It holds 208 paragraphs from 108 articles (counted from the part file before the freeze) and has not been read by any script. The `reserve-pythia410m` part (200 paragraphs, 116 articles) stays unread.
- **Models:** Pythia-410M (final checkpoint), float32; GPT-2 Small, float32 (X only).
- **From WikiText-2 train, as in exploration:** the count-shift class means, the ablation means, the resample pool and the early-resample donors.
- **Samples:** as in the exploration scripts the confirmation scripts copy.
  - Width items: up to 100 at width 60 for D and X and up to 80 for S, built as in exploration (one per paragraph, as many as the part allows).
  - Natural text: 40 paragraphs at width 60 for S, and 40 at each of widths 40 and 60 for G.
  - X uses 40 paragraphs per width for behaviour and 30 for probes and attention.

## Predictions (each could fail)
Thresholds are about half the explored sizes, or a sign or rank test, as in rounds 2 and 3. Explored values (WikiText-2 train) are given in brackets.

**D, the decision step** (`code/p4_01_decision.py`). Two sources are introduced after layer 5:
- COUNT: +6 characters at the decision token;
- WIDTH: the line-start newline's state taken from a prompt with a previous line at least 10 characters shorter.

Components are frozen at the decision token to their unedited values. The share lost is 1 − mean(effect with the components frozen) / mean(effect).
- **D1, the mid-layer writers:** freezing MLPs 14, 16 and 18 removes ≥ 0.11 of the COUNT effect [0.22], with the item-bootstrap lower bound above 0. Its share minus the mean share of 10 random MLP triples [0.04] has a paired lower bound above 0.
- **D2, the late dampers:** freezing MLPs 21 and 23 makes both effects larger: the share is ≤ −0.10 for COUNT [−0.36] and for WIDTH [−0.37], with upper bounds below 0.
- **D3, the width is written onto the count's direction:** take the cosine between the mean residual changes from COUNT and from WIDTH at the decision token.
  - D3a: the cosine after layer 12 minus the cosine after layer 6 is ≥ 0.20 [0.48], with lower bound above 0.
  - D3b: the cosine after layer 12 is ≥ 0.65 [0.79].
- **D4, the main direct writer:** among all 288 heads of layers 6–23, L17H1 has the largest positive linearised direct effect on log P(newline), for COUNT [0.97, next 0.20] and for WIDTH [0.43, next 0.14].

**S, sufficiency** (`code/p4_02_sufficiency.py`; mean ablation is the tested condition, resample ablation is reported). The circuit and the scopes are those of exp25. Five random circuits of the same size are drawn with seed 0.
- **S1, the late circuit at the decision token** (DEC scope):
  - S1a: AUROC is at least the intact AUROC minus 0.02 [0.980 against 0.979].
  - S1b: the width effect kept is ≥ 0.33 [0.66]. Its paired item-bootstrap lower bound exceeds the largest share kept by any random circuit [0.26].
- **S2, the whole circuit** (BOTH scope): the separation kept is ≥ 0.15 [0.29]. Its paragraph-bootstrap lower bound exceeds the largest share kept by any random circuit [0.13].

**G, the shape of the count** (`code/p4_03_geometry.py`; split-half analysis as in exp23).
- **G1, one dominant linear axis:** at widths 40 and 60, after layer 5, the top direction carries ≥ 0.50 of half B's between-class variance [0.71–0.77]. A straight line in characters used explains ≥ 0.85 of it [0.95–0.99]. With the labels shuffled, the overall linear R² is < 0.20 [≤ 0.03 in exp23].
- **G2, a curved second direction:** at widths 40 and 60, after layer 6, the second direction's spline R² minus its linear R² is ≥ 0.30 [0.65–0.79], and its A–B reliability is ≥ 0.70 [0.93–0.97].
- **G3, readers ramp to the edge, without head-specific offsets:** at widths 40 and 60, the attention of L6H2, L8H15 and L17H1 to the line-start newline is fitted by a monotone function with R² ≥ 0.90 [0.99–1.00] and peaks at ≤ 5 characters remaining [0–2].

**X, what GPT-2 Small lacks** (`code/p4_04_roles.py`, run for both models).
- **X1, the count is present in GPT-2:** its best-layer ridge R² for characters used is ≥ 0.70 [0.82].
- **X2, but barely used:** GPT-2's largest count-shift effect over layers is ≤ 0.5 × Pythia-410M's [0.59 against 2.51].
- **X3, a weaker width effect:** GPT-2's behavioural width effect is ≤ 0.5 × Pythia-410M's [0.34 against 1.72].
- **X4, no strong counter head:** among heads with ≥ 0.05 of their attention on the line start and r ≤ −0.5 with characters used (the eight with the most negative r), GPT-2's largest share is ≤ 0.5 × Pythia-410M's [0.27 against 0.80], and Pythia-410M's is ≥ 0.60 [0.80].
- **X5, a short-lived stored width:**
  - GPT-2: patching the line-start newline from the short-line prompt after any of layers 8–11 moves log P(newline) by less than 0.10 in absolute value [≤ 0.04].
  - Pythia-410M: the same patch after layer 8 moves it by ≥ 0.25 [0.41].

**What counts against.** Any named part failing. Each lettered group (D, S, G, X) is reported as passing only if all its parts pass.

**Not tested prospectively.** The training dynamics (item 5; `exp26`), which stay exploratory and descriptive. The sub-additivity of COUNT and WIDTH and the resample-ablation numbers are reported without verdicts.

## Analysis
- Bootstrap: 2,000 resamples with seed 0, over items (D, S1b, X3) or paragraphs (S2). Paired where both quantities come from the same items or paragraphs. Ratios in X2 and X3 use point estimates.
- Thresholds are fixed in advance, so there is no multiple-test correction.
- Verdicts and tables come from `code/report4_tables.py`, which reads only the saved outputs and checks its tables verbatim against `REPORT-4.md`.
- Pipeline checks ran on the WikiText-2 train split before the freeze; the train runs of p4_01 and p4_03 are the exploratory reference values above. They are not results.

## Stopping rules
- One run per script on the round-4 part.
- A crash is fixed with a dated note in `AMENDMENTS.md`.
- No re-runs to improve a result.
- The results are written up whichever way they go.

## Compute
Local machine, in the memory cage, under the shared GPU lock, one job at a time; about 30 minutes in total. No downloads beyond the fetched text.

## Order of runs
1. Freeze: digests in `FROZEN.md`, files made read-only.
2. `run.sh p4_01_decision.py conf4-pythia410m`.
3. `run.sh p4_02_sufficiency.py conf4-pythia410m`.
4. `run.sh p4_03_geometry.py conf4-pythia410m`.
5. `run.sh p4_04_roles.py gpt2 conf4-pythia410m`, then `run.sh p4_04_roles.py pythia-410m conf4-pythia410m`.
