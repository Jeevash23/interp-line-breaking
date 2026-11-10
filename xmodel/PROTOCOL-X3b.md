# D1-X confirmation protocol X3b (CONFIRMATORY): the line-breaking roles in Gemma-4-E2B

Written 7 Nov 2026, early afternoon, under `CHARTER.md` and Amendment 3 (`AMENDMENTS.md`), after exploration on WikiText-2 train and on Gemma's exploration part of the third English batch (`NOTEBOOK.md`). The author's next steps (06:05 UTC), logged in `../../DECISIONS.md`, asked for Gemma-4-E2B and SmolLM2-1.7B to be taken through this protocol and its single held-out run.

Frozen, with its spec file, before any script reads `part-conf-gemma4-e2b`. Digests are in `FROZEN.md`. Later changes are dated entries in `AMENDMENTS.md`.

**SmolLM2-1.7B is not covered here.** Its head sets had not been chosen when this file was frozen. The loader in `code/xm3_common.py` (frozen with `PROTOCOL-X3.md`) maps SmolLM2's confirmation part to this file as well. So once this file is read-only, that part is protected by procedure, not by code:
- no confirmation spec for SmolLM2 exists, and `code/xm3_conf.py` cannot run without one;
- SmolLM2's sets, thresholds and predictions will be frozen in a dated addendum, `PROTOCOL-X3b-smollm2.md`, before any script reads `part-conf-smollm2-17b`.

## Question
Pythia-410M breaks lines with four roles: counters, a stored length at the newline, token correctors, and readers. Five more models have the same roles, confirmed on held-out text (`PROTOCOL-X1.md`, `PROTOCOL-X3.md`, `REPORT-*.md`). Gemma-4-E2B is the first model in this series with any of these features:
- interleaved sliding-window and global attention;
- keys and values shared across its last 20 layers;
- a per-layer token embedding.

Exploration found the four roles concentrated in fewer heads than in any earlier model. This protocol tests, on text no script has read, whether the named heads do these jobs at stated sizes, beyond layer-matched random heads.

## Material and model
- **Text:** `data/newtext3/part-conf-gemma4-e2b.jsonl`.
  - 193 paragraphs from 110 English Wikipedia articles created January 2025 to September 2026, none of them in the first two batches.
  - The articles were split by article hash before any reading (`data/newtext3/MANIFEST.md`; digests in `FROZEN.md`). Paragraph order seed 100.
  - Gemma 4 was released in 2026, so some of these articles may be in its training data. Here "held out" means that no analysis has read the part.
- **Model:** `google/gemma-4-E2B` (base; snapshot 433813a6…).
  - Text decoder in bfloat16 on the local GPU, loaded as in `code/xm3_common.py`, with the beginning-of-sequence token prepended and excluded from every measure. The image and audio encoders are not loaded. The per-layer embedding table is read row by row from the weights file.
  - Set-up checks (`outputs/xm3_gemma_fit.json`): chunked outputs equal the model's own output exactly; peak GPU memory 4.96 GB.
  - **No float32 check** (Amendment 3: too much memory). This is a stated limit.
- **Script:** `code/xm3_conf.py specs/conf-gemma4-e2b.json conf-gemma4-e2b`, run once.
  - Measures, samples, ablation means (WikiText-2 train), decision points (letter-ending word ends), line-start marker (any token containing a newline), random sets (10 per test, layer-matched, never a head named in any role, numpy seed 0) and bootstrap (2,000 resamples, seed 0) are exactly as in `PROTOCOL-X1.md` and `PROTOCOL-X3.md`.
  - Lw = 7.
  - Gemma has 8 query heads per layer, so a random set draws from at most 7 heads of each layer, fewer where candidate heads sit.

## Tests
G (gate), C (counters), T (token corrector), S (stored length, newline-only ablation, with a lower bound on the decision kept) and R (readers) are as defined in `PROTOCOL-X1.md`. In exploration, Gemma's stored-length heads kept 68–69% of the natural separation, so the S form (a partial dissociation, as for OLMo's early writers) is used, not the X form used for Qwen3. Thresholds are about half the smaller of the two explored sizes, rounded down. For S, the decision-kept threshold is 15–20 points below the smaller explored value, as in `PROTOCOL-X1.md`.

| Test | Set | Explored (train / new text) | Threshold | Random limit |
|---|---|---|---|---|
| C | L6H3, L12H3 | 72% / 67% | share ≥ 0.30 | mean < 0.15 |
| T | L6H3 | ablated 0.35, 0.27, 0.23 / 0.38, 0.32, 0.26 at widths 40, 60, 80 | ≥ 0.11 at 40, 60 and 80 | mean < 0.10 |
| S | L6H1, L6H3, L7H6 | 36%, kept 69% / 46%, kept 68% | share ≥ 0.18; kept ≥ 0.50 | mean < 0.15 |
| R | L9H5, L12H3 | 47% / 47% | share ≥ 0.20 | mean < 0.15 |

## Predictions that would not be made from the earlier models alone
- **Two heads carry the count.** In every earlier model the counter set has four to six heads. Here two heads (L6H3, L12H3) should carry at least 30% of the count effect.
- **One head in three roles.** L6H3 is in C, T and S. It is also a dilution head: its attention to the line-start newline falls as the line fills (r = −0.77 in exploration). So one head counts, keeps the token count out of the decision, and writes the stored length at the newline.
- **The count carrier is also a reader.** L12H3 is in C and R. This is OLMo-2-1B's arrangement (L6H2), not that of Pythia-1B, Qwen2.5 or Qwen3.

## What counts
Each role that passes is confirmed for Gemma-4-E2B. If the gate fails, the role tests are still run and reported, but no role claim is made.

## What is reported
`REPORT-gemma4-e2b.md` gives every test's numbers and verdict as frozen, whichever way they go. Its tables are generated from the saved output by `code/xm3_report_tables.py` and checked with `--check`, and its prose numbers are checked by `code/xm3_report_prose_check.py`. Nulls and failures are stated plainly. No re-runs and no threshold changes after the confirmation part has been read.

## Compute
- One GPU job through `code/run.sh` (memory cage, shared GPU lock, at most one D1-X job waiting).
- About 30 minutes, which is above the charter's 20-minute guide; the script is not split, so that it matches the transfer check exactly.
- A crash before any output is written may be re-run once, unchanged, and logged.
