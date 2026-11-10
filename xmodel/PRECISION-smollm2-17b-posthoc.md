# SmolLM2-1.7B: float32 precision check (POST HOC)

Written 7 Nov 2026, 12:10 UTC. This is the float32 check that `PROTOCOL-X3b-smollm2.md` lists as a stated limit ("may still be run and reported as a post-hoc check; it does not change any verdict here"). It was run after the held-out confirmation (09:59–10:07 UTC), once enough memory was free. No frozen file, spec or confirmed output was changed. `REPORT-smollm2-17b.md` and `REPORT-CROSSMODEL.md` are unchanged; this note is the addendum to both.

## What was run
- **Script:** `code/xm3_e00_precision.py smollm2-17b fp32cpu`, the same precision spot check used for Pythia-1B, OLMo-2-1B and both Qwen3 models. It computes the newline log-odds at every letter-ending decision point of the first 12 WikiText-2 train paragraphs (seed 0) wrapped at width 60.
- **Run:** float32 on the CPU, 17:34–12:05 UTC, in a memory cage (9 GB, 2 GB swap) while holding the shared GPU lock. Available memory just before the start: 10 GB.
- **Comparison:** `code/xm3_e00_compare.py smollm2-17b fp32cpu bf16gpu`, against the bfloat16 GPU half saved at 05:10 UTC (`outputs/xm_e00_smollm2-17b_bf16gpu.json`).
- **Outputs (new files):** `outputs/xm_e00_smollm2-17b_fp32cpu.json`, `outputs/xm3_e00_compare_smollm2-17b.json`, and the log `outputs/xm3_e00_smollm2-17b_fp32cpu.log`.

## Result
| | float32 (CPU) | bfloat16 (GPU) |
|---|---|---|
| Decision points | 1264 | 1264 |
| AUROC at width 60 | 0.9874 | 0.9871 |
| Separation | 3.737 | 3.737 |

Point by point, the two precisions' newline log-odds correlate at r = 0.9996. The mean absolute difference is 0.075, the 95th percentile 0.19 and the largest 0.63. This matches the two Qwen3 models (r = 0.9997 for both).

**bfloat16 agrees with float32 for SmolLM2-1.7B on the decision measure.**

## What this does and does not cover
- The check compares the model's decision at the end of a line, the measure behind the gate and every role test. It does not re-run the role ablations in float32. The same holds for the float32 checks of the other models.
- Gemma-4-E2B still has no float32 check, because it does not fit in the local machine's memory. That limit stands.

The numbers in this note are checked against the saved outputs by `code/xm3_precision_note_check.py`.
