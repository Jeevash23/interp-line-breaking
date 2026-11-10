# D1-X charter amendments

Dated changes to `CHARTER.md` (frozen 6 Nov 2026 06:33 UTC). The charter itself is not edited.

## Amendment 1, 6 Nov 2026 about 07:15 UTC: OLMo-2-1B added
- **Reason.** The author approved the OLMo-2-1B download (the author's approval is quoted in the original record). The download is about 5.95 GB.
- **Model.** `allenai/OLMo-2-0425-1B` (base, final checkpoint): 16 layers × 16 heads, separate key and value per head, 1.48B parameters.
- **Precision.** float32 is about 5.9 GB and does not fit the 6 GB GPU, so OLMo runs in **bfloat16** on the GPU. Before any role search, a float32 CPU spot check on a small WikiText-2 train sample compares the gate measure (AUROC and separation at width 60) and the newline log-odds at decision points. If they disagree materially, the bfloat16 numbers are flagged and the choice is logged in `NOTEBOOK.md`. The same rule is applied to Pythia-1B, which the charter already runs in bfloat16.
- **Material.** As for the other models: WikiText-2 train for exploration, adding `part-explore-olmo1b.jsonl` once the split exists; `part-conf-olmo1b.jsonl` only under a frozen protocol.
- **Order and budget.** OLMo-2-1B comes third, after Pythia-1B and Qwen2.5-0.5B, with the same gate, candidate rule, one-week limit per model and stop rules. The two-week total in the charter is unchanged; if three models need more, that is parked for the author.
- **Unchanged.** Everything else in the charter, including letter-ending decision points, the line-start marker rule (any token containing a newline) and the GPU lock.

## Amendment 2, 6 Nov 2026 about 13:30 UTC: follow-up on OLMo-2-1B's reader overshoot
- **Reason.** The author approved one more D1-X item: explain why OLMo-2-1B's reader share exceeds 100%. The instruction was to explore first on train or the exploration parts, then, if a clear explanation emerges, add a short frozen follow-up and test it once on unused held-out material.
- **Exploration** (`code/xm_e06_overshoot.py`; WikiText-2 train and `part-explore-olmo1b`) gave a consistent account. It is written up with its predictions in `PROTOCOL-X2.md`.
- **Held-out material.** OLMo's new-text confirmation part has already been used by `PROTOCOL-X1.md`. The follow-up therefore uses the WikiText-2 **validation** split, which no OLMo script and no D1-X script has read; D1 read it for Pythia-410M only.
- **Unchanged.** `PROTOCOL-X1.md` and its verdicts.

## Amendment 3, 6 Nov 2026 about 17:20 UTC: newer open models
- **Reason.** The author asked for the line-breaking roles to be tested in newer open models overnight, downloads of several GB included. The request is quoted in the original record and logged in `../../DECISIONS.md`.
- **Models.** Base (pretrained) checkpoints, so that they compare with the base models already studied. All ungated on Hugging Face (`gated: false`, checked 17:10 UTC).
  - **Qwen3-0.6B-Base** (`Qwen/Qwen3-0.6B-Base`): 28 layers × 16 query heads, 8 key/value heads; 1.2 GB. The post-trained `Qwen3-0.6B` already on disk is not used, because the other models are base models.
  - **Qwen3-1.7B-Base** (`Qwen/Qwen3-1.7B-Base`): 28 layers × 16 query heads, 8 key/value heads; 3.4 GB.
  - **Gemma-4-E2B** (`google/gemma-4-E2B`, base): 35 layers × 8 query heads, 1 key/value head, sliding-window and global layers, and the last 20 layers reuse earlier layers' keys and values. The checkpoint is 10.2 GB in bfloat16 because it holds image and audio encoders and a per-layer token-embedding table. Only the text decoder runs on the GPU. The per-layer embedding table is a lookup, so it stays on the CPU in bfloat16 and the image and audio encoders are not used. Nothing is quantised. If the text decoder does not fit in about 5.5 GB of GPU memory for 700-token inputs, Gemma is skipped and the reason logged.
  - **SmolLM2-1.7B** (`HuggingFaceTB/SmolLM2-1.7B`): 24 layers × 32 heads, separate key and value per head; 3.4 GB. Optional: run fourth, if time allows.
  - **Skipped as gated:** `google/gemma-3-1b-pt` and `meta-llama/Llama-3.2-1B` (manual approval required). There is no Hugging Face token on the local machine, and no licence is accepted for the author. Logged as an owner decision.
- **Precision.** All four run in bfloat16 on the GPU. Before any role search, a float32 CPU spot check on a small WikiText-2 train sample compares the gate measure and the newline log-odds at decision points (as in Amendment 1) for Qwen3-0.6B-Base, Qwen3-1.7B-Base and SmolLM2-1.7B. Gemma-4-E2B in float32 needs more than the local machine's free memory, so its check is skipped and recorded as a limit.
- **Beginning-of-text token.** Gemma models expect a beginning-of-sequence token. For Gemma-4-E2B it is prepended to every input. All positions, patches and measures refer to the text tokens, and the extra position is never ablated or patched. The other models get no such token, as before.
- **Material.**
  - **Exploration:** WikiText-2 train, then `part-explore-<model>` of a third English Wikipedia batch, `data/newtext3/` in this folder.
  - **The third batch** (`code/fetch_newtext3.py`, frozen before it ran) takes articles created January 2025 to September 2026 from new starting days, the 4th and 18th of each month. It excludes every article of the first two batches, including the main study's unread `reserve-pythia410m`. It is split by article hash into an exploration and a confirmation part for each model.
  - **Confirmation:** `part-conf-<model>` only, under a frozen `PROTOCOL-X3.md`.
  - **Limit stated in advance:** Gemma 4 and SmolLM2 were trained later than the earlier models, and some 2025 articles may predate a model's training data. "Held out" here means that no analysis has read the part.
- **Order.** Qwen3-0.6B-Base, Qwen3-1.7B-Base, Gemma-4-E2B, then SmolLM2-1.7B if time allows. The gate, candidate rule and stop rules are the charter's. Key-only or value-only tests are at the level of a key/value group where heads share one.
- **GPU.** D1 round 5 (main study) and A3 (area 10) have priority. At most one D1-X job waits on `~/.cache/d1-gpu.lock`, every job runs in the memory cage, and none starts while `~/.cache/d1-gpu.round3` exists.
- **Unchanged.** `PROTOCOL-X1.md`, `PROTOCOL-X2.md` and their verdicts. Code for the new models is in new files (`code/xm3_*.py`); the frozen `xm_common.py` is not edited.
