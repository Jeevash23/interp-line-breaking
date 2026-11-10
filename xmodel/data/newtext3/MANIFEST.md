# Third English Wikipedia batch (D1-X Amendment 3)

Fetched 6–7 Nov 2026 (finished 22:02 UTC the previous day) by `code/fetch_newtext3.py`, frozen before it ran (digest in `../../FROZEN.md`).

- **Articles:** 2,520 English Wikipedia articles created January 2025 to September 2026, up to 60 from each of 42 starting days (the 4th and 18th of each month). Plain-text extracts, split into paragraphs as for the first two batches.
- **Excluded:** every page id of the first two batches (`../../../data/newtext/` and `../../../data/newtext2/articles.jsonl`, 3,150 ids, which include the hub's unread round-5 reserve). Checked after the fetch: 0 of the 2,520 ids overlap. The reserve's text was not read.
- **Split:** article k goes to part `int(sha256("D1X-newtext3-" + pageid), 16) % 8`, in the order explore-qwen3-06b, conf-qwen3-06b, explore-qwen3-17b, conf-qwen3-17b, explore-gemma4-e2b, conf-gemma4-e2b, explore-smollm2-17b, conf-smollm2-17b. Paragraph and article counts per part are in `fetch_summary.json`.
- **Use:** `part-explore-<model>` for exploration; `part-conf-<model>` only under a frozen `PROTOCOL-X3.md` (the loader in `code/xm3_common.py` refuses it otherwise). `extracts.jsonl` and `articles.jsonl` are the same file.
- **Digests:** `SHA256SUMS`.
