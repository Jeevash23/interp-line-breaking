# Second new-text batch (D1 round 4): manifest

- **Source:** English Wikipedia through the public MediaWiki API, as for the first batch (`../newtext/MANIFEST.md`).
- **Fetched:** 6 Nov 2026. The fetch finished at 2026-11-06T14:23:07 UTC (`fetch_summary.json`), sequentially, at about one request per second, with a generic research User-Agent.
- **Licence:** CC BY-SA 4.0. Titles, page ids and revision ids are in `articles.jsonl` for attribution. The text is used for research measurements only and is not redistributed.
- **Script:** `../../code/fetch_newtext2.py` (digest in `../../FROZEN.md`, frozen before the fetch's results were read). The selection rule is the first batch's, with these changes:
  - starting points on the 8th and 22nd of each month, January 2025 to September 2026 (42 starts);
  - up to 15 pages per start;
  - every article of the first batch excluded (checked: no page id in common);
  - two parts by article hash: `int(sha256("D1-newtext2-" + pageid), 16) % 2`.
- **Caveat:** as for the first batch, a page created in 2025–26 can contain older text moved without a summary saying so.
- `extracts.jsonl` is the on-disk cache the script fills one article at a time. It is identical to `articles.jsonl` (same digest). `pages.json` is the selected page list.

| Part | Use | Articles | Paragraphs |
|---|---|---|---|
| conf4-pythia410m | D1 round 4 confirmation (`../../PROTOCOL-4.md`) | 108 | 208 |
| reserve-pythia410m | unread, kept for a later round | 116 | 200 |

Articles fetched: 630; articles contributing at least one paragraph: 224; paragraphs: 408.
