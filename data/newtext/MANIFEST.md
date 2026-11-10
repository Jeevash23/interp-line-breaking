# New text for D1 round 3 and D1-X: manifest

- **Source:** English Wikipedia, through the public MediaWiki API (`https://en.wikipedia.org/w/api.php`): the page-creation log (`list=logevents`, `letype=create`), page information, and plain-text extracts of the current revision (TextExtracts, `explaintext`, wiki-style headings).
- **Fetched:** 6 Nov 2026; the fetch finished at 2026-11-06T10:40:19 UTC (`fetch_summary.json`). One request at a time, about one per second, with a generic research User-Agent.
- **Licence:** Wikipedia text is CC BY-SA 4.0. Each article's title, page id and revision id (`lastrevid`) are kept in `articles.jsonl` for attribution. The text is used here for research measurements only and is not redistributed.
- **Script:** `../../code/fetch_newtext.py` (digest in `../../FROZEN.md`). Its docstring states the selection rule, which was fixed before fetching:
  1. Creation log read forward from 00:00 UTC on the 1st and the 15th of each month, January 2025 to September 2026 (42 starting points).
  2. Creations skipped when the edit summary matches `redirect|split|merg|cop(y|ied)|move|translat|from \[\[|draft` (case-insensitive), because such text may be older than its page.
  3. Pages kept if they still exist, are not redirects, are in namespace 0 and have at least 6,000 bytes of wikitext; up to 60 per starting point, in log order.
  4. Paragraphs: lines of the extract with whitespace collapsed; dropped if they start with "=", are under 600 or over 1,800 characters, or contain "@", "{\displaystyle" or any character above U+024F; at most the first 3 per article.
  5. Parts by article: `int(sha256("D1-newtext-" + pageid), 16) % 7`, in the order conf-pythia410m, explore-pythia1b, conf-pythia1b, explore-qwen05b, conf-qwen05b, explore-olmo1b, conf-olmo1b.
- **Caveat:** a page created in 2025–26 can still contain older text (for example, material moved from another article without a summary saying so). The summary filter removes the declared cases only.

## Counts

Articles fetched: 2,520 (60 per starting point). Articles contributing at least one paragraph: 946. Paragraphs: 1,740.

| Part | Use | Articles | Paragraphs |
|---|---|---|---|
| conf-pythia410m | Pythia-410M round 3 (`../../PROTOCOL-3.md`) | 146 | 264 |
| explore-pythia1b | D1-X exploration, Pythia-1B | 130 | 235 |
| conf-pythia1b | D1-X confirmation, Pythia-1B | 146 | 274 |
| explore-qwen05b | D1-X exploration, Qwen2.5-0.5B | 118 | 213 |
| conf-qwen05b | D1-X confirmation, Qwen2.5-0.5B | 143 | 269 |
| explore-olmo1b | D1-X exploration, OLMo-2-1B (approved 6 Nov 2026) | 131 | 247 |
| conf-olmo1b | D1-X confirmation, OLMo-2-1B | 132 | 238 |

The split was made by the fetch script and recorded in `../../FROZEN.md` before any script read the text for results. Confirmation parts are read only under a protocol frozen beforehand for that model. The OLMo parts belong to D1-X.

`SHA256SUMS` in this folder lists every file here.
