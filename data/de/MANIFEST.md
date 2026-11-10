# German Wikipedia text for D1 round 5

Fetched on 6 Nov 2026 (finished 2026-11-06T17:03:14+00:00 UTC) by `code/fetch_de.py` (rule in its docstring) from de.wikipedia.org: main-namespace articles created from January 2025 to September 2026 (the 8th of each month as starting points, up to 12 pages each, at least 6,000 bytes), plain-text extracts, up to 3 paragraphs per article of 600–1,800 characters with no '@', TeX or non-Latin characters. 252 articles, 1.6 MB on disk.

| Part | Paragraphs | Articles | Use |
|---|---|---|---|
| explore-de | 108 | 64 | round-5 exploration |
| conf5-de | 114 | 60 | round-5 confirmation; frozen and unread until `PROTOCOL-5.md` is frozen |

Split by article: `int(sha256("D1-de-" + pageid), 16) % 2`. Licence: CC BY-SA 4.0 (Wikipedia contributors). The articles were created after Pythia's training data was collected, and German is a small share of that data.
