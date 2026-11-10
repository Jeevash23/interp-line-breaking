"""Fetch fresh English Wikipedia prose (articles created 2025-01-01 .. 2026-09-30) for D1 round 3 and
the cross-model study, and split it into parts BEFORE any of it is read for results.

Selection rule (fixed before fetching):
 1. Page-creation log (list=logevents, letype=create, namespace 0), read forward from 00:00 UTC on
    the 1st and the 15th of each month, Jan 2025 to Sep 2026 (42 starting points).
 2. Skip creations whose edit summary mentions a redirect, split, merge, copy, move or translation
    (the text may be older than its page).
 3. Keep pages that still exist, are not redirects, are in namespace 0, and whose current wikitext
    is >= 6,000 bytes; up to 60 such pages per starting point, in log order.
 4. Plain-text extract of the current revision (TextExtracts, explaintext, wiki-style headings).
 5. Paragraphs: lines of the extract, whitespace collapsed; dropped if they start with "=", are
    shorter than 600 or longer than 1,800 characters, contain "@", "{\\displaystyle", or any
    character above U+024F; at most the first 3 kept per article.
 6. Parts, by article (all of an article's paragraphs go to one part):
    part = int(sha256("D1-newtext-" + pageid), 16) % 7, named in PARTS below.
Licence: Wikipedia text is CC BY-SA 4.0; titles and revision ids are recorded for attribution.
"""
import json, re, time, hashlib, urllib.request, urllib.error, urllib.parse, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

OUT = Path(__file__).resolve().parent.parent / "data" / "newtext"
API = "https://en.wikipedia.org/w/api.php"
UA = "D1-linebreak-research/1.0 (non-commercial interpretability research script)"
PARTS = ["conf-pythia410m", "explore-pythia1b", "conf-pythia1b", "explore-qwen05b", "conf-qwen05b", "explore-olmo1b", "conf-olmo1b"]
SKIP = re.compile(r"redirect|split|merg|cop(y|ied)|move|translat|from \[\[|draft", re.I)

def get(params, tries=8):
    """Polite client: one request at a time, about 1 per second, backing off on HTTP 429 / errors
    (Retry-After honoured). Changed 6 Nov 2026 after the first attempt was rate-limited at its
    first starting point; the selection rule is unchanged."""
    params = dict(params, format="json", formatversion="2", maxlag="5")
    url = API + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        time.sleep(1.0)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                d = json.loads(r.read().decode())
            if d.get("error", {}).get("code") == "maxlag": time.sleep(10); continue
            return d
        except urllib.error.HTTPError as e:
            err = e; time.sleep(float(e.headers.get("Retry-After") or 30 * (k + 1)))
        except Exception as e:
            err = e; time.sleep(5 * (k + 1))
    raise err

starts = []
for y, m in [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 10)]:
    for d in (1, 15): starts.append(f"{y}-{m:02d}-{d:02d}T00:00:00Z")

pages = []; seen = set()
for st in starts:
    kept = 0; cont = {}
    while kept < 60:
        r = get(dict(action="query", list="logevents", letype="create", lenamespace=0, lestart=st, ledir="newer", lelimit=500, **cont))
        evs = [e for e in r["query"]["logevents"] if not SKIP.search(e.get("comment", "")) and e.get("pageid")]
        for i in range(0, len(evs), 50):
            chunk = evs[i:i + 50]
            info = get(dict(action="query", prop="info", pageids="|".join(str(e["pageid"]) for e in chunk)))
            ok = {p["pageid"]: p for p in info["query"]["pages"] if not p.get("missing") and not p.get("redirect") and p.get("ns") == 0 and p.get("length", 0) >= 6000}
            for e in chunk:
                p = ok.get(e["pageid"])
                if p and p["pageid"] not in seen and kept < 60:
                    seen.add(p["pageid"]); kept += 1
                    pages.append(dict(pageid=p["pageid"], title=p["title"], created=e["timestamp"], lastrevid=p["lastrevid"], length=p["length"], start=st))
        if "continue" not in r: break
        cont = {"lecontinue": r["continue"]["lecontinue"]}
    print(st, kept, len(pages), flush=True)

def extract(p):
    r = get(dict(action="query", prop="extracts", explaintext=1, exsectionformat="wiki", pageids=p["pageid"]))
    return dict(p, extract=r["query"]["pages"][0].get("extract", ""))
arts = []
for q, p in enumerate(pages):
    arts.append(extract(p))
    if q % 200 == 0: print('extracts', q, flush=True)

def paragraphs(text):
    out = []
    for line in text.split("\n"):
        s = re.sub(r"\s+", " ", line.strip())
        if not s or s.startswith("=") or not (600 <= len(s) <= 1800): continue
        if "@" in s or "{\\displaystyle" in s or any(ord(c) > 0x24F for c in s): continue
        out.append(s)
        if len(out) == 3: break
    return out

fetched = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
with open(OUT / "articles.jsonl", "w") as f:
    for a in arts: f.write(json.dumps(a) + "\n")
parts = {n: [] for n in PARTS}
for a in arts:
    k = int(hashlib.sha256(f"D1-newtext-{a['pageid']}".encode()).hexdigest(), 16) % 7
    for q, s in enumerate(paragraphs(a["extract"])):
        parts[PARTS[k]].append(dict(pageid=a["pageid"], title=a["title"], lastrevid=a["lastrevid"], idx=q, text=s))
for n, ps in parts.items():
    with open(OUT / f"part-{n}.jsonl", "w") as f:
        for p in ps: f.write(json.dumps(p) + "\n")
summary = dict(fetched_utc=fetched, n_articles=len(arts), parts={n: dict(paragraphs=len(ps), articles=len({p["pageid"] for p in ps})) for n, ps in parts.items()})
(OUT / "fetch_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
