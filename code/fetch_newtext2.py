"""Second new-text batch for D1 round 4 (Pythia-410M only), fetched 6 Nov 2026 because every part of
the first batch (data/newtext/) is already assigned. Same selection rule as fetch_newtext.py, except:
 1. Starting points are the 8th and the 22nd of each month, Jan 2025 to Sep 2026 (42 starts), and
    up to 15 pages are kept per start (not 60).
 2. Pages already in data/newtext/articles.jsonl are excluded, so the two batches share no article.
 3. Two parts by article: part = int(sha256("D1-newtext2-" + pageid), 16) % 2, named
    conf4-pythia410m (round 4's confirmation) and reserve-pythia410m (kept unread for later).
 4. Extracts are cached on disk one article at a time (extracts.jsonl), so a crash loses nothing.
Steps 2-5 of fetch_newtext.py's rule (summary filter, page filter, extracts, paragraph filter,
at most 3 paragraphs per article) are unchanged. Licence: CC BY-SA 4.0.
"""
import json, re, time, hashlib, urllib.request, urllib.error, urllib.parse, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

OUT = Path(__file__).resolve().parent.parent / "data" / "newtext2"
OUT.mkdir(exist_ok=True)
PREV = {json.loads(l)["pageid"] for l in open(OUT.parent / "newtext" / "articles.jsonl")}
API = "https://en.wikipedia.org/w/api.php"
UA = "D1-linebreak-research/1.0 (non-commercial interpretability research script)"
PARTS = ["conf4-pythia410m", "reserve-pythia410m"]
NPER = 15
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
    for d in (8, 22): starts.append(f"{y}-{m:02d}-{d:02d}T00:00:00Z")

pages = []; seen = set()
for st in starts:
    kept = 0; cont = {}
    while kept < NPER:
        r = get(dict(action="query", list="logevents", letype="create", lenamespace=0, lestart=st, ledir="newer", lelimit=500, **cont))
        evs = [e for e in r["query"]["logevents"] if not SKIP.search(e.get("comment", "")) and e.get("pageid")]
        for i in range(0, len(evs), 50):
            chunk = evs[i:i + 50]
            info = get(dict(action="query", prop="info", pageids="|".join(str(e["pageid"]) for e in chunk)))
            ok = {p["pageid"]: p for p in info["query"]["pages"] if not p.get("missing") and not p.get("redirect") and p.get("ns") == 0 and p.get("length", 0) >= 6000}
            for e in chunk:
                p = ok.get(e["pageid"])
                if p and p["pageid"] not in seen and p["pageid"] not in PREV and kept < NPER:
                    seen.add(p["pageid"]); kept += 1
                    pages.append(dict(pageid=p["pageid"], title=p["title"], created=e["timestamp"], lastrevid=p["lastrevid"], length=p["length"], start=st))
        if "continue" not in r: break
        cont = {"lecontinue": r["continue"]["lecontinue"]}
    print(st, kept, len(pages), flush=True)

def extract(p):
    r = get(dict(action="query", prop="extracts", explaintext=1, exsectionformat="wiki", pageids=p["pageid"]))
    return dict(p, extract=r["query"]["pages"][0].get("extract", ""))
(OUT / "pages.json").write_text(json.dumps(pages))
CACHE = OUT / "extracts.jsonl"
done = {json.loads(l)["pageid"]: json.loads(l) for l in open(CACHE)} if CACHE.exists() else {}
arts = []
with open(CACHE, "a") as cf:
    for q, p in enumerate(pages):
        if p["pageid"] not in done:
            a = extract(p); cf.write(json.dumps(a) + "\n"); cf.flush(); done[p["pageid"]] = a
        arts.append(done[p["pageid"]])
        if q % 100 == 0: print('extracts', q, flush=True)

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
    k = int(hashlib.sha256(f"D1-newtext2-{a['pageid']}".encode()).hexdigest(), 16) % 2
    for q, s in enumerate(paragraphs(a["extract"])):
        parts[PARTS[k]].append(dict(pageid=a["pageid"], title=a["title"], lastrevid=a["lastrevid"], idx=q, text=s))
for n, ps in parts.items():
    with open(OUT / f"part-{n}.jsonl", "w") as f:
        for p in ps: f.write(json.dumps(p) + "\n")
summary = dict(fetched_utc=fetched, n_articles=len(arts), parts={n: dict(paragraphs=len(ps), articles=len({p["pageid"] for p in ps})) for n, ps in parts.items()})
(OUT / "fetch_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
