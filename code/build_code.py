"""Python-code text for D1 round 5 (AMENDMENTS.md, Amendment 2, item 4), built 6 Nov 2026 from the Python
3.14 standard library on disk (/usr/lib/python3.14, top-level *.py files and packages, excluding test/,
idlelib/, site-packages/ and __pycache__). Rule:
 1. Each module is parsed; every top-level function or class (and every method of a top-level class)
    gives its source segment, with its decorators.
 2. All runs of whitespace (newlines, indentation) are collapsed to a single space. The result is a
    'paragraph' of code words, later wrapped greedily like prose.
 3. Kept if 600-1800 characters, ASCII only, no tab, and no word longer than 18 characters
    (so that every width from 20 up can wrap it). At most 3 per file, in source order.
 4. Two parts by file: part = int(sha256("D1-code-" + relative path), 16) % 2, named explore-code
    (exploration) and conf5-code (round 5's confirmation).
The code is in Pythia's training distribution in kind (the Pile has GitHub code), and older versions of
these modules may have been seen; the test is of the mechanism, not of unseen text. Licence: PSF.
"""
import ast, json, re, hashlib, sys
from pathlib import Path
SRC = Path("/usr/lib/python3.14"); OUT = Path(__file__).resolve().parent.parent / "data" / "code"; OUT.mkdir(exist_ok=True)
SKIPDIR = {"test", "tests", "idlelib", "site-packages", "__pycache__", "lib2to3", "turtledemo"}
PARTS = ["explore-code", "conf5-code"]
parts = {n: [] for n in PARTS}; nfiles = 0
for f in sorted(SRC.rglob("*.py")):
    rel = f.relative_to(SRC).as_posix()
    if set(rel.split("/")[:-1]) & SKIPDIR or rel.startswith("test"): continue
    try: src = f.read_text(encoding="utf-8"); tree = ast.parse(src)
    except Exception: continue
    nfiles += 1
    nodes = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes.append(n)
            if isinstance(n, ast.ClassDef):
                nodes += [m for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    lines = src.splitlines(); out = []
    for n in nodes:
        start = min([d.lineno for d in n.decorator_list] + [n.lineno]) - 1
        seg = "\n".join(lines[start:n.end_lineno])
        s = re.sub(r"\s+", " ", seg).strip()
        if len(s) > 1800: s = s[:1800].rsplit(" ", 1)[0]
        if not (600 <= len(s) <= 1800) or not s.isascii() or max(len(w) for w in s.split(" ")) > 38: continue
        out.append(s)
        if len(out) == 3: break
    k = int(hashlib.sha256(f"D1-code-{rel}".encode()).hexdigest(), 16) % 2
    for q, s in enumerate(out): parts[PARTS[k]].append(dict(file=rel, idx=q, text=s))
for n, ps in parts.items():
    with open(OUT / f"part-{n}.jsonl", "w") as fh:
        for p in ps: fh.write(json.dumps(p) + "\n")
summary = dict(source=str(SRC), python=sys.version.split()[0], n_files_parsed=nfiles,
               parts={n: dict(paragraphs=len(ps), files=len({p["file"] for p in ps})) for n, ps in parts.items()})
(OUT / "build_summary.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))
