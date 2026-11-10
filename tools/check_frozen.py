"""Check the study's freeze records against the files they name.

    python tools/check_frozen.py          # after tools/unpack.py; exit status 1 if any file fails

The freeze records are the Markdown tables in FROZEN.md (the Pythia-410M rounds) and xmodel/FROZEN.md (the
cross-model part): each row gives a file, relative to the record's folder, and the SHA-256 it had when it was
frozen. xmodel/FROZEN.md also copies digests from the data folders' SHA256SUMS listings (indented lines). Each
digest passes in one of four ways:

- unchanged: the file in this release has exactly that digest;
- changed for the release (documented): release-changes.json lists the file with that digest as the original and
  the file's current digest as the release copy (personal paths, local clock times and similar wording replaced;
  RELEASE-CHANGES.md lists every such file);
- superseded by a later dated entry: the same record digests the same file again further down (an amendment was
  appended and frozen again), and that later digest passes;
- not shipped (documented): release-changes.json lists the file under frozen_files_not_shipped.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
RECORDS = [("FROZEN.md", ""), ("xmodel/FROZEN.md", "xmodel")]
ROW = re.compile(r"\|\s*([^|]+?)\s*\|\s*([0-9a-f]{64})\s*\|")
LISTED = re.compile(r"^\s{4}([0-9a-f]{64})\s+(\S+)\s*$")
LISTING = re.compile(r"`([^`]*SHA256SUMS)`")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def entries(record: str, base: str):
    """[(release path, digest)] in the order the record gives them."""
    out, listing = [], None
    for line in (RELEASE / record).read_text().splitlines():
        found = LISTING.findall(line)
        if found:
            listing = os.path.dirname(found[-1])
        m, l2 = ROW.match(line), LISTED.match(line)
        if m:
            name = re.sub(r"\s*\(.*\)$", "", m.group(1))
            out.append((os.path.normpath(os.path.join(base, name)), m.group(2)))
        elif l2 and listing is not None:
            out.append((os.path.normpath(os.path.join(base, listing, l2.group(2))), l2.group(1)))
    return out


def main():
    changes = json.loads((RELEASE / "release-changes.json").read_text())
    changed, absent = changes["changed_files"], changes["frozen_files_not_shipped"]
    digest_cache = {}

    def current(p):
        if p not in digest_cache:
            f = RELEASE / p
            digest_cache[p] = sha(f) if f.is_file() else None
        return digest_cache[p]

    def passes(p, d):
        cur = current(p)
        if cur == d:
            return "unchanged"
        ch = changed.get(p)
        if ch and ch.get("original_sha256") == d and cur == ch.get("release_sha256"):
            return "changed for the release (documented)"
        if p in absent and absent[p]["original_sha256"] == d:
            return "not shipped (documented)"
        return None

    totals, failures = {}, []
    for record, base in RECORDS:
        rows = entries(record, base)
        counts = {}
        for i, (p, d) in enumerate(rows):
            status = passes(p, d)
            if status is None:
                later = [d2 for p2, d2 in rows[i + 1:] if p2 == p and d2 != d]
                if later and passes(p, later[-1]):
                    status = "superseded by a later dated entry"
            if status is None:
                failures.append(f"{record}: {p} ({'missing' if current(p) is None else 'digest differs'})")
                status = "FAILED"
            counts[status] = counts.get(status, 0) + 1
            totals[status] = totals.get(status, 0) + 1
        print(f"{record}: {len(rows)} digests: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    for f in failures:
        print("FAIL", f)
    print("total: " + ", ".join(f"{v} {k}" for k, v in sorted(totals.items())))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
