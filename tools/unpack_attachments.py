"""Unpack the release attachments into the folders the code reads, checking every SHA-256.

    python tools/unpack_attachments.py            # unpack every attachment found in attachments/
    python tools/unpack_attachments.py --check    # report what is there, change nothing

Two archives of saved outputs are attached to the GitHub release instead of being kept in the repository, because
of their size (data/ATTACHMENTS.json lists them, with every member's SHA-256). Download them into attachments/ in
this folder. Nothing in `make check` needs them: exploration-activations.tar.xz holds exp03's saved activations,
which Figure 2 is drawn from, and large-records.tar.xz holds the round-4 train-split run's mean activations and the
per-item arrays of the post hoc reruns. A missing attachment is reported and skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import sys
import tarfile
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
FOLDER = RELEASE / "attachments"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    spec = json.loads((RELEASE / "data/ATTACHMENTS.json").read_text())
    failed = False
    for name, rec in sorted(spec.items()):
        arc = FOLDER / name
        if not arc.exists():
            print(f"{name}: not downloaded (attachments/{name}); skipped")
            continue
        packed = arc.read_bytes()
        if sha(packed) != rec["sha256"]:
            print(f"{name}: SHA-256 differs from data/ATTACHMENTS.json")
            failed = True
            continue
        counts = {}
        with tarfile.open(fileobj=io.BytesIO(lzma.decompress(packed))) as tar:
            for m in tar.getmembers():
                want = rec["members"].get(m.name)
                target = (RELEASE / m.name).resolve()
                if not m.isfile() or want is None or not str(target).startswith(str(RELEASE.resolve()) + "/"):
                    sys.exit(f"{name}: unexpected member {m.name}")
                if target.exists():
                    status = "present" if sha(target.read_bytes()) == want["sha256"] else "DIFFERS"
                elif args.check:
                    status = "absent"
                else:
                    data = tar.extractfile(m).read()
                    if sha(data) != want["sha256"]:
                        sys.exit(f"{name}: member {m.name} does not match its recorded SHA-256")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    status = "unpacked"
                failed |= status == "DIFFERS"
                counts[status] = counts.get(status, 0) + 1
        print(f"{name}: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
