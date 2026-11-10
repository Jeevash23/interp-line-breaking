"""Download the checkpoints the study used, at the pinned revisions, and check every file's SHA-256.

    python tools/cache_models.py --list                 # each checkpoint, its size and what used it
    python tools/cache_models.py pythia-410m            # one model (needs network access)
    python tools/cache_models.py --all                  # every checkpoint in models/models.json (about 34 GB)
    python tools/cache_models.py --check pythia-410m    # check what is already there (no network)

The study code loads its models from the Hugging Face cache by repository name, with the Hub set offline
(code/common.py), so this script downloads into that cache (HF_HOME, or ~/.cache/huggingface by default) at
the revision listed in models/models.json. The six Pythia-410M training checkpoints of exploration item exp26
(names pythia-410m-step1000 to pythia-410m-step48000) go to ~/.cache/d1-checkpoints/<step>/, where
code/exp26_checkpoints.py reads them. If a repository asks you to accept its terms, accept them on the Hub and log
in (huggingface-cli login) before downloading it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
SPEC = json.loads((RELEASE / "models/models.json").read_text())["models"]
CKPT = Path.home() / ".cache" / "d1-checkpoints"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def target(name, fname, download):
    from huggingface_hub import hf_hub_download
    s = SPEC[name]
    if s["revision"].startswith("step"):
        local = CKPT / s["revision"]
        if download:
            local.mkdir(parents=True, exist_ok=True)
            return Path(hf_hub_download(s["repo"], fname, revision=s["revision"], local_dir=str(local)))
        return local / fname
    return Path(hf_hub_download(s["repo"], fname, revision=s["revision"], local_files_only=not download))


def run(name, download):
    bad = []
    for fname, meta in SPEC[name]["files"].items():
        try:
            p = target(name, fname, download)
        except Exception as e:  # noqa: BLE001 (missing file, offline cache miss, gated repository)
            bad.append(f"{fname}: {type(e).__name__}")
            continue
        if not p.exists():
            bad.append(f"{fname}: missing")
        elif sha256(p) != meta["sha256"]:
            bad.append(f"{fname}: SHA-256 differs from the record")
    s = SPEC[name]
    print(f"{name}: {s['repo']}@{s['revision'][:12]}: " + ("all files present and matching" if not bad else "; ".join(bad)))
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true", help="check only; download nothing")
    args = ap.parse_args()
    names = list(SPEC) if args.all else args.names
    if args.list or not names:
        for n, s in SPEC.items():
            size = sum(m["bytes"] for m in s["files"].values()) / 1e9
            print(f"{n:24s} {s['repo']}@{s['revision'][:12]}  {size:5.2f} GB  ({s['used_by']})")
        return
    unknown = [n for n in names if n not in SPEC]
    if unknown:
        sys.exit(f"unknown name(s): {unknown}; see --list")
    ok = all([run(n, not args.check) for n in names])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
