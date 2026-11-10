"""Round-5 text loader (AMENDMENTS.md, Amendment 2). load(split, seed) returns paragraphs shuffled with seed.
  train                : WikiText-2 train (common.split_paragraphs)
  reserve-pythia410m   : the unread part of the second English batch (data/newtext2/) -- confirmation only
  explore-code, conf5-code : Python standard-library code, whitespace collapsed (data/code/, build_code.py)
  explore-de, conf5-de : German Wikipedia, new articles 2025-26 (data/de/, fetch_de.py)"""
import json, random
from pathlib import Path
from common import split_paragraphs
D = Path(__file__).resolve().parent.parent / "data"
FILES = {"reserve-pythia410m": D / "newtext2" / "part-reserve-pythia410m.jsonl",
         "explore-code": D / "code" / "part-explore-code.jsonl", "conf5-code": D / "code" / "part-conf5-code.jsonl",
         "explore-de": D / "de" / "part-explore-de.jsonl", "conf5-de": D / "de" / "part-conf5-de.jsonl"}
CONF = {"reserve-pythia410m", "conf5-code", "conf5-de"}
def load(split, seed=0):
    if split == "train": return split_paragraphs("train", seed=seed)
    ps = [json.loads(l)["text"] for l in open(FILES[split])]
    random.Random(seed).shuffle(ps)
    return ps
