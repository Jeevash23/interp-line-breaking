"""Loader for the new-text parts (data/newtext/part-<name>.jsonl), written by fetch_newtext.py.
load_paras("train", seed) returns WikiText-2 train paragraphs (for --smoke checks only);
load_paras("<part name>", seed) returns that part's paragraphs, shuffled with the given seed."""
import json, random
from pathlib import Path
from common import split_paragraphs
NEW = Path(__file__).resolve().parent.parent / "data" / "newtext"
def load_paras(split, seed=0):
    if split == "train": return split_paragraphs("train", seed=seed)
    ps = [json.loads(l)["text"] for l in open(NEW / f"part-{split}.jsonl")]
    random.Random(seed).shuffle(ps)
    return ps
