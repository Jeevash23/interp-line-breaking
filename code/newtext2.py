"""Paragraph loader for D1 round 4. split "train" gives WikiText-2 train paragraphs (common.train_paragraphs);
"conf4-pythia410m" gives the second new-text batch's round-4 part (data/newtext2/), shuffled with the seed.
The reserve part is never loaded here."""
import json, random
from pathlib import Path
from common import train_paragraphs
NEW2 = Path(__file__).resolve().parent.parent / "data" / "newtext2"
def paras(split, seed):
    if split == "train": return train_paragraphs(seed=seed)
    assert split == "conf4-pythia410m", split
    ps = [json.loads(l)["text"] for l in open(NEW2 / f"part-{split}.jsonl")]
    random.Random(seed).shuffle(ps)
    return ps
