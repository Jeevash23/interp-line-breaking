"""Shared helpers for D1 (line-breaking). Exploratory code.

Only the WikiText-2 *train* split is read here. Validation and test are reserved
for a later confirmation protocol (CHARTER.md).
"""
import os, re, random, json, hashlib
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "wikitext2"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MODELS = {
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "gpt2": "openai-community/gpt2",
}


# ---------------------------------------------------------------- text
_DETOK = [
    (" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."),
    (" , ", ", "), (" . ", ". "), (" ; ", "; "), (" : ", ": "),
    (" 's", "'s"), (" n't", "n't"), (" 're", "'re"), (" 've", "'ve"), (" 'll", "'ll"), (" 'd", "'d"), (" 'm", "'m"),
    ("( ", "("), (" )", ")"), ('" ', '"'),
]


def detok(s: str) -> str:
    s = s.strip()
    for a, b in _DETOK:
        s = s.replace(a, b)
    s = re.sub(r" ([.,;:!?%])", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    # fix the quote rule above, which glues an opening quote to the next word
    # but also a closing quote; leave as is (rare, harmless for character counting)
    return s.strip()


def train_paragraphs(min_chars=600, max_chars=1800, seed=0):
    """Plain-prose paragraphs from the WikiText-2 train split, detokenised."""
    import pyarrow.parquet as pq
    t = pq.read_table(DATA / "train-00000-of-00001.parquet").column("text").to_pylist()
    paras = []
    for line in t:
        s = line.strip()
        if not s or s.startswith("="):
            continue
        s = detok(s)
        if min_chars <= len(s) <= max_chars and "@" not in s and "<unk>" not in s:
            paras.append(s)
    rng = random.Random(seed)
    rng.shuffle(paras)
    return paras


def wrap(text: str, width: int):
    """Greedy word wrap. Returns list of lines or None if a word is longer than width."""
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        if len(w) > width:
            return None
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def split_paragraphs(split, min_chars=600, max_chars=1800, seed=0):
    """As train_paragraphs, for a named split ("train", "validation"). Added 6 Nov 2026 (day 2)
    for the confirmation protocol; the validation split is read only by the conf*.py scripts."""
    import pyarrow.parquet as pq
    t = pq.read_table(DATA / f"{split}-00000-of-00001.parquet").column("text").to_pylist()
    paras = []
    for line in t:
        s = line.strip()
        if not s or s.startswith("="):
            continue
        s = detok(s)
        if min_chars <= len(s) <= max_chars and "@" not in s and "<unk>" not in s:
            paras.append(s)
    rng = random.Random(seed)
    rng.shuffle(paras)
    return paras


# ---------------------------------------------------------------- models
DTYPES = {"pythia-410m": torch.float32, "gpt2": torch.float32, "pythia-1b": torch.bfloat16}


def load(name, dtype=None, device="cuda", attn="sdpa"):
    """float32 by default (Pythia's late residual stream overflows float16); bf16 for 1B."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    repo = MODELS[name]
    tok = AutoTokenizer.from_pretrained(repo)
    dtype = dtype or DTYPES[name]
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, attn_implementation=attn).to(device).eval()
    return tok, model


def newline_id(tok):
    ids = tok("\n", add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, ids
    return ids[0]


# ---------------------------------------------------------------- positions
def word_end_positions(tok, lines, width, first_line=2):
    """Tokenise the wrapped text and list the word-end decision points.

    A decision point is a token index i such that token i+1 is either the newline
    (a break, label 1) or a token starting with a space (a continuation, label 0).
    Only points on line index >= first_line are kept, so the width can be inferred
    from earlier lines.
    Returns (input_ids, records).
    """
    text = "\n".join(lines)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    # line index and column for each character
    line_of, col_of = [], []
    for li, ln in enumerate(lines):
        for ci in range(len(ln)):
            line_of.append(li); col_of.append(ci)
        line_of.append(li); col_of.append(len(ln))  # the newline char itself
    recs = []
    for i in range(len(ids) - 1):
        s1, e1 = offs[i + 1]
        if s1 >= len(text):
            continue
        nxt = text[s1:e1]
        _, e0 = offs[i]
        if e0 == 0 or e0 > len(text):
            continue
        prev_char = text[e0 - 1]
        if prev_char in (" ", "\n"):
            continue
        li = line_of[e0 - 1]
        if li < first_line:
            continue
        if nxt == "\n" or (nxt.startswith("\n")):
            label = 1
            nextword = lines[li + 1].split(" ")[0] if li + 1 < len(lines) else None
        elif nxt.startswith(" "):
            label = 0
            rest = text[s1 + 1:]
            nextword = re.split(r"[ \n]", rest, maxsplit=1)[0]
        else:
            continue
        if nextword is None:
            continue
        used = col_of[e0 - 1] + 1  # characters used on this line up to and including token i
        recs.append(dict(i=i, line=li, used=used, width=width, remaining=width - used,
                         label=label, nextlen=len(nextword), prevlen=len(lines[li - 1]),
                         ntok_line=None))
    # tokens used on the current line so far
    line_start_tok = {}
    for i in range(len(ids)):
        s, e = offs[i]
        if s < len(text):
            li = line_of[s]
            line_start_tok.setdefault(li, i)
    for r in recs:
        r["ntok_line"] = r["i"] - line_start_tok[r["line"]] + 1
    return ids, recs


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ties
    order = np.argsort(allv)
    sv = allv[order]
    r = np.empty(len(allv)); i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = r[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def save_json(obj, name):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1))
    return p
