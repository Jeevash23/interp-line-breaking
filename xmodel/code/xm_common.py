"""D1-X shared helpers (exploratory and confirmatory scripts import this). Model adapters for
GPT-NeoX (Pythia) and Qwen2, head-level hooks on the attention output projection's input, decision
points restricted to letter-ending word ends (CHARTER.md), and the new-text loader."""
import os, sys, json, random
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
D1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(D1 / "code"))
import numpy as np, torch
from common import wrap, word_end_positions, auroc, newline_id, train_paragraphs, split_paragraphs
from newtext import load_paras
OUT = Path(__file__).resolve().parent.parent / "outputs"; OUT.mkdir(exist_ok=True)
MODELS = {"pythia-410m": ("EleutherAI/pythia-410m", torch.float32), "pythia-1b": ("EleutherAI/pythia-1b", torch.bfloat16),
          "qwen05b": ("Qwen/Qwen2.5-0.5B", torch.float32), "olmo1b": ("allenai/OLMo-2-0425-1B", torch.bfloat16)}
if os.environ.get("XM_DTYPE") == "float32":  # CPU / float32 spot checks (AMENDMENTS.md, Amendment 1)
    MODELS = {k: (v[0], torch.float32) for k, v in MODELS.items()}
DEV = os.environ.get("XM_DEVICE", "cuda")

class M:
    """Loaded model with per-head hooks. self.patch[l] = list of (head, positions tensor, values);
    self.capture=True stores the o-proj input of every layer in self.cap[l] (T, H*dh)."""
    def __init__(self, name, attn="sdpa"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        repo, dt = MODELS[name]
        self.name = name; self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=dt, attn_implementation=attn).to(DEV).eval()
        c = self.model.config
        self.L, self.H = c.num_hidden_layers, c.num_attention_heads
        self.dh = getattr(c, "head_dim", None) or c.hidden_size // self.H
        if hasattr(self.model, "gpt_neox"):
            self.layers = self.model.gpt_neox.layers; self.oproj = [l.attention.dense for l in self.layers]
        else:
            self.layers = self.model.model.layers; self.oproj = [l.self_attn.o_proj for l in self.layers]
        self.nl = newline_id(self.tok)
        self.patch, self.cap, self.capture, self.rpatch, self.rcap = {}, {}, False, {}, {}
        for l, m in enumerate(self.oproj): m.register_forward_pre_hook(self._zhook(l))
        for l, m in enumerate(self.layers): m.register_forward_hook(self._rhook(l))
    def _zhook(self, l):
        def f(m, a):
            z = a[0]
            if self.capture: self.cap[l] = z[0].detach().clone()
            lst = self.patch.get(l)
            if not lst: return None
            z = z.clone()
            for e in lst:
                h, pos, val = e[:3]; row = e[3] if len(e) > 3 else slice(None)
                z[row, pos, h * self.dh:(h + 1) * self.dh] = val.to(z.dtype)
            return (z,) + tuple(a[1:])
        return f
    def _rhook(self, l):
        def f(m, a, o):
            x = o[0] if isinstance(o, tuple) else o
            if self.capture: self.rcap[l] = x[0].detach().clone()
            lst = self.rpatch.get(l)
            if not lst: return None
            x = x.clone()
            for e in lst:
                pos, val = e[:2]; row = e[2] if len(e) > 2 else slice(None)
                x[row, pos] = val.to(x.dtype)
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        return f
    def lp(self, ids, capture=False):
        """log P(newline) at every position (numpy, float32)."""
        self.capture = capture
        with torch.no_grad():
            out = torch.log_softmax(self.model(torch.tensor([ids], device=DEV)).logits[0].float(), -1)[:, self.nl].cpu().numpy()
        self.capture = False
        return out
    def lp_last_batch(self, ids, B):
        """log P(newline) at the last position for B copies of ids (rows patched separately)."""
        with torch.no_grad():
            lg = self.model(torch.tensor([ids] * B, device=DEV), logits_to_keep=1).logits[:, -1].float()
        return torch.log_softmax(lg, -1)[:, self.nl].cpu().numpy()
    def nlpos(self, ids):
        """Line-start marker positions: tokens whose text contains a newline. In Pythia this is the
        newline token; in Qwen it may be fused with punctuation ('.\\n', ';\\n')."""
        if not hasattr(self, "_nlc"): self._nlc = {}
        out = []
        for i, t in enumerate(ids):
            if t not in self._nlc: self._nlc[t] = "\n" in self.tok.decode([t])
            if self._nlc[t]: out.append(i)
        return out
    def heads(self): return [(l, h) for l in range(self.L) for h in range(self.H)]

def logodds(v): return v - np.log1p(-np.exp(np.minimum(v, -1e-6)))

def decision_points(tok, lines, W):
    """common.word_end_positions, keeping only word ends whose last character is a letter."""
    ids, recs = word_end_positions(tok, lines, W)
    text = "\n".join(lines)
    offs = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    keep = [r for r in recs if text[offs[r["i"]][1] - 1].isalpha()]
    return ids, keep

def wrapped_items(paras, W, n, min_lines=5, max_tok=700, tok=None):
    out = []
    for p in paras:
        if len(out) >= n: break
        lines = wrap(p, W)
        if lines is None or len(lines) < min_lines: continue
        ids, recs = decision_points(tok, lines, W)
        if len(ids) > max_tok: continue
        out.append((ids, recs, lines))
    return out

def width_items(paras, m, n, W=60, seed=200):
    tok = m.tok
    """As conf02/c2_02: previous line shortened by >= 10 characters; decision at a letter-ending
    word end with 4..14 characters left."""
    rng = random.Random(seed); items = []
    for p in paras:
        if len(items) >= n: break
        lines = wrap(p, W)
        if lines is None or len(lines) < 6: continue
        k = rng.randrange(3, min(6, len(lines) - 1)); words = lines[k].split(" ")
        js = [j for j in range(2, len(words)) if 4 <= W - len(" ".join(words[: j + 1])) <= 14 and words[j][-1:].isalpha()]
        if not js: continue
        prefix = " ".join(words[: js[0] + 1]); before = lines[:k]
        pw = before[-1].split(" "); short = before[-1]
        while len(pw) > 2 and len(before[-1]) - len(short) < 10: pw = pw[:-1]; short = " ".join(pw)
        if len(before[-1]) - len(short) < 10: continue
        ib = tok("\n".join(before) + "\n" + prefix, add_special_tokens=False)["input_ids"]
        is_ = tok("\n".join(before[:-1] + [short]) + "\n" + prefix, add_special_tokens=False)["input_ids"]
        pb, ps = m.nlpos(ib), m.nlpos(is_)
        if not pb or not ps or len(pb) != len(ps): continue
        nb, ns = pb[-1], ps[-1]
        if len(ib) - nb != len(is_) - ns or ib[nb] != is_[ns]: continue
        items.append((ib, is_, nb, ns))
    return items

def spline(rem): return np.c_[rem, *[np.maximum(rem - k, 0) for k in [2, 5, 8, 12, 16, 24, 32]]]
def token_coef(X, y):
    Mx = np.c_[np.ones(len(y)), spline(X[:, 0]), X[:, 1]]
    return float(np.linalg.lstsq(Mx, y, rcond=None)[0][-1])

def save_json(obj, name):
    (OUT / name).write_text(json.dumps(obj, indent=1)); return OUT / name
