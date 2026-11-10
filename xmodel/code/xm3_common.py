"""D1-X Amendment 3 helpers: the newer models (Qwen3-0.6B-Base, Qwen3-1.7B-Base, Gemma-4-E2B,
SmolLM2-1.7B). Re-exports the frozen xm_common.py helpers unchanged and replaces three things:
 * MODELS and the model class M (generic layer finding; Gemma's text decoder on the GPU with its
   per-layer embedding table on the CPU; a beginning-of-sequence token for Gemma);
 * load_paras, which reads the third English batch (data/newtext3/) and refuses a confirmation part
   until PROTOCOL-X3.md exists and is read-only (frozen).
Beginning-of-sequence token (AMENDMENTS.md, Amendment 3): for models with BOS set, m.model(...)
prepends it and strips it from logits and attentions; the head and residual hooks shift every
non-negative position by one, so position 0 (the BOS) is never ablated or patched; captures
(m.cap, m.rcap) and m.lp() refer to text tokens only. Scripts therefore index text tokens exactly as
for the earlier models. m.off is 1 with a BOS, 0 without."""
import os, json, random, stat
from pathlib import Path
from xm_common import *            # noqa: F401,F403  (frozen helpers: wrap, decision_points, width_items, ...)
import xm_common as _xc
import numpy as np, torch

MODELS = dict(_xc.MODELS)
MODELS.update({"qwen3-06b": ("Qwen/Qwen3-0.6B-Base", torch.bfloat16), "qwen3-17b": ("Qwen/Qwen3-1.7B-Base", torch.bfloat16),
               "gemma4-e2b": ("google/gemma-4-E2B", torch.bfloat16), "smollm2-17b": ("HuggingFaceTB/SmolLM2-1.7B", torch.bfloat16)})
if os.environ.get("XM_DTYPE") == "float32":
    MODELS = {k: (v[0], torch.float32) for k, v in MODELS.items()}
DEV = os.environ.get("XM_DEVICE", "cuda")
BOS_MODELS = {"gemma4-e2b"}

X = Path(__file__).resolve().parent.parent
NT3 = X / "data" / "newtext3"
# Which frozen protocol must exist before each confirmation part may be read (set 7 Nov 2026 before
# either protocol was written): the two Qwen3 models under PROTOCOL-X3, Gemma and SmolLM2 under X3b.
CONF_PROTOCOL = {"conf-qwen3-06b": "PROTOCOL-X3.md", "conf-qwen3-17b": "PROTOCOL-X3.md",
                 "conf-gemma4-e2b": "PROTOCOL-X3b.md", "conf-smollm2-17b": "PROTOCOL-X3b.md"}
def load_paras(split, seed=0):
    """'train' = WikiText-2 train; otherwise part-<split>.jsonl of the third batch. A confirmation part
    ('conf-*') loads only once its protocol (CONF_PROTOCOL) exists and is not writable (frozen)."""
    if split == "train": return split_paragraphs("train", seed=seed)
    if split.startswith("conf-"):
        p = X / CONF_PROTOCOL[split]
        if not p.exists() or (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)):
            raise SystemExit(f"refusing to read {split}: {p.name} is not frozen")
    ps = [json.loads(l)["text"] for l in open(NT3 / f"part-{split}.jsonl")]
    random.Random(seed).shuffle(ps)
    return ps


class _BosModel:
    """Callable stand-in for the HF model that prepends BOS and strips it from the outputs. A call with
    output_attentions=True runs only the base model (no output layer): callers use only the attention
    weights, and this keeps Gemma's 262k-wide logits out of GPU memory."""
    def __init__(self, model, bos): self._m, self._bos = model, bos
    def __getattr__(self, k): return getattr(self._m, k)
    def __call__(self, input_ids, **kw):
        b = torch.full((input_ids.shape[0], 1), self._bos, dtype=input_ids.dtype, device=input_ids.device)
        x = torch.cat([b, input_ids], 1)
        if kw.get("output_attentions"):
            o = self._m.model(input_ids=x, **kw)
        else:
            o = self._m(x, **kw)
            if not kw.get("logits_to_keep"): o.logits = o.logits[:, 1:]
        if getattr(o, "attentions", None) is not None: o.attentions = tuple(a[..., 1:, 1:] for a in o.attentions)
        return o


class _RowTable(torch.nn.Module):
    """Gemma-4's per-layer token-embedding lookup, reading only the needed rows from the safetensors
    file (memory-mapped). Same output as Gemma4TextScaledWordEmbedding: row * embed_scale in bfloat16."""
    def __init__(self, path, key, scale, dt):
        super().__init__()
        from safetensors import safe_open
        self._f = safe_open(str(path), "pt", device="cpu"); self._sl = self._f.get_slice(key); self._dt = dt
        self.scale = scale; self.embedding_dim = self._sl.get_shape()[1]
    def forward(self, ids):
        u, inv = torch.unique(ids.cpu(), return_inverse=True)
        rows = torch.cat([self._sl[int(i):int(i) + 1] for i in u]).to(self._dt)
        out = rows[inv] * torch.tensor(self.scale).to(self._dt)
        return out.to(ids.device)


def _shift(pos, off):
    if off == 0: return pos
    if isinstance(pos, slice):
        st = off if pos.start is None else (pos.start + off if pos.start >= 0 else pos.start)
        sp = pos.stop if (pos.stop is None or pos.stop < 0) else pos.stop + off
        return slice(st, sp, pos.step)
    if isinstance(pos, (int, np.integer)): return int(pos) + off if pos >= 0 else int(pos)
    t = torch.as_tensor(pos)
    return torch.where(t >= 0, t + off, t)


def _find_layers(model, n):
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n and name.endswith("layers") and hasattr(mod[0], "self_attn"):
            if "vision" in name or "audio" in name: continue
            best = mod
    return best


class M:
    """As xm_common.M (same attributes and hooks), for the Amendment-3 models."""
    def __init__(self, name, attn="sdpa"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        repo, dt = MODELS[name]
        self.name = name; self.tok = AutoTokenizer.from_pretrained(repo)
        if name == "gemma4-e2b":
            self.model = self._load_gemma(repo, dt, attn)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=dt, attn_implementation=attn).to(DEV).eval()
        c = self.model.config; c = getattr(c, "text_config", None) or c
        self.L, self.H = c.num_hidden_layers, c.num_attention_heads
        self.layers = _find_layers(self.model, self.L)
        self.oproj = [l.self_attn.o_proj for l in self.layers]
        # per-layer head width: Gemma-4's global layers have wider heads (512) than its sliding-window layers (256)
        self.dhl = [o.in_features // self.H for o in self.oproj]
        self.dh = max(set(self.dhl), key=self.dhl.count)  # the common width (all layers for Qwen and SmolLM2)
        self.off = 1 if name in BOS_MODELS else 0
        if self.off:
            self.model = _BosModel(self.model, self.tok.bos_token_id)
        self.nl = newline_id(self.tok)
        self.patch, self.cap, self.capture, self.rpatch, self.rcap = {}, {}, False, {}, {}
        for l, m in enumerate(self.oproj): m.register_forward_pre_hook(self._zhook(l))
        for l, m in enumerate(self.layers): m.register_forward_hook(self._rhook(l))

    def _load_gemma(self, repo, dt, attn):
        """Text decoder on the GPU in bfloat16; nothing is quantised. Built directly: an empty model,
        the image and audio encoders removed (never run), the per-layer embedding table (a lookup,
        262,144 x 8,960, 4.7 GB) replaced by _RowTable, which reads the rows of the tokens in each input
        from the weights file (same values, same scale, same bfloat16 arithmetic), and every other
        text weight read from the weights file straight onto the GPU. The output layer is tied to the
        token embedding, as in the checkpoint. The key and value weights stored for layers 15-34 are
        not used by the model (those layers reuse earlier layers' keys and values) and are not loaded."""
        from transformers import AutoModelForCausalLM, AutoConfig
        from huggingface_hub import snapshot_download
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        cfg = AutoConfig.from_pretrained(repo)
        st = Path(snapshot_download(repo, local_files_only=True)) / "model.safetensors"
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg, attn_implementation=attn, dtype=dt)
        base = model.model
        for n in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"): setattr(base, n, None)
        lm = base.language_model
        lm.embed_tokens_per_layer = _RowTable(st, "model.language_model.embed_tokens_per_layer.weight",
                                              float(lm.embed_tokens_per_layer.scalar_embed_scale), dt)
        dev = 0 if DEV == "cuda" else DEV
        with safe_open(str(st), "pt", device="cpu") as f:
            have = set(f.keys())
            for k in list(model.state_dict()):
                if k == "lm_head.weight": continue
                assert k in have, k
                t = f.get_tensor(k)
                set_module_tensor_to_device(model, k, dev, value=t.to(dt) if t.is_floating_point() else t)
        model.lm_head.weight = lm.embed_tokens.weight
        model = model.to(DEV).eval()
        assert not any(p.is_meta for p in model.parameters()) and not any(b.is_meta for b in model.buffers())
        return model

    def _logit_chunks(self, ids, n=128):
        """Next-token logits (float32) for the text positions, in chunks of n positions. Gemma: the
        output layer and the final soft-capping are applied per chunk to the base model's last hidden
        state (same computation as the model's own head)."""
        x = torch.tensor([ids], device=DEV)
        if not self.off:
            for c in self.model(x).logits[0].split(n): yield c.float()
            return
        raw = self.model._m
        b = torch.full((1, 1), self.tok.bos_token_id, dtype=x.dtype, device=x.device)
        h = raw.model(input_ids=torch.cat([b, x], 1)).last_hidden_state[0, 1:]
        cap = raw.config.get_text_config().final_logit_softcapping
        for c in h.split(n):
            lg = raw.lm_head(c)
            if cap is not None: lg = torch.tanh(lg / cap) * cap
            yield lg.float()

    def _zhook(self, l):
        def f(m, a):
            z = a[0]
            if self.capture: self.cap[l] = z[0, self.off:].detach().clone()
            lst = self.patch.get(l)
            if not lst: return None
            z = z.clone()
            for e in lst:
                h, pos, val = e[:3]; row = e[3] if len(e) > 3 else slice(None)
                z[row, _shift(pos, self.off), self.hs(l, h)] = val.to(z.dtype).to(z.device)
            return (z,) + tuple(a[1:])
        return f

    def _rhook(self, l):
        def f(m, a, o):
            x = o[0] if isinstance(o, tuple) else o
            if self.capture: self.rcap[l] = x[0, self.off:].detach().clone()
            lst = self.rpatch.get(l)
            if not lst: return None
            x = x.clone()
            for e in lst:
                pos, val = e[:2]; row = e[2] if len(e) > 2 else slice(None)
                x[row, _shift(pos, self.off)] = val.to(x.dtype).to(x.device)
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        return f

    def lp(self, ids, capture=False):
        """log P(newline) at every text position (numpy, float32); log-softmax taken in chunks."""
        self.capture = capture
        with torch.no_grad():
            out = torch.cat([(c[:, self.nl] - torch.logsumexp(c, -1)) for c in self._logit_chunks(ids)]).cpu().numpy()
        self.capture = False
        return out

    def nll(self, ids):
        """Next-token loss at text positions 0..T-2 (predicting ids[1:]), numpy; in chunks."""
        t = torch.tensor(ids[1:] + [0], device=DEV); out, k = [], 0
        with torch.no_grad():
            for c in self._logit_chunks(ids):
                tt = t[k:k + len(c)]; out.append(torch.logsumexp(c, -1) - c.gather(1, tt[:, None])[:, 0]); k += len(c)
        return torch.cat(out)[:-1].cpu().numpy()

    def lp_last_batch(self, ids, B):
        with torch.no_grad():
            lg = self.model(torch.tensor([ids] * B, device=DEV), logits_to_keep=1).logits[:, -1].float()
        return torch.log_softmax(lg, -1)[:, self.nl].cpu().numpy()

    def hs(self, l, h):
        """Slice of head h in layer l's o-proj input."""
        return slice(h * self.dhl[l], (h + 1) * self.dhl[l])

    nlpos = _xc.M.nlpos
    heads = _xc.M.heads
