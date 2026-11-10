# Model checkpoints

No weights or tokeniser files are included, except four `config.json` files in `configs/`, from which the paper's
check reads the architecture details it states (layers, heads, hidden sizes). The study code loads checkpoints from
the Hugging Face cache by repository name, with the Hub set offline. `tools/cache_models.py` downloads a checkpoint
into that cache at the revision the study used and stops if any file's SHA-256 differs from `models.json`:

| Name | Hugging Face repository | Revision | Size | Used by |
|---|---|---|---|---|
| `pythia-410m` | `EleutherAI/pythia-410m` | `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` | 0.91 GB | every round of the main study (Sections 4-7) |
| `pythia-1b` | `EleutherAI/pythia-1b` | `f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2` | 2.09 GB | exploration exp01; cross-model PROTOCOL-X1 |
| `gpt2` | `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 0.55 GB | exploration exp01, exp24; round-4 test X |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | 1.00 GB | cross-model PROTOCOL-X1 |
| `olmo-2-1b` | `allenai/OLMo-2-0425-1B` | `a1847dff35000b4271fa70afc5db10fd29fedbdf` | 5.95 GB | cross-model PROTOCOL-X1 and PROTOCOL-X2 |
| `qwen3-1.7b-base` | `Qwen/Qwen3-1.7B-Base` | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | 3.45 GB | cross-model PROTOCOL-X3 |
| `qwen3-0.6b-base` | `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | 1.20 GB | cross-model PROTOCOL-X3 |
| `gemma-4-e2b` | `google/gemma-4-E2B` | `433813a669639fd7eaa83155828e9000ed357d53` | 10.28 GB | cross-model PROTOCOL-X3b |
| `smollm2-1.7b` | `HuggingFaceTB/SmolLM2-1.7B` | `effd688a12921b4cc83e3312b6feb579f70f9c71` | 3.43 GB | cross-model PROTOCOL-X3b (SmolLM2 addendum) |
| `pythia-410m-step1000` | `EleutherAI/pythia-410m` | `step1000` | 0.91 GB | training checkpoints, exploration exp26 (Section 8.1); the code reads them from ~/.cache/d1-checkpoints/step1000 |
| `pythia-410m-step2000` | `EleutherAI/pythia-410m` | `step2000` | 0.91 GB | training checkpoints, exploration exp26 (Section 8.1); the code reads them from ~/.cache/d1-checkpoints/step2000 |
| `pythia-410m-step4000` | `EleutherAI/pythia-410m` | `step4000` | 0.91 GB | training checkpoints, exploration exp26 (Section 8.1); the code reads them from ~/.cache/d1-checkpoints/step4000 |
| `pythia-410m-step8000` | `EleutherAI/pythia-410m` | `step8000` | 0.91 GB | training checkpoints, exploration exp26 (Section 8.1); the code reads them from ~/.cache/d1-checkpoints/step8000 |
| `pythia-410m-step16000` | `EleutherAI/pythia-410m` | `step16000` | 0.91 GB | training checkpoints, exploration exp26 (Section 8.1); the code reads them from ~/.cache/d1-checkpoints/step16000 |
| `pythia-410m-step48000` | `EleutherAI/pythia-410m` | `step48000` | 0.91 GB | training checkpoints, exploration exp26 (Section 8.1); the code reads them from ~/.cache/d1-checkpoints/step48000 |

Total: about 34.3 GB. The six `pythia-410m-step*` entries are training checkpoints of Pythia-410M (exploration
item exp26), which `code/exp26_checkpoints.py` reads from `~/.cache/d1-checkpoints/<step>/`; `tools/cache_models.py`
puts them there. `models.json` lists every file with its SHA-256, taken from the copies the runs used.

```sh
python tools/cache_models.py --list                  # the table above
python tools/cache_models.py pythia-410m             # the main study's model (make models)
python tools/cache_models.py --all                   # every checkpoint
python tools/cache_models.py --check pythia-410m     # check what is in the cache, without network access
```

`tools/get_models.py`, the download helper shared with other releases, reads the same `models.json`; it downloads
into `models/<name>/`, which the study code does not read, so use `tools/cache_models.py` for reruns.

The checkpoints are distributed under their own licences: Apache 2.0 for Pythia, Qwen2.5, Qwen3, OLMo 2, SmolLM2
and Gemma 4, and a modified MIT licence for GPT-2. The four `config.json` files included here are from the Qwen3,
Gemma 4 and SmolLM2 repositories (Apache 2.0).
