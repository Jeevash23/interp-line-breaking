# Environment

The experiments ran in one Python 3.12 environment on a single 6 GB NVIDIA GPU, and the paper's numbers and figures
were computed in a second one with the same core packages. `pyproject.toml` declares the packages the code imports,
pinned to the versions of the experiment environment; `uv.lock` records the whole environment, with every other
package held to the version installed where the study ran (`constraint-dependencies` in `pyproject.toml`).

| Package | Version | Needed for |
|---|---|---|
| Python | 3.12.14 (`uv` installs it if it is missing) | everything |
| numpy | 2.5.3 | `make check`, the experiments |
| torch | 2.14.1 (CUDA 13 build from PyPI) | `make check` (round 4 reads a saved tensor file), the experiments |
| matplotlib | 3.11.1 | `make figures` |
| transformers | 5.19.0 | the experiments (`experiment` extra) |
| accelerate, safetensors, tokenizers, huggingface-hub, pyarrow | 1.15.0, 0.8.0, 0.23.3, 2.1.1, 25.0.1 | the experiments (`experiment` extra) |

The paper's numbers were checked with numpy 2.5.2 and torch 2.13.0 before this release, and with the versions above
in a copy of this release; both gave 649 of 649 checks (paper version 3). Paper version 4 (10 November 2026) gave 648 of 648 checks, paper version 5 (10 November 2026) 661 of 661 checks, and paper version 6 (10 November 2026) 712 of 712 checks, with the versions above.

## A fresh install

```sh
uv sync --frozen                      # what make setup runs: 40 packages, about 3.05 GB of wheels
uv sync --frozen --extra experiment   # what make setup-experiment runs: 65 packages, about 3.13 GB
```

The sizes are for Linux x86-64 and come from `uv.lock`. Most of the download is PyTorch (555 MB) and the NVIDIA CUDA
libraries it depends on. Installing creates `.venv/` in this folder.

## A smaller CPU-only environment

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv numpy==2.5.3 matplotlib==3.11.1
uv pip install --python .venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

This is enough for `make check`, `make tables` and `make figures` (a few hundred MB). It is not the locked
environment; for model reruns use `make setup-experiment`.

## Models

The code loads checkpoints from the Hugging Face cache by repository name and sets the Hub offline
(`code/common.py`, `xmodel/code/xm_common.py`), so the checkpoints must be in the cache first:
`tools/cache_models.py` puts them there at the revisions the study used and checks every file (see
`models/README.md`).
