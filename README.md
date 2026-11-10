# Line-breaking in Pythia-410M: code and data release

This folder holds the code, frozen protocols, held-out text and saved outputs behind the paper *Counting by
dilution and a width stored at the newline: a partial account of line-breaking in Pythia-410M* (version 6,
10 November 2026). Every number, table and figure in the paper can be regenerated from the saved outputs on a CPU, without a
model or network access, and `make check` runs the paper's 712 number checks against them.

## What the paper finds

When text is wrapped at a fixed width, a model that predicts the line breaks has to track how many characters the
current line holds, infer the width from earlier lines and compare the two. The study asked how Pythia-410M does
this. An exploration under a frozen charter proposed an account, and five confirmation rounds then tested its
predictions on held-out text, each with hypotheses and thresholds frozen before the text was read. Because the
predictions were written after exploration, the rounds show that the account repeats on new text. Three rounds used
Wikipedia articles created after the model was trained (such pages can still contain older text, so overlap
with the training data remains possible); the fifth also used Python code and German text.

- **Counting by dilution.** A head in layer 5 (L5H6) counts characters by how much attention it gives to the start
  of the line, which thins as the line grows; three other heads count with it. A second head (L5H7) removes a
  per-token bias from that count.
- **Storing the width at the newline.** At the newline that starts a line, the same counting heads record the length
  of the line that has just ended.
- **Reading it back.** Three later heads read that stored length at the decision point, one through the newline's
  value (L8H15) and one through its key (L17H1). The decision is assembled across layers 6 to 18, written mainly by
  one head and three MLPs, and held back by two neurons in MLPs 21 and 23 that suppress the newline as a break
  becomes likely.
- **What did not hold.** One prediction failed in all three rounds that tested it: at width 40 the intact model keeps
  a small weight on the number of tokens. The fifth round traced this to counting by dilution itself. The two width
  roles were confirmed at width 60; in the fifth round's eight further conditions of width and text type the
  readers held in five and the stored length was detectable in one.
- **How complete the account is.** The named circuit keeps the ranking of breaks but not their size. With 24
  components found by a search, its fidelity to the intact model's decision is 0.79 under mean ablation but only
  0.22 under resample ablation. Neither figure is a share of the mechanism understood, and the account is partial.
- **Other models.** GPT-2 Small carries a decodable count but uses it a quarter as strongly. Seven other open models
  (Pythia-1B, Qwen2.5-0.5B, OLMo-2-1B, Qwen3-1.7B-Base, Qwen3-0.6B-Base, Gemma-4-E2B, SmolLM2-1.7B) have heads in
  the same four roles, and all 29 frozen role tests passed, but the heads and their arrangement differ between model
  families.

Each result carries its label in the records and the paper: **exploratory** (under a charter), **confirmatory**
(under a protocol frozen before the held-out text was read) or **post hoc**.

## What is here

| Path | Contents |
|---|---|
| `CHARTER.md`, `AMENDMENTS.md` | The exploration charter of the main study and its three dated amendments |
| `PROTOCOL.md`, `PROTOCOL-2.md` ... `PROTOCOL-5.md` | The five confirmation protocols, each frozen before its held-out text was read |
| `REPORT.md`, `REPORT-2.md` ... `REPORT-5.md`, `POSTHOC.md` | The round reports, as written after each run, and the post hoc checks of rounds 1 and 2 |
| `FROZEN.md` | The freeze record: every frozen file with its SHA-256 and the time it was frozen |
| `code/` | Every script of the main study: exploration (`exp01`-`exp29`), the confirmation rounds (`conf0*`, `c2_*`, `c3_*`, `p4_*`, `p5_*`), the text fetchers, the post hoc checks and the report-table scripts |
| `xmodel/` | The cross-model part (paper Sections 9.2–9.4): its charter, amendments, freeze record, protocols X1, X2, X3 and X3b, reports, code, run specifications (`specs/`), job lists (`q*.txt`) and the third batch of new text (`data/newtext3/`) |
| `data/` | The text the study read: WikiText-2 (`wikitext2/`), the three batches of new English Wikipedia text (`newtext/`, `newtext2/`, `xmodel/data/newtext3/`), Python standard-library code (`code/`) and German Wikipedia text (`de/`), each split into frozen exploration and confirmation parts with a `MANIFEST.md` and `SHA256SUMS` |
| `data/saved-outputs/` | The per-run outputs, packed as five archives (see its `README.md`); two larger archives are attached to the GitHub release (`data/ATTACHMENTS.json`) |
| `paper/v6-2026-11-10/` | The paper's LaTeX source and the scripts that compute its numbers (`values.py`), write its tables (`make_tables.py`), draw its figures (`make_figures.py`) and check every number in the text (`check_paper.py`) |
| `models/` | The checkpoints the study used, with revisions and file digests (`models.json`), and copies of four model configuration files the paper's check reads |
| `tools/` | Unpacking (of the saved outputs and of the release attachments), the freeze-record check and model download helpers |
| `pyproject.toml`, `uv.lock`, `ENVIRONMENT.md` | The Python environment |
| `RELEASE-CHANGES.md`, `release-changes.json` | Every file that differs from the original study folder, with the original's SHA-256, and what was left out |

## Install

```sh
make setup        # uv sync --frozen: creates .venv from uv.lock (Python 3.12)
```

This downloads about 3.05 GB of packages (40 packages, mostly PyTorch and its CUDA libraries; measured from
`uv.lock`). `make check` needs only NumPy and PyTorch, and Matplotlib for the figures; `ENVIRONMENT.md` gives a
CPU-only route of a few hundred MB. Pass `PYTHON=/path/to/python` to any `make` target to use another environment.

## Check the paper's numbers (CPU, about 8 minutes, no model, no network)

```sh
make check
```

This unpacks the saved outputs and the compressed text files (checking every SHA-256 against
`data/PACKED-FILES.json`), checks both freeze records against the files they name, and runs the paper's own
`check_paper.py`. That script

1. regenerates every table of the paper from the saved outputs and requires it to match `tables/`;
2. cross-checks the values against the frozen round reports, which were written by the frozen table scripts;
3. reruns the frozen round-5 table script (`code/report5_tables.py`, with its bootstrap) and requires its table to
   equal the saved one and to appear line by line in `REPORT-5.md`;
4. asserts every number quoted in the text, and fails on any number no check covers.

Most of the time is spent rerunning the round-4 and round-5 bootstraps. The end of the output should read:

```text
total: 30 changed for the release (documented), 3 superseded by a later dated entry, 109 unchanged
712 checks, 0 failed
```

"Changed for the release" are frozen records and scripts in which personal or machine-specific wording was
replaced (see below); "superseded" are earlier digests of the two amendment files, which were frozen again after
each new amendment.

## Tables, figures and the PDF

```sh
make tables       # rewrite paper/v6-2026-11-10/tables/ from the saved outputs
make figures      # Figures 2-5 into paper/v6-2026-11-10/figures/ (Figure 1 is drawn in the LaTeX source)
make pdf          # the paper, with latexmk (needs a TeX installation)
```

Figure 4 uses the saved activations of exploration item exp03, which nothing else needs. They are too large for
the repository (275 MB compressed, 298 MB unpacked) and are attached to the GitHub release as
`exploration-activations.tar.xz`, with a second attachment, `large-records.tar.xz`, that holds the round-4
train-split run's mean activations and the per-item arrays of the post hoc reruns. Download both into
`attachments/`; `make figures` unpacks them with `make attachments`, which checks every member against
`data/ATTACHMENTS.json`.

## Rerunning the experiments

Nothing in the paper needs a model to run again, but every run can be repeated. The scripts in `code/` and
`xmodel/code/` are run from their own folder (`python <script>.py [arguments]`). They load their models from the
Hugging Face cache by repository name, with the Hub set offline, so download the checkpoints first:

```sh
make setup-experiment                          # the environment with transformers and its companions
python tools/cache_models.py --list            # every checkpoint, its revision and size
python tools/cache_models.py pythia-410m       # Pythia-410M, 0.91 GB: all five rounds of the main study
python tools/cache_models.py --all             # all 15 checkpoints, about 34 GB
```

The original launchers are kept as they ran, with machine paths replaced: `code/run.sh` and `xmodel/code/run.sh`
start one script in a memory-limited scope while holding a GPU lock, and `code/chain5.sh`, `code/chain5b.sh`,
`code/queue5.sh`, `xmodel/code/queue.sh` with the job lists `xmodel/q*.txt` give the exact arguments of each run.
Each protocol's "Order of runs" section lists its confirmation runs. Scripts write into `outputs/` (or
`xmodel/outputs/`), so rerun from a copy of this folder to keep the saved outputs for comparison.

All runs used one 6 GB NVIDIA GPU (compute capability 8.9): the main study's Pythia-410M runs in float32, the
cross-model runs in float32 or bfloat16 as each protocol states. The protocols give the expected GPU time of each
confirmation round: about an hour for round 1, one to one and a half hours for round 2, about 10 minutes for round 3
and about 30 minutes for round 4; PROTOCOL-X3b gives about 30 minutes for one cross-model confirmation. The exploration
items are mostly a few minutes each; the training-checkpoint item (exp26) reads six checkpoints of 0.91 GB, which
`tools/cache_models.py pythia-410m-step1000` (and so on) downloads to where that script reads them.

The text fetchers (`code/fetch_newtext.py`, `fetch_newtext2.py`, `fetch_de.py`, `xmodel/code/fetch_newtext3.py`)
query the Wikipedia API and would not return the same articles today; the fetched text is included and frozen.

## Material and licences

- **Code, records and saved outputs:** MIT (`LICENSE`).
- **WikiText-2** (`data/wikitext2/`, the parquet files of `Salesforce/wikitext`, configuration
  `wikitext-2-raw-v1`): CC BY-SA 3.0 and GFDL, as distributed by its authors.
- **English and German Wikipedia text** (`data/newtext/`, `data/newtext2/`, `data/de/`, `xmodel/data/newtext3/`):
  extracts of articles created in 2025 and 2026, fetched through the Wikipedia API on 6 November 2026 (UTC; each
  `fetch_summary.json` gives the time), under CC BY-SA 4.0. Each record keeps the article's title and page id, so
  the source can be attributed.
- **Python code text** (`data/code/`): paragraphs built from the Python 3.14 standard library, under the Python
  Software Foundation License.
- **Model configuration files** (`models/configs/`): from the model repositories, under Apache 2.0.
- **Checkpoints** are not included. Pythia, Qwen2.5, Qwen3, OLMo 2, SmolLM2 and Gemma 4 are distributed under
  Apache 2.0, GPT-2 under a modified MIT licence.

## Changes from the original study folder

The release was exported from the study folder in which the work ran. Personal and machine-specific strings
(absolute paths, local clock times, verbatim quotes of the author's messages, and wording about how the work was
coordinated) were replaced in some records and scripts, and two lines of `check_paper.py` read the model
configuration files from `models/configs/` instead of a local model cache. Every such file is listed in
`RELEASE-CHANGES.md` with its original SHA-256, and the freeze check accepts exactly those changes. Left out: the
exploration notebooks (dated working logs; the scripts and outputs they describe are included), run logs, smoke
tests, the superseded float16 runs, and the paper's build products and working notes.

## Use of language-model assistants

Language-model assistants were used throughout this work, under the author's direction, to help design the studies,
to write and run the code for the exploration, the experiments and the analyses, to draft and revise the text and to
check the references against their online records. All reported results come from the saved study outputs, and every
number in the paper was generated from those outputs and checked by an independent script.

## Citation

Richard Jeevash. *Counting by dilution and a width stored at the newline: a partial account of line-breaking in Pythia-410M*. Preprint, 2026. See `CITATION.cff`; the preprint's link will be added here when it is public.

This repository is archived at Zenodo: https://doi.org/10.5281/zenodo.24112443 (all releases; release v1.0 is
https://doi.org/10.5281/zenodo.24112444).
