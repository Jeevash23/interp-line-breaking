# D1-X charter: do other small models break lines with the same roles?

Area 14, study D1, cross-model part (D1-X). Drafted on 6 Nov 2026 at the author's request ("New text, then models"); finished and frozen on 6 Nov 2026, before any exploration ran under it. The digest is in `FROZEN.md` in this folder (`../FROZEN.md` belongs to the Pythia-410M rounds). The D1 charter (`../CHARTER.md`) covers Pythia-410M only, so this charter is needed.

Everything done under this charter is **exploratory**. A candidate becomes a result only after a frozen confirmation protocol tests it on that model's held-out confirmation part of the new text.

Before the freeze, pipeline checks were run of `code/xm_e01_scan.py` and `code/xm_e02_width.py` on a handful of WikiText-2 train paragraphs (`--smoke`, 4–6 items per measure). Their outputs are labelled `_train_smoke`; they are pipeline checks, not results, and no threshold below was set from them.

## Question
In Pythia-410M, confirmed on two held-out WikiText-2 splits (`../REPORT.md`, `../REPORT-2.md`; a third round on new text, `../PROTOCOL-3.md`, had not run when this charter was frozen), line-breaking in fixed-width prose uses four roles:
- **Counters:** heads that count the characters on the current line at word ends. Ablating them everywhere destroys the break/no-break separation.
- **A stored length at the newline:** the same heads, run at the line-start newline token, record the length of the line just ended. Ablating them at newline tokens only removes much of the effect of the previous line's length and leaves the count working.
- **Token correctors:** heads whose ablation gives the number of tokens on the line a positive weight in the break decision.
- **Readers:** heads at the decision token that carry the stored length from the newline to the decision. Freezing them removes most of the previous line's effect.

The question is whether Pythia-1B (same family and tokeniser, larger) and Qwen2.5-0.5B (different family, tokeniser and training data) have components in these four roles, and how the roles are arranged. Head numbers are not expected to match across models. Any answer counts, including:
- no clear roles;
- roles spread over many components;
- a different arrangement, such as the width held in MLPs, or no stored length at the newline.

## Models
- **Pythia-1B** (EleutherAI, final checkpoint, on disk): 16 layers × 8 heads. bfloat16, as in exp15. A float32 spot check of the main measures is allowed where it fits in memory; if the two disagree materially, the float32 value is used and the difference noted.
- **Qwen2.5-0.5B** (base, on disk, 0.99 GB): 24 layers × 14 query heads sharing 2 key/value heads. float32. Head-level ablation acts on each query head's output; key and value routes are shared within a group of 7 heads, so key-only or value-only tests in Qwen are at the level of the group, not the head.
- **OLMo-2-1B is not on disk** (only its configuration file). Its weights are a download of several GB, so it is parked for the author. Its two text parts stay unread.
- **Pythia-410M** may be run on WikiText-2 train or on a model's exploration part as a reference for the measures, never on a confirmation part.

## Material
- **Exploration starts on the WikiText-2 train split**, as D1's did. It is exploratory material, and every model in the plan may have seen it in training.
- **New text:** English Wikipedia articles created between January 2025 and September 2026 (`../data/newtext/`, `MANIFEST.md`), fetched for the main study. It postdates the training data of all three models. The main study splits it by article into frozen parts, recorded in `../FROZEN.md`. For each model:
  - **exploration** may add `part-explore-<model>.jsonl` (`explore-pythia1b`, `explore-qwen05b`) once the split exists;
  - **confirmation** uses only `part-conf-<model>.jsonl` (`conf-pythia1b`, `conf-qwen05b`), under a protocol frozen beforehand. No script, and no person working on this part, reads a confirmation part before that model's protocol is frozen.
  - The Pythia-410M and OLMo parts are never read here.
- Ablation means come from WikiText-2 train, as in D1.
- Constructed inputs may be used freely in exploration.
- **Tokeniser difference.** Qwen's tokeniser merges sentence punctuation with a following newline (".\n" is one token), so after punctuation the break and the punctuation are one prediction. In D1-X, decision points are therefore restricted to word ends whose last character is a letter, for both labels and in every model, so that the measures compare like with like. The line-start marker is the token that carries the newline: the newline token in Pythia, and in Qwen either the newline token or a punctuation token fused with it. "Newline position" means that token throughout.

## Measures and tools
As in D1:
- the newline log-odds at word-end decision points, the break/no-break separation and AUROC;
- the token coefficient (spline in characters remaining plus tokens on the line);
- the width items (previous line shortened by 10 or more characters) and the newline-residual patch by layer;
- attention to the line-start newline;
- head and MLP mean-ablation and resample ablation, at all positions or at newline tokens only;
- patching head outputs at the newline;
- freezing heads at the decision token;
- random-head controls of matched size.

## Budget
- Up to two weeks of exploration in total. Local machine only (6 GB NVIDIA GPU, 6 GB, GPU memory under about 5 GB), in the memory cage, one GPU job at a time under the shared lock `~/.cache/d1-gpu.lock`; the main study's Pythia-410M round 3 has priority. Each job under about 20 minutes. No Modal, and no downloads over 300 MB.
- One model at a time: Pythia-1B first, then Qwen2.5-0.5B. Report between models.

## Gate
A model goes on to the mechanism search only if its break AUROC at width 60 (letter-ending decision points, 40 paragraphs) is at least 0.90 on the WikiText-2 train split. Otherwise the behaviour is written up and the model stops there.

Once the model's new-text exploration part exists, the same AUROC is measured on it. A model below 0.90 there does not go to confirmation; its exploration is written up instead.

## What counts as a candidate finding
For each role, a small set of components (up to six heads, or named MLPs) that meets both conditions:
- it does the role's job clearly beyond matched random controls on exploration material;
- it gives a prediction for the confirmation part, stated with a threshold before the confirmation part is read.

A role with no such components is recorded as "not found", along with what was tried.

## Stop rules
- Stop a model when each role has a candidate or has been searched without one.
- Also stop after one week per model.
- Park for the author if a download over 300 MB or Modal use seems necessary.

## If nothing is found
Report the behaviour and the role scans plainly. In exp15, Pythia-1B's head-edit test was not specific, so a null there is a real possibility, and a null is a reportable result.

## Record keeping
- `NOTEBOOK.md` in this folder: dated entries, dead ends included.
- Scripts in `code/` and outputs in `outputs/` in this folder.
- `STATUS.md` in this folder: a short state to fold into the area status.
- Decisions taken while the author is away go in `../../DECISIONS.md`, as append-only entries tagged "D1-X".
- Later changes to this charter are dated amendments in `AMENDMENTS.md` in this folder, never silent edits.
