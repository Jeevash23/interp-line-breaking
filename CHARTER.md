# D1 charter: how does Pythia-410M decide where to break a line?

Area 14 (mechanism discovery), study D1. Written 6 November 2026 and frozen before any exploration ran (digest in `FROZEN.md`). Picked by the author on 6 Nov from `../GAPS.md`, Target 1. On the same evening the author asked for the exploration to start and run on the local machine overnight.

Everything done under this charter is **exploratory**. A candidate mechanism found here is not a result. It becomes one only after a separate, frozen `PROTOCOL.md` tests its predictions on held-out material.

## Question

When text is wrapped at a fixed width, how does Pythia-410M decide that the next token should be a newline rather than the next word? Specifically, how does it combine:

- the characters already used on the current line;
- the line width, which it has to infer from earlier lines;
- the length of the word that would come next?

And what is missing in GPT-2 Small, which does this task poorly?

Any answer counts. Candidate accounts at the outset, none favoured:

- a character count compared with an inferred width (the Claude 3.5 Haiku account in arXiv 2601.04480);
- a token count rather than a character count;
- copying the previous line's length or its token count;
- cues from punctuation or word identity;
- something not on this list.

## Models

- **Pythia-410M** (EleutherAI, final checkpoint, on disk): the main model.
- **Pythia-1B** (final checkpoint, on disk): a stronger contrast within the same family and tokeniser.
- **GPT-2 Small** (on disk): the weak contrast.
- Not used unless the author approves a download: Pythia-160M (about 330 MB) and any intermediate checkpoints.

## Material

- **WikiText-2 raw** (Salesforce/wikitext, `wikitext-2-raw-v1`, 7.75 MB, public).
- **The train split is the only material used in exploration.** The validation and test splits are downloaded but not opened by any exploration script; they are kept for a later confirmation protocol. Constructed inputs (synthetic lines built to test a candidate, such as tokens with unusual character counts) may be made freely and are exploratory material.
- Text is re-wrapped greedily at chosen widths (range 20 to 160 characters), with a newline between lines.

## Measures

At each word end (a position where the next true token is either a newline or a word that starts with a space), the model's log-probability of the newline token. From that:

- **break accuracy**: how often the newline is the top token where a break is due;
- **false-break rate**: how often it is the top token where no break is due;
- **separation**: the area under the ROC curve of the newline log-probability between the two kinds of position;
- the same measures split by distance to the boundary.

## Tools

- Behaviour sweeps over width, position in the paragraph, word length, characters per token and punctuation.
- Linear probes (ridge regression, held-out folds within the train split) for characters since the last newline, characters remaining, the width and the next word's length, by layer.
- The logit lens.
- Attention analysis: heads that attend to the previous newline or to the end of the previous line.
- Activation patching and ablation of heads, MLPs and low-rank subspaces between prompts that differ in a controlled way, with random-head and random-subspace controls.
- Logogram may be used for looking around.

## Budget

- Up to four weeks of exploration. Local machine only (6 GB NVIDIA GPU, 6 GB). No Modal.
- Each script runs inside the memory cage from the global rules. GPU use stays under about 5 GB, because the GPU crashed twice under heavy load on 3 Nov.
- One download: WikiText-2 (7.75 MB).

## What counts as a candidate finding

A candidate mechanism is a small set of named components (heads, MLPs or neurons) or a low-dimensional variable that meets both conditions:

- **It moves the decision as predicted.** Intervening on it moves the newline decision in the predicted direction and roughly the predicted size, on train-split material, clearly beyond matched random controls. For example, patching in a count that says five more characters have been used should move the break earlier by about the length of a short word.
- **It makes a prediction nobody would make without it.** For example, a constructed input that should make the model break too early or too late, or an edit whose effect is specific.

Partial findings are recorded as such: for example, a variable that is decodable but has no causal effect, or a head set that matters but whose role is unclear.

## Stop rules

- Stop at four weeks.
- Stop early, and write up what was found, if either of these holds:
  - the behaviour is too weak to study (Pythia-410M's break accuracy below 40% at every width tried);
  - after two weeks of work, no intervention moves the break decision by more than its random controls.
- Stop and park for the author if a download over about 300 MB or Modal use seems necessary.

## If nothing is found

Write up the behavioural map: which cues each model uses, how accuracy depends on width and boundary distance, and why GPT-2 does worse. Add the probe-to-causal gap: whether the decodable character count is used at all.

## Record keeping

- `NOTEBOOK.md`: dated entries of what was tried and what came out, dead ends included.
- Scripts in `code/`, outputs in `outputs/` with the script name and date.
- Routine decisions within this charter are taken and logged in `NOTEBOOK.md` with their reason. Anything the shared rules reserve for the author is parked there.
