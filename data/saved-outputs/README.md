# Saved outputs

The studies' results, packed as deterministic tar archives compressed with xz. Member paths are relative to the release root, so `make unpack` (or `python tools/unpack.py`) puts every file where the study code reads it and checks each SHA-256 against `data/PACKED-FILES.json`. Text records inside the archives had personal strings replaced, as listed in `RELEASE-CHANGES.md`.

| Archive | Files | Packed (MB) | Unpacked (MB) | Contents |
|---|---|---|---|---|
| `crossmodel.tar.xz` | 104 | 0.4 | 2.2 | The cross-model part (xmodel/: PROTOCOL-X1, X2, X3, X3b): exploration and confirmation runs of seven models |
| `exploration.tar.xz` | 43 | 10.1 | 14.5 | Exploration exp01-exp29 (CHARTER.md, AMENDMENTS.md), except exp03's activations |
| `round1-3.tar.xz` | 18 | 0.1 | 0.4 | Confirmation rounds 1-3 (PROTOCOL.md, -2, -3) and the post hoc checks of POSTHOC.md |
| `round4.tar.xz` | 15 | 23.2 | 38.4 | Round 4 (PROTOCOL-4.md), with the pipeline runs on the train split |
| `round5.tar.xz` | 25 | 4.9 | 9.6 | Round 5 (PROTOCOL-5.md), with the pipeline runs on train and the exploration parts |

Total packed size: 38.7 MB. Rendered figures are not included; `make figures` draws them from these outputs.

Two further archives are attached to the GitHub release instead of the repository, because of their size: `exploration-activations.tar.xz` (exp03's saved activations, read only for Figure 2) and `large-records.tar.xz` (the round-4 train-split run's mean activations and the post hoc reruns' per-item arrays). `data/ATTACHMENTS.json` lists their members with their SHA-256, and `make attachments` unpacks and checks them once they are in `attachments/`. Not included at all: run logs, the smoke tests run before each freeze (pipeline checks, not results) and the superseded float16 runs of exp01-exp03.
