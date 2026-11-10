# D1 freeze record

| File | SHA-256 | Frozen |
|---|---|---|
| CHARTER.md | ca2e73138d3a71736f0a38bfa761e48bc71da02d82617138d3413da4dd50320c | 2026-11-05 19:33 UTC |
| PROTOCOL.md | 962dd65bd85b88cc2f197b2927eb3b47d1f2aca8b8f3c9711648f205ae550b58 | 2026-11-06 02:26 UTC |
| code/conf01_token.py | c5041737893134b22d0116ca48c68f11ee9bfa9bd37939b9076d8b16cef842ed | 2026-11-06 02:26 UTC |
| code/conf02_width.py | ced22552da549f2c66d489f00694b63cd30a04e1be24b1ac76194771027e4983 | 2026-11-06 02:26 UTC |
| code/common.py | af2ae5c83c14ad069eb6bb967e46ca48df53c8d4f554cc77e98ff3120b7f951d | 2026-11-06 02:26 UTC |
| data/wikitext2/validation-00000-of-00001.parquet | 204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c | 2026-11-06 02:26 UTC |

Frozen before any exploration script was written or run. Later changes to the charter are dated amendments in `AMENDMENTS.md`, never silent edits.

The confirmation protocol and its inputs were frozen on 6 Nov 2026 at the author's yes ("Yes, freeze and run"), before any script read the validation split for results (only a count of eligible paragraphs, 845, was taken). The test split stays unread.


## Second confirmation round (`PROTOCOL-2.md`, test split)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL-2.md | f394ed0e1b4bfa63338c348371b6e22ca1ed9c5a0246f05399c60e43239e1fe0 | 2026-11-06 05:24 UTC |
| code/c2_01_token.py | 33b45e9b312366253087304822ac8c31474896ad47b5877bbc094094665135ee | 2026-11-06 05:24 UTC |
| code/c2_02_width.py | 590a474e45efc85a70b119553d87a70a090cb3d1dc1668e4c4782c93a0385c3d | 2026-11-06 05:24 UTC |
| code/c2_03_newline.py | c6d351c7c06d402ac2c217c645cd20d33279bc35578bf6c8d0f90b28bc67c92e | 2026-11-06 05:24 UTC |
| code/c2_04_paths.py | b01d7413bf934b10f3fd2d47930fbf193660b30a2f83ab3fd66c1929c427e615 | 2026-11-06 05:24 UTC |
| code/common.py | af2ae5c83c14ad069eb6bb967e46ca48df53c8d4f554cc77e98ff3120b7f951d | 2026-11-06 05:24 UTC |
| code/report2_tables.py | 1aa1b59e42541a142ab37d82792df282e3a9d14320e459588547152c0227b203 | 2026-11-06 05:24 UTC |
| data/wikitext2/test-00000-of-00001.parquet | 5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91 | 2026-11-06 05:24 UTC |

Frozen at the author's choice of a second confirmation round, before any script read the test split. `code/common.py` is unchanged since the first freeze (same digest). `code/report2_tables.py` only formats saved outputs and is digested for the record. The test parquet's digest matches `data/wikitext2/SHA256SUMS`. Pipeline checks ran on the train split with `--smoke` before the freeze; their outputs are labelled `_train_smoke` and are not results.


## Third confirmation round (`PROTOCOL-3.md`, new text) and the new-text split

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL-3.md | fda58b11dfded7f6af41b76774fa88b072444fc0ddfa6472ff6036ff14f4a5c1 | 2026-11-06 10:41 UTC |
| code/c3_01_token.py | ab6b982e3405714d14e7c4c5e9a23cc6d8c29ecf8a9ac4c04ee7f4e0c11cb110 | 2026-11-06 10:41 UTC |
| code/c3_02_width.py | 4a64fea5d9fb55e551495086c586bdfd7292fc414d025ab3e0d65a3989c41eca | 2026-11-06 10:41 UTC |
| code/c3_03_newline.py | 038e79f16d0d923e5fa7ffb73a9a1ec5235926d1e7d61602b6be7d8de9d962c8 | 2026-11-06 10:41 UTC |
| code/c3_04_paths.py | 009f614de11c6e2b67e368099aabfcbe64fb389fb59e0d81176c874b5e9ab47f | 2026-11-06 10:41 UTC |
| code/newtext.py | ce9397f133131242a05edb89bad947fee7903a6d931365c226e78f648ace443e | 2026-11-06 10:41 UTC |
| code/common.py | af2ae5c83c14ad069eb6bb967e46ca48df53c8d4f554cc77e98ff3120b7f951d | 2026-11-06 10:41 UTC |
| code/fetch_newtext.py | 8a320a296e7ba0bf44634651fe35250b5647a8b398353f3f7cb610c851efe263 | 2026-11-06 10:41 UTC |
| code/report3_tables.py | e11a41034465bdcbddae99ea30e7833a71ca126773909c59c3409d392e771732 | 2026-11-06 10:41 UTC |
| data/newtext/articles.jsonl | d37ce9a3c394aeb82a1d3d08deceb1b87bcda7f6d76bd79028eb0fd68c231177 | 2026-11-06 10:41 UTC |
| data/newtext/fetch_summary.json | 4d696e31cafbe6c84530e9929667abcc365130256d4f44be8a80c97bf1e5f4fc | 2026-11-06 10:41 UTC |
| data/newtext/part-conf-pythia410m.jsonl | 2cd914eb3254d88f8040c40d95248d302b3b7718790c9f65d9045b419c19a801 | 2026-11-06 10:41 UTC |
| data/newtext/part-explore-pythia1b.jsonl | a659d9a7b6bf5445ae7f949156dc4243754f552c2017fcfb6cfbe422914b5440 | 2026-11-06 10:41 UTC |
| data/newtext/part-conf-pythia1b.jsonl | b9e9b47f72d183319099d5f8b7465126686c79a15bd297ad1b9fb8a60a6a6381 | 2026-11-06 10:41 UTC |
| data/newtext/part-explore-qwen05b.jsonl | 86d3b175f4a3e4700dac585feb48c8565ba0cd628eb27f3b48b7a11fcc1f7763 | 2026-11-06 10:41 UTC |
| data/newtext/part-conf-qwen05b.jsonl | fdb644fa36d4a56e3549bebb41e33b01b2a4cbe0b2af5153109445b752e7113b | 2026-11-06 10:41 UTC |
| data/newtext/part-explore-olmo1b.jsonl | 9ff1ffaa3f11fb04be4590fce27aa9824aa4edc05730bc7060fd20f778aa8b42 | 2026-11-06 10:41 UTC |
| data/newtext/part-conf-olmo1b.jsonl | 81f4e519b525b24518f4728abbd770ec39fe674ef538bf6bf8a7b5d02f3a574e | 2026-11-06 10:41 UTC |
| data/newtext/MANIFEST.md | 69b65bc08334b68d17884b7b9eb911820a5944625146ec3f32a00047a7457eaa | 2026-11-06 10:41 UTC |
| data/newtext/SHA256SUMS | d2093379254fef67ad3453c728e4d41b5ccb86c7f349d08b38bdaf6516d79850 | 2026-11-06 10:41 UTC |

Frozen on 6 Nov 2026, right after the fetch finished and before any script read the new text for results (only line counts per part were taken). The seven part files are the split for round 3 and for D1-X (`xmodel/CHARTER.md`); each confirmation part is read only under a protocol frozen beforehand for its model. `code/common.py` is unchanged since the first freeze. `code/report3_tables.py` only formats saved outputs. Pipeline checks of the c3 scripts ran on the WikiText-2 train split with `--smoke` before the freeze; their outputs are labelled `_train_smoke` and are not results.


## Round-4 exploration (`AMENDMENTS.md`, Amendment 1)

| File | SHA-256 | Frozen |
|---|---|---|
| AMENDMENTS.md | 8fc50566590c8b6b2f15d7c4f97985ce884add1ec2c6adea94871f87ce7acf07 | 2026-11-06 12:58 UTC |
| code/fetch_newtext2.py | 9c3e718a6744a1c335ffc3a61f87a8c4b2220143ff39e461dbaeab19f2f72d8a | 2026-11-06 12:58 UTC |

Frozen on 6 Nov 2026 at the author's go-ahead for round 4, before any round-4 exploration ran. The fetch of the second text batch had started a few minutes earlier under this script; its parts are digested when it finishes. Later amendments are appended to `AMENDMENTS.md` and digested again here.


## Fourth confirmation round (`PROTOCOL-4.md`, second new-text batch)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL-4.md | da7e46cf1e66af73c1d899aa4e7a07d0f6f88ef3b393e1aac70f13f4596f799f | 2026-11-06 14:23 UTC |
| code/p4_01_decision.py | ceb1bd21b2545daaa78f590abf0c8e2fabf2146e8ab9b7fedd882904c74d7888 | 2026-11-06 14:23 UTC |
| code/p4_02_sufficiency.py | 88a9cfa3a92b51e1e1fece96fc578f62cc737b768ddd758f11ac4221fecbbbc0 | 2026-11-06 14:23 UTC |
| code/p4_03_geometry.py | 8fd1f7d41f107a74d7e448bc6a9d4cbf8a06d8d1ad9fd6dabe17b7b49a8b20a1 | 2026-11-06 14:23 UTC |
| code/p4_04_roles.py | 7861de89c0fdbbba1d03354ad02201771d6a43b89e04abf5c97468ca288c87a4 | 2026-11-06 14:23 UTC |
| code/newtext2.py | c6cc67f6b5b7dcee84961338ee21f04045eeea1b3c8faefbc95274ebcf032bd0 | 2026-11-06 14:23 UTC |
| code/common.py | af2ae5c83c14ad069eb6bb967e46ca48df53c8d4f554cc77e98ff3120b7f951d | 2026-11-06 14:23 UTC |
| code/report4_tables.py | ecef3b48811a0e4098dc16c7fc17a889dae4026112f616c36793253a3de5d925 | 2026-11-06 14:23 UTC |
| data/newtext2/articles.jsonl | 81e7a8e5e4fac6c42caf7038b2679bdf0f1e20e319f19d7aae9c4da359ed5819 | 2026-11-06 14:23 UTC |
| data/newtext2/extracts.jsonl | 81e7a8e5e4fac6c42caf7038b2679bdf0f1e20e319f19d7aae9c4da359ed5819 | 2026-11-06 14:23 UTC |
| data/newtext2/pages.json | a2f88b967562e4fef8134e041e4a8d5690c57c28e29d3591243ccdde192573e3 | 2026-11-06 14:23 UTC |
| data/newtext2/fetch_summary.json | 247793e5511c0638e4b0ec81264dcc6860fb90053f4af97fb0206457be3fb077 | 2026-11-06 14:23 UTC |
| data/newtext2/part-conf4-pythia410m.jsonl | bdc193ed2bfea51bf65485316f1ccfd9ba7d64e9723e7ff2f07159f13a2cfbe0 | 2026-11-06 14:23 UTC |
| data/newtext2/part-reserve-pythia410m.jsonl | 0293bb6902a7d83f2b32ff2c142bf1e796a7b340143527f9fe232ce435e759e2 | 2026-11-06 14:23 UTC |
| data/newtext2/MANIFEST.md | b40abc691a5a5fac1694a6920b6aadfe0b378d7443bbfda064d79083e41a537c | 2026-11-06 14:23 UTC |
| data/newtext2/SHA256SUMS | fa5549556570dcdb62d2e74f168fee80875d91137eb38c89602cb59bfeddaf06 | 2026-11-06 14:23 UTC |

Frozen on 6 Nov 2026, as soon as the second text batch finished fetching and before any script read it for results (only line counts per part were taken). The `reserve-pythia410m` part stays unread. `code/common.py` is unchanged since the first freeze. Pipeline checks ran on the WikiText-2 train split (`--smoke` and full train runs of each script); the train runs give the exploratory reference values in `PROTOCOL-4.md` and are not results.


## Round-5 exploration (`AMENDMENTS.md`, Amendment 2)

| File | SHA-256 | Frozen |
|---|---|---|
| AMENDMENTS.md | e3bcbc106319cb1b925922bf6296bc8e96b7f635c6b8df07d9f03aa64093d504 | 2026-11-06 16:33 UTC |

Frozen on 6 Nov 2026 at the author's request for a robustness round, before any round-5 exploration ran. The new text parts are digested when they are built.

## Round-5 text: Python code (`data/code/`)

| File | SHA-256 | Frozen |
|---|---|---|
| code/build_code.py | b967af02dc1724360bcff1b4e32092b853ffb456d4a2c62478225068831d651f | 2026-11-06 16:40 UTC |
| data/code/part-explore-code.jsonl | b6b26ba76833e9fde5e47d5b0bae0ddc68ef7f57a51fcfdfc5ec6d15b0e1b453 | 2026-11-06 16:40 UTC |
| data/code/part-conf5-code.jsonl | d6804a0b6a6c8fab541721678419e897b2a9fedf947f7be615588e2e3eb13a33 | 2026-11-06 16:40 UTC |
| data/code/build_summary.json | 41c00d25a274be1fd69ebd736ae513360d881b3905a1d34257a8d211d4ead235 | 2026-11-06 16:40 UTC |
| data/code/MANIFEST.md | 224a4b5d2bee07dd808347c6837ccdc035dca3e1167a2b54e0d4b38c54873222 | 2026-11-06 16:40 UTC |
| data/code/SHA256SUMS | e9e36dcc560ebf904f39ceea476c5093bebe43315893313bf112defaac7f317e | 2026-11-06 16:40 UTC |

Frozen on 6 Nov 2026 as soon as the parts were built, before any script read them. Only paragraph counts were taken. `conf5-code` stays unread until `PROTOCOL-5.md` is frozen.

## Round-5 text: German Wikipedia (`data/de/`)

| File | SHA-256 | Frozen |
|---|---|---|
| code/fetch_de.py | af2eccd881156bfe5598a14648817a94c216143a016640748ba4698b6cca6347 | 2026-11-06 17:03 UTC |
| data/de/part-explore-de.jsonl | c191d53df4c4a3a8e90a5de8ac20c20164c4e579bb46cb9803e11cb74387e403 | 2026-11-06 17:03 UTC |
| data/de/part-conf5-de.jsonl | 6e0f3ce242fdb1f7d2af6fc7891d2db65be0f7b833cd4dfb648c0ff2fd570eb5 | 2026-11-06 17:03 UTC |
| data/de/extracts.jsonl | e13bc05555da8029446f4e93ed0c6c74386670c93c0b6d89392d241e97912acc | 2026-11-06 17:03 UTC |
| data/de/pages.json | 47906562055604a70c37811a8177762987bd1bf65769137f45f685e7e003937c | 2026-11-06 17:03 UTC |
| data/de/fetch_summary.json | e71e5a19b9cc65180e3c140371bb75b6e5bfeb4fb4a02cbdd4a6fb586d46735b | 2026-11-06 17:03 UTC |
| data/de/MANIFEST.md | 6b89a006dff1a88e532aeee5839cc7ac3d4d04cda449865db1d372c449c4e543 | 2026-11-06 17:03 UTC |
| data/de/SHA256SUMS | 7de55a42ca79556097271c97cabe1de4a1bb27ecc7948b0526e2391bde5520f0 | 2026-11-06 17:03 UTC |

Frozen on 6 Nov 2026 as soon as the fetch finished, before any script read the text. Only paragraph counts were taken. `conf5-de` stays unread until `PROTOCOL-5.md` is frozen.

## Fifth confirmation round (`PROTOCOL-5.md`)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL-5.md | 24950e53257e24c7d650930903779bc1e60d5ece3133a74467947f85b1ea7584 | 2026-11-06 22:22 UTC |
| code/p5_01_sufficiency.py | 1d6dfe6d64b88d2d3da93a3fba7f9a1ab89798daa51024279aa8d3b323d27da5 | 2026-11-06 22:22 UTC |
| code/p5_02_tokenbias.py | 90df862ef542724dc88a09262a49bf037ba99aac4392dd17d31a39f88c58ed3c | 2026-11-06 22:22 UTC |
| code/p5_03_damping.py | 4adbc89672102b1be1d9e8b46b7ad0f8b605bc22acd339f309322cf717c35e79 | 2026-11-06 22:22 UTC |
| code/p5_04_general.py | 1140e03a409c031e910c457de7f425334648258a237f58103e63201a40adff8e | 2026-11-06 22:22 UTC |
| code/newtext5.py | e9e8d631556e5a8baadb9b01b35ce71965603e84ffd2d228672a26c1a0dd4734 | 2026-11-06 22:22 UTC |
| code/report5_tables.py | 06138b49ea49a6ef8f1f1705f97057ec042a56fa6dcef2d43ffc2992ff09bf9b | 2026-11-06 22:22 UTC |
| code/common.py | af2ae5c83c14ad069eb6bb967e46ca48df53c8d4f554cc77e98ff3120b7f951d | 2026-11-06 22:22 UTC |
| xmodel/code/xm_common.py | b16c86dccd6a32d9a294c7b458a22fbf10bd5bcc7fdafba0c8c699360a8ea24d | 2026-11-06 22:22 UTC |

Frozen on 7 Nov 2026, after the round-5 exploration and before any script read the confirmation parts (`reserve-pythia410m`, `conf5-code`, `conf5-de`). `code/common.py` is unchanged since the first freeze, and `xmodel/code/xm_common.py` (imported by `p5_04_general.py`) is unchanged since D1-X froze it. Pipeline checks ran on train and the exploration parts (`--smoke` and full train runs); they give the explored values in `PROTOCOL-5.md` and are not results. `report5_tables.py` reads a test folder only when the environment variable P5_OUT is set, which is used for pipeline checks only.
