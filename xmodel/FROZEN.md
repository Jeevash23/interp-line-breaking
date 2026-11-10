# D1-X freeze record

| File | SHA-256 | Frozen |
|---|---|---|
| CHARTER.md | 0f3cec7e7e5865f298b37d121052fa6350c1d7f7289f9251fe6722a7019b7238 | 2026-11-06 06:33 UTC |

The charter was frozen before any exploration ran under it. The main study's earlier `--smoke` pipeline checks (WikiText-2 train, 4–6 items) are labelled `_train_smoke` and are not results. Later changes are dated amendments in `AMENDMENTS.md`, never silent edits. Confirmation protocols are added below when frozen.

| File | SHA-256 | Frozen |
|---|---|---|
| AMENDMENTS.md (Amendment 1, OLMo-2-1B added) | 13276cd0fe0036daf94fc92ca94e199464375be082c64a63ec0f428bb947097e | 2026-11-06 07:08 UTC |

## Confirmation protocol PROTOCOL-X1 (frozen before any script read a confirmation part)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL-X1.md | 6addc736b3af483e4d3136544ccd2ff040c35662711dcd07bd5b74dbb5fcdd64 | 2026-11-06 11:38 UTC |
| specs/conf-pythia1b.json | 3bf539e6db482117fc93f92d9fb0a811b72e224cae2933ea20d9f3e424e09562 | 2026-11-06 11:38 UTC |
| specs/conf-qwen05b.json | 33c26797587841ec80c39e58b0a5d70a2b3a3c5370e57195802f1996fac6d254 | 2026-11-06 11:38 UTC |
| specs/conf-olmo1b.json | 850090b5a1869976dca549f1aff1068c0a18885d26e101284226dce20ae50213 | 2026-11-06 11:38 UTC |
| code/xm_conf.py | 668edd22a748c757709f296ada111eb7b68246d69d2ba5dc24f5781bb015cf36 | 2026-11-06 11:38 UTC |
| code/xm_common.py | b16c86dccd6a32d9a294c7b458a22fbf10bd5bcc7fdafba0c8c699360a8ea24d | 2026-11-06 11:38 UTC |
| code/xm_report_tables.py | d370990165b67e18fabb460602a2ca30031d9f9f84693132429f5825e7f40f81 | 2026-11-06 11:38 UTC |
| code/run.sh | 19f9e53e3d2f12fe6b3e116709e6bd6e71974bf3d3466e3cb25d92b5bfc2f00b | 2026-11-06 11:38 UTC |
| ../code/common.py | af2ae5c83c14ad069eb6bb967e46ca48df53c8d4f554cc77e98ff3120b7f951d | 2026-11-06 11:38 UTC |
| ../code/newtext.py | ce9397f133131242a05edb89bad947fee7903a6d931365c226e78f648ace443e | 2026-11-06 11:38 UTC |

Confirmation parts, as listed in `../data/newtext/SHA256SUMS` (digests copied from that listing, files not opened):

    81f4e519b525b24518f4728abbd770ec39fe674ef538bf6bf8a7b5d02f3a574e  part-conf-olmo1b.jsonl
    b9e9b47f72d183319099d5f8b7465126686c79a15bd297ad1b9fb8a60a6a6381  part-conf-pythia1b.jsonl
    fdb644fa36d4a56e3549bebb41e33b01b2a4cbe0b2af5153109445b752e7113b  part-conf-qwen05b.jsonl

Exploration and transfer checks used WikiText-2 train and the three `part-explore-*` files only. Model weights: Pythia-1B and Qwen2.5-0.5B from the Hugging Face cache as used throughout; OLMo-2-1B snapshot a1847dff35000b4271fa70afc5db10fd29fedbdf.

## Follow-up protocol PROTOCOL-X2 (Amendment 2; frozen before any OLMo script read the WikiText-2 validation split)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL-X2.md | c694caea258e9f97e47ad0e8bb582fdb48e5052d010230f4ec0cad7ebcb0a5e6 | 2026-11-06 13:19 UTC |
| AMENDMENTS.md | dd30690b29c86cc9afa74583e0ce7d6ab50cd771ebd9610281438a2dcb96e199 | 2026-11-06 13:19 UTC |
| code/xm_c2_overshoot.py | 28742417647c63081bd71bc37bfedcde1f5d8252cf673fd9a8d6f3de6e55f767 | 2026-11-06 13:19 UTC |
| code/xm_common.py | b16c86dccd6a32d9a294c7b458a22fbf10bd5bcc7fdafba0c8c699360a8ea24d | 2026-11-06 13:19 UTC |
| ../code/common.py | af2ae5c83c14ad069eb6bb967e46ca48df53c8d4f554cc77e98ff3120b7f951d | 2026-11-06 13:19 UTC |

WikiText-2 validation, as listed in `../data/wikitext2/SHA256SUMS`:

    204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c  validation-00000-of-00001.parquet

## Amendment 3 data fetch (frozen 2026-11-06 17:13 UTC, before the fetch ran)

| File | SHA-256 | Frozen |
|---|---|---|
| code/fetch_newtext3.py | 17fcd11afe834338531906c6e2e32875a4d56bee824a5bf9e02b8d2ec4d4ddd1 | 2026-11-06 17:13 UTC |

## Amendment 3 (frozen 2026-11-06 17:15 UTC, before any Amendment-3 model was run)

| File | SHA-256 | Frozen |
|---|---|---|
| AMENDMENTS.md (Amendments 1–3) | 5438b262545ff27bb844cf7a8531488ba9d57ce6696db648e3146851b3215fcc | 2026-11-06 17:15 UTC |

## Amendment 3 third English batch (frozen 2026-11-07 00:22 UTC, after the fetch and before any script read it)

`data/newtext3/SHA256SUMS`:

    f9c227e3f6fc1f0c3a3143a20c9a2b1adaf45d8f86ea1c671bcf66342eede55a  articles.jsonl
    f9c227e3f6fc1f0c3a3143a20c9a2b1adaf45d8f86ea1c671bcf66342eede55a  extracts.jsonl
    1725a2bd97a936b7ce9d8010488b5dba11f1fd290eca3c8a16d96d1d4722f1ac  pages.json
    626f48cbee3111f177f45dbea9d627cf646da3e0e78f5db3a6cf84c038720d3b  fetch_summary.json
    70bb941138aa7bdd148e190bc656d69cc41400712416f0f983edb8330997f9dd  part-conf-gemma4-e2b.jsonl
    1880335de81e64c9578704e9e35fe4ad907003edb8aa43ce5532e6de2b3c599a  part-conf-qwen3-06b.jsonl
    2fbb0a3d6b5eb03a98103403eb78b336e48fd26936b3a029d5dbeb0fdef85ed8  part-conf-qwen3-17b.jsonl
    f6b3c1295f1ffbdb34c08e6c68a314a99e801622dc6af56aa1ec40ab61413350  part-conf-smollm2-17b.jsonl
    dd3cf3c69e1fdf9f2599ee6056758483ae992b930ebd62b3b806a3133da2f49a  part-explore-gemma4-e2b.jsonl
    c8f10074178c5401bc331e364ff9429bfc997bc98abee78f7ceac822713dccb5  part-explore-qwen3-06b.jsonl
    55554ad3c43dacb71bd81991af4d1e42c6726d66014fc1de9027dbccfb28ba89  part-explore-qwen3-17b.jsonl
    8d6cc79b3526d7b4281b3caa5fb4586905e6b41de1839b431e6b2e41ddff7764  part-explore-smollm2-17b.jsonl

| File | SHA-256 |
|---|---|
| data/newtext3/MANIFEST.md | ec8c492550111b502d1667a54cd3a8ea59a2dd5f5c9dd38f8f9fb06ed53113ec |
| data/newtext3/SHA256SUMS | 6ae290d588f86ef14a93b7044a31ce45ba8378d48f7107adb3692e55b7af7c18 |

## Confirmation protocol PROTOCOL-X3 (Amendment 3; frozen 2026-11-07 03:30 UTC, before any script read part-conf-qwen3-17b or part-conf-qwen3-06b)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL-X3.md | c272554dbf56ddc130e3d689ac7f50e5e5acc9b9e591c55cbb8a47a4ce76cfd1 | 2026-11-07 03:30 UTC |
| specs/conf-qwen3-17b.json | 8ff63ece96007384d397967b9852ee8833dd2cb3df62f62fc9271e09597f3e21 | 2026-11-07 03:30 UTC |
| specs/conf-qwen3-06b.json | 9f87e29a6a401b436829e549209c044967c99c3169bf55ecfb8e1ec7b6b8a10a | 2026-11-07 03:30 UTC |
| code/xm3_conf.py | 8056a9e4957290cf00003aacd81309529a82bb30b97975af6442c4d39862762a | 2026-11-07 03:30 UTC |
| code/xm3_common.py | cb96c1b1eeca68c50fa3f4c4690fd93e98e4d72f3a97791ddb2c51b0d52dd309 | 2026-11-07 03:30 UTC |
| code/xm_common.py | b16c86dccd6a32d9a294c7b458a22fbf10bd5bcc7fdafba0c8c699360a8ea24d | 2026-11-07 03:30 UTC |
| ../code/common.py | af2ae5c83c14ad069eb6bb967e46ca48df53c8d4f554cc77e98ff3120b7f951d | 2026-11-07 03:30 UTC |
| code/xm3_report_tables.py | e81dce3e92d1144382e8257ba2d93a4c506f5ec9d248a903c132ea0650c558f8 | 2026-11-07 03:30 UTC |
| code/run.sh | 19f9e53e3d2f12fe6b3e116709e6bd6e71974bf3d3466e3cb25d92b5bfc2f00b | 2026-11-07 03:30 UTC |
| AMENDMENTS.md | 5438b262545ff27bb844cf7a8531488ba9d57ce6696db648e3146851b3215fcc | 2026-11-07 03:30 UTC |

Exploration and transfer checks for these two models used WikiText-2 train and part-explore-qwen3-17b / part-explore-qwen3-06b only. Weights: Qwen/Qwen3-1.7B-Base snapshot ea980cb0…, Qwen/Qwen3-0.6B-Base snapshot da87bfb6….

## Confirmation protocol PROTOCOL-X3b, Gemma-4-E2B (Amendment 3; frozen 2026-11-07 07:43 UTC, before any script read part-conf-gemma4-e2b)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL-X3b.md | 099b0837d53a7bd84cc1a689b93e58ace170310ba0250f8d79fe8ae9e2e5e84f | 2026-11-07 07:43 UTC |
| specs/conf-gemma4-e2b.json | 534209a62a057934d9561a0345618149a7e8a161f8f2a1f8bffb9cdffeb2edac | 2026-11-07 07:43 UTC |
| code/xm3_conf.py | 8056a9e4957290cf00003aacd81309529a82bb30b97975af6442c4d39862762a | 2026-11-07 07:43 UTC |
| code/xm3_common.py | cb96c1b1eeca68c50fa3f4c4690fd93e98e4d72f3a97791ddb2c51b0d52dd309 | 2026-11-07 07:43 UTC |
| code/xm_common.py | b16c86dccd6a32d9a294c7b458a22fbf10bd5bcc7fdafba0c8c699360a8ea24d | 2026-11-07 07:43 UTC |
| ../code/common.py | af2ae5c83c14ad069eb6bb967e46ca48df53c8d4f554cc77e98ff3120b7f951d | 2026-11-07 07:43 UTC |
| code/xm3_report_tables.py | e81dce3e92d1144382e8257ba2d93a4c506f5ec9d248a903c132ea0650c558f8 | 2026-11-07 07:43 UTC |
| code/run.sh | 19f9e53e3d2f12fe6b3e116709e6bd6e71974bf3d3466e3cb25d92b5bfc2f00b | 2026-11-07 07:43 UTC |
| AMENDMENTS.md | 5438b262545ff27bb844cf7a8531488ba9d57ce6696db648e3146851b3215fcc | 2026-11-07 07:43 UTC |

The shared code is unchanged since PROTOCOL-X3 (same digests). Exploration and transfer checks for Gemma used WikiText-2 train and part-explore-gemma4-e2b only. Weights: google/gemma-4-E2B snapshot 433813a6…. SmolLM2-1.7B is not covered (see PROTOCOL-X3b.md); part-conf-smollm2-17b stays unread until PROTOCOL-X3b-smollm2.md is frozen.

## PROTOCOL-X3b addendum, SmolLM2-1.7B (Amendment 3; frozen 2026-11-07 09:59 UTC, before any script read part-conf-smollm2-17b)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL-X3b-smollm2.md | d80b68bb49f44e59811846044a199c1c22f83c80c3cdd9855fc80357538e6723 | 2026-11-07 09:59 UTC |
| specs/conf-smollm2-17b.json | 98c63c455a04beeeff63ccad155efbc7d7cc62948380d430e56d9ecd5647bb0a | 2026-11-07 09:59 UTC |
| PROTOCOL-X3b.md | 099b0837d53a7bd84cc1a689b93e58ace170310ba0250f8d79fe8ae9e2e5e84f | 2026-11-07 09:59 UTC |
| code/xm3_conf.py | 8056a9e4957290cf00003aacd81309529a82bb30b97975af6442c4d39862762a | 2026-11-07 09:59 UTC |
| code/xm3_common.py | cb96c1b1eeca68c50fa3f4c4690fd93e98e4d72f3a97791ddb2c51b0d52dd309 | 2026-11-07 09:59 UTC |
| code/xm_common.py | b16c86dccd6a32d9a294c7b458a22fbf10bd5bcc7fdafba0c8c699360a8ea24d | 2026-11-07 09:59 UTC |
| ../code/common.py | af2ae5c83c14ad069eb6bb967e46ca48df53c8d4f554cc77e98ff3120b7f951d | 2026-11-07 09:59 UTC |
| code/xm3_report_tables.py | e81dce3e92d1144382e8257ba2d93a4c506f5ec9d248a903c132ea0650c558f8 | 2026-11-07 09:59 UTC |
| code/run.sh | 19f9e53e3d2f12fe6b3e116709e6bd6e71974bf3d3466e3cb25d92b5bfc2f00b | 2026-11-07 09:59 UTC |
| AMENDMENTS.md | 5438b262545ff27bb844cf7a8531488ba9d57ce6696db648e3146851b3215fcc | 2026-11-07 09:59 UTC |

Shared code unchanged since PROTOCOL-X3 (same digests). Exploration and transfer checks for SmolLM2 used WikiText-2 train and part-explore-smollm2-17b only. Weights: HuggingFaceTB/SmolLM2-1.7B snapshot effd688a….
