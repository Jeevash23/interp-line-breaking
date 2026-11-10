# Changes from the original study folder

This release was exported from the folder in which the study ran. Files are copied byte for byte except those listed
below. There are three kinds of change:

- **adapted:** `paper/v5-2026-11-10/check_paper.py`, whose two architecture checks read the model configuration files
  from `models/configs/` (shipped copies) instead of a local model cache.
- **redacted:** records and scripts in which personal or machine-specific strings were replaced: absolute paths by
  placeholders (`<home>`, `<hf-cache>`, `<environment>`, `<portfolio-root>`, `<workspace>`), the path of a temporary
  folder by `/tmp/`, local clock times by the same instant in UTC, verbatim quotes of the author's messages by a
  short note that the original record quotes them, and wording about how the work was coordinated between working
  sessions by neutral wording ("the main study"). The machine is described as "the local machine" and its GPU as a
  6 GB NVIDIA GPU. The counts show how many strings of each kind were replaced. Nothing else changed: no number,
  threshold, prediction or verdict.
- **not shipped:** the exploration notebooks (`NOTEBOOK.md`, `xmodel/NOTEBOOK.md`: dated working logs of what was
  tried; the exploration scripts and their outputs are included), a status note, run logs, smoke tests (pipeline
  checks run before each freeze, which the records say are not results), the superseded float16 runs of exploration
  items exp01-exp03, and the paper's build products, own Makefile, working notes, reference-check records, arXiv tarball and its builder. None of these is named in a
  freeze record.

The third-party text the study read (WikiText-2, Wikipedia extracts, Python standard-library code) is shipped exactly
as frozen. Text files larger than 2 MB are shipped compressed (`.xz`) and the saved outputs as archives under
`data/saved-outputs/`; `tools/unpack.py` restores them byte for byte and checks each SHA-256 against
`data/PACKED-FILES.json`. Compression is not a change and is not listed below.

Several frozen records (protocols, amendments, the freeze records themselves) are among the redacted files. Each
keeps the frozen digest of its original; `make check-freezes` (`tools/check_frozen.py`) accepts a redacted copy only
when this file records that exact change.

## Files that differ from the project originals

| File | Change | Original SHA-256 | Frozen in |
|---|---|---|---|
| `AMENDMENTS.md` | redacted (approval wording x6, machine details x2, quoted message x2) | `e3bcbc106319cb1b925922bf6296bc8e96b7f635c6b8df07d9f03aa64093d504` | `FROZEN.md` |
| `CHARTER.md` | redacted (approval wording x5, machine details x3, process wording x2) | `ca2e73138d3a71736f0a38bfa761e48bc71da02d82617138d3413da4dd50320c` | `FROZEN.md` |
| `FROZEN.md` | redacted (approval wording x4, process wording x3, time zone x76) | `e59929f38aff3e3daec1ea4dfa810f16a4063c9c9c92d718a0b1f4a82d00c2bc` | - |
| `POSTHOC.md` | redacted (approval wording x1, process wording x1) | `d3c481982d16e464e9d9db6ec202452ef14e64cd9c7fc23cdef4d5dab90a143a` | - |
| `PROTOCOL-2.md` | redacted (approval wording x2, machine details x2, process wording x1) | `f394ed0e1b4bfa63338c348371b6e22ca1ed9c5a0246f05399c60e43239e1fe0` | `FROZEN.md` |
| `PROTOCOL-3.md` | redacted (approval wording x1, machine details x2, process wording x1) | `fda58b11dfded7f6af41b76774fa88b072444fc0ddfa6472ff6036ff14f4a5c1` | `FROZEN.md` |
| `PROTOCOL-4.md` | redacted (machine details x1) | `da7e46cf1e66af73c1d899aa4e7a07d0f6f88ef3b393e1aac70f13f4596f799f` | `FROZEN.md` |
| `PROTOCOL-5.md` | redacted (time zone x3) | `24950e53257e24c7d650930903779bc1e60d5ece3133a74467947f85b1ea7584` | `FROZEN.md` |
| `PROTOCOL.md` | redacted (approval wording x1, machine details x2) | `962dd65bd85b88cc2f197b2927eb3b47d1f2aca8b8f3c9711648f205ae550b58` | `FROZEN.md` |
| `REPORT-2.md` | redacted (time zone x1) | `3a60f71e0b908e58fcd90a3abbcfc3c3e2400f062e388bd5d635c457146c088d` | - |
| `REPORT-3.md` | redacted (process wording x1, time zone x2) | `c1d494a64c7ff49a6f7578dbd3c0079417696599154a0455ec5963d309d3a898` | - |
| `REPORT-4.md` | redacted (approval wording x1, time zone x2) | `723bb28bf4b7b06dfa8bea8bb4c4bbd6bc3a634ec0abf0f6fa10777ea4576aa1` | - |
| `REPORT-5.md` | redacted (time zone x3) | `9f536e1839733ae450d818924bf970bcbe76f9b5d01c4f942fb745843f0a66c8` | - |
| `REPORT.md` | redacted (approval wording x1, time zone x2) | `5705b3694b31bb8febddc67502dd3a0e7c4a67bfbc5fd3a30350442ce7a85a9b` | - |
| `code/chain5.sh` | redacted (absolute path x1, environment path x1) | `5faeef7e6d6ce4a7c8f4268078acd12094d85180f2bd8d7ce345a9f92f3fc176` | - |
| `code/chain5b.sh` | redacted (absolute path x1, environment path x1, process wording x1, time zone x1) | `c248f774d33264e79c8ab49cd07cfb5423aaaadc28afdeb8ffb4fe17a641d5d2` | - |
| `code/p5_04_general.py` | redacted (time zone x2) | `1140e03a409c031e910c457de7f425334648258a237f58103e63201a40adff8e` | `FROZEN.md` |
| `code/queue5.sh` | redacted (absolute path x1, environment path x1, time zone x1) | `3059ad6109a99dcda0f188c67a2318f995d37b4d8672b41ef7fee47d3aab059d` | - |
| `code/run.sh` | redacted (absolute path x1, environment path x1, process wording x1) | `78eb1792c3a9ca942e4b18c5a296c7d64764a821b0619aabc377ec3ee0e7d23f` | - |
| `data/code/MANIFEST.md` | redacted (machine details x1) | `224a4b5d2bee07dd808347c6837ccdc035dca3e1167a2b54e0d4b38c54873222` | `FROZEN.md` |
| `paper/v5-2026-11-10/check_paper.py` | adapted: the model configuration files are read from the copies shipped in models/configs/ | `6d17ca0ea199f006c4a8aea1c89c530bcfe35ef6a68b707d056661903562a606` | - |
| `xmodel/AMENDMENTS.md` | redacted (approval wording x5, machine details x2, process wording x4, quoted message x2, time zone x6) | `5438b262545ff27bb844cf7a8531488ba9d57ce6696db648e3146851b3215fcc` | `xmodel/FROZEN.md` |
| `xmodel/CHARTER.md` | redacted (approval wording x4, machine details x2, process wording x8) | `0f3cec7e7e5865f298b37d121052fa6350c1d7f7289f9251fe6722a7019b7238` | `xmodel/FROZEN.md` |
| `xmodel/FROZEN.md` | redacted (process wording x1, time zone x54) | `6cf7b338ad5edfa303c2c06af3f7b7036c712c675c4c41234f0d6c485d84e3b8` | - |
| `xmodel/PRECISION-smollm2-17b-posthoc.md` | redacted (machine details x1, time zone x4) | `9bb360b211978b04b1093409dcdc8ac2a99c23024c922d8709808622a74fd591` | - |
| `xmodel/PROTOCOL-X1.md` | redacted (approval wording x1, machine details x1, process wording x1) | `6addc736b3af483e4d3136544ccd2ff040c35662711dcd07bd5b74dbb5fcdd64` | `xmodel/FROZEN.md` |
| `xmodel/PROTOCOL-X2.md` | redacted (approval wording x2, process wording x1) | `c694caea258e9f97e47ad0e8bb582fdb48e5052d010230f4ec0cad7ebcb0a5e6` | `xmodel/FROZEN.md` |
| `xmodel/PROTOCOL-X3.md` | redacted (machine details x1) | `c272554dbf56ddc130e3d689ac7f50e5e5acc9b9e591c55cbb8a47a4ce76cfd1` | `xmodel/FROZEN.md` |
| `xmodel/PROTOCOL-X3b-smollm2.md` | redacted (machine details x2, time zone x1) | `d80b68bb49f44e59811846044a199c1c22f83c80c3cdd9855fc80357538e6723` | `xmodel/FROZEN.md` |
| `xmodel/PROTOCOL-X3b.md` | redacted (approval wording x1, machine details x1, process wording x1, time zone x1) | `099b0837d53a7bd84cc1a689b93e58ace170310ba0250f8d79fe8ae9e2e5e84f` | `xmodel/FROZEN.md` |
| `xmodel/REPORT-CROSSMODEL.md` | redacted (machine details x1) | `2969add61cebd1c5b44c94f0a5679b3e88280aaa350721ff5e92e51643169076` | - |
| `xmodel/REPORT-gemma4-e2b.md` | redacted (machine details x1, time zone x2) | `9d991b6e4c8c93ab9df8100d40513557991975f9ff6b411b4327ec2f9ef69bc3` | - |
| `xmodel/REPORT-olmo1b.md` | redacted (approval wording x1, time zone x2) | `b8c8ee43561eeb0c3a000fd3229c1bf31e7f8788a5b45307ffd0d4f9db12aa5b` | - |
| `xmodel/REPORT-pythia-1b.md` | redacted (time zone x3) | `cf21052c3fba31bc92d8ecead06f2a8678778b7153593ba96c1078f4836c722b` | - |
| `xmodel/REPORT-qwen05b.md` | redacted (time zone x3) | `f1a4d058a5e2d0c37e3f764fc63246fe6891cd77e0598b8d33b81a4ad2290411` | - |
| `xmodel/REPORT-qwen3-06b.md` | redacted (time zone x2) | `59f5413f16b99c1d2203d6d96b994da0d1c6ec96642750ac64ed4768e6a2a226` | - |
| `xmodel/REPORT-qwen3-17b.md` | redacted (time zone x2) | `ce825b5c96fb2e9b25416103833f50a838e67b00a8932190bc267719847db333` | - |
| `xmodel/REPORT-smollm2-17b.md` | redacted (machine details x1, time zone x2) | `107adfb06aa71aca5ee50723aca5ad3fdbc026494eed80fa9c54dd8803f3906e` | - |
| `xmodel/code/download_x3.sh` | redacted (environment path x1, process wording x1, time zone x1) | `86afa3606cc15bbad4466c133a2e77e5a19cce61e123c779f537ef4ecca831b4` | - |
| `xmodel/code/fetch_newtext3.py` | redacted (process wording x1) | `17fcd11afe834338531906c6e2e32875a4d56bee824a5bf9e02b8d2ec4d4ddd1` | `xmodel/FROZEN.md` |
| `xmodel/code/run.sh` | redacted (absolute path x3, environment path x1, process wording x1) | `19f9e53e3d2f12fe6b3e116709e6bd6e71974bf3d3466e3cb25d92b5bfc2f00b` | `xmodel/FROZEN.md` |
| `xmodel/code/xm3_e04_roles.py` | redacted (time zone x1) | `355680c1fac3e0aa721a8013ed96ec5175edeb83ab94efdf0f6c805e25bd5dac` | - |
| `xmodel/code/xm3_report_tables.py` | redacted (scratch path x1) | `e81dce3e92d1144382e8257ba2d93a4c506f5ec9d248a903c132ea0650c558f8` | `xmodel/FROZEN.md` |
| `xmodel/code/xm_e04_roles.py` | redacted (time zone x1) | `112e41b9325626a5a4dd7e9719a25616f26ba200fb16d562aeb7d50f599b5cb7` | - |
| `xmodel/code/xm_e06_overshoot.py` | redacted (process wording x1) | `1388b09dd77cfe5cdef6605fd1c7c48d58870e2f79deb2a40d4ad388776ff49b` | - |
| `xmodel/code/xm_report_tables.py` | redacted (scratch path x1) | `d370990165b67e18fabb460602a2ca30031d9f9f84693132429f5825e7f40f81` | `xmodel/FROZEN.md` |
| `xmodel/data/newtext3/MANIFEST.md` | redacted (time zone x1) | `ec8c492550111b502d1667a54cd3a8ea59a2dd5f5c9dd38f8f9fb06ed53113ec` | `xmodel/FROZEN.md` |
