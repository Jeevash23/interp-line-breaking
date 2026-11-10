"""D1-X Amendment 3: merge the chunks of xm3_e01_scan.py (--chunk K/N) into one file, as an unchunked run
would have saved it. Usage: xm3_e01_merge.py <model> <part> <N>
Chunk 0 holds the behaviour and attention parts; chunks 1..N hold the single-head ablations (MLPs in chunk N).
The intact measures must agree across chunks (same paragraphs, same model); the script checks this."""
import sys, json
from pathlib import Path
O = Path(__file__).resolve().parent.parent / "outputs"
MODEL, PART, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
ch = [json.load(open(O / f"xm_e01_{MODEL}_{PART}_chunk{k}of{N}.json")) for k in range(N + 1)]
intact = [c["ablation"]["intact"] for c in ch]
spread = max(abs(i["separation"] - intact[0]["separation"]) for i in intact)
assert spread < 1e-3, f"intact separation differs across chunks by {spread}"
out = {k: v for k, v in ch[0].items() if k not in ("ablation", "top_separation_drop", "top_token_coef")}
heads, mlps = {}, {}
for c in ch: heads.update(c["ablation"]["heads"]); mlps.update(c["ablation"]["mlps"])
assert len(heads) == out["L"] * out["H"], (len(heads), out["L"] * out["H"])
out["ablation"] = dict(intact=intact[0], heads=heads, mlps=mlps)
out["top_separation_drop"] = sorted(((k, v["separation"]) for k, v in heads.items()), key=lambda x: x[1])[:10]
out["top_token_coef"] = sorted(((k, v["token_coef"]) for k, v in heads.items()), key=lambda x: -x[1])[:10]
out["merged_from"] = [f"xm_e01_{MODEL}_{PART}_chunk{k}of{N}.json" for k in range(N + 1)]; out["intact_spread"] = spread
json.dump(out, open(O / f"xm_e01_{MODEL}_{PART}.json", "w"), indent=1)
print("intact", intact[0], "spread", spread)
print("sep", out["top_separation_drop"]); print("tok", out["top_token_coef"])
print("mlps", sorted(((k, round(v["separation"], 3)) for k, v in mlps.items()), key=lambda x: x[1])[:6])
