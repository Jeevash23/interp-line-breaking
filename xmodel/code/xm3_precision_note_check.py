"""Check every number in PRECISION-smollm2-17b-posthoc.md against the saved outputs (post hoc, 7 Nov 2026).
Usage: xm3_precision_note_check.py"""
import json, sys, numpy as np
from pathlib import Path
X = Path(__file__).resolve().parent.parent; O = X / "outputs"
note = (X / "PRECISION-smollm2-17b-posthoc.md").read_text()
c = json.load(open(O / "xm3_e00_compare_smollm2-17b.json"))
A = json.load(open(O / "xm_e00_smollm2-17b_fp32cpu.json")); B = json.load(open(O / "xm_e00_smollm2-17b_bf16gpu.json"))
d = np.abs(np.array(A["logodds"]) - np.array(B["logodds"]))
q = {m: json.load(open(O / f"xm3_e00_compare_{m}.json"))["r"] for m in ("qwen3-06b", "qwen3-17b")}
bad = 0
def check(text, ok=True):
    global bad
    ok = ok and text in note; bad += not ok; print("ok " if ok else "BAD", text)
check(f"| Decision points | {c['n']} | {c['n']} |", A["labels"] == B["labels"] and A["dtype"] == "torch.float32")
check(f"| AUROC at width 60 | {c['auroc'][0]:.4f} | {c['auroc'][1]:.4f} |")
check(f"| Separation | {c['separation'][0]:.3f} | {c['separation'][1]:.3f} |")
check(f"r = {c['r']:.4f}"); check(f"mean absolute difference is {c['mean_abs_diff']:.3f}")
check(f"95th percentile {np.percentile(d, 95):.2f}"); check(f"largest {c['max_abs_diff']:.2f}")
check(f"(r = {q['qwen3-06b']:.4f} for both)", f"{q['qwen3-06b']:.4f}" == f"{q['qwen3-17b']:.4f}")
print("\nNOTE CHECK:", "all numbers match" if not bad else f"{bad} problems"); sys.exit(1 if bad else 0)
