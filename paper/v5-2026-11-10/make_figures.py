"""Draws the paper's data figures from saved outputs (CPU only). Figure 1 (the schematic) is drawn in TikZ in main.tex.
  fig-decision.pdf  : round 4 (held-out text): cosine of the COUNT and WIDTH residual changes by layer; share of each effect
                      lost when blocks of components are frozen at the decision token.
  fig-geometry.pdf  : (a) class means of the residual after layer 5 by characters used, in their top two principal
                      directions (WikiText-2 train, exploratory activations); (b) readers' and counters' attention to the
                      line-start newline by characters remaining (round 4, held-out text, width 60).
  fig-training.pdf  : role measures across training checkpoints (exploratory).
  fig-round5.pdf    : round 5 (held-out text): (a) the leftover token weight against mean tokens on the line, 11 text
                      and width conditions; (b) change in the likely next token's log-probability when the two damping
                      neurons are mean-ablated, by the intact probability, newline against other tokens; (c) fidelity of
                      the round-4 circuit plus the first k components of the fidelity path, mean and resample ablation.
Run inside the memory cage: python make_figures.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import values

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
O = values.O
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "font.family": "serif", "figure.dpi": 150})
BLUE, ORANGE, GREY, GREEN, RED = "#2b6ca3", "#d9822b", "#8c8c8c", "#3a8f5c", "#b03a3a"
J = lambda f: json.load(open(os.path.join(O, f)))
meta = dict(metadata={"Creator": None, "Producer": None, "CreationDate": None})

# ---------------- decision
S = J("p4_01_decision_conf4-pythia410m.json")["summary"]
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
L = list(range(5, 24)); cs = [S["cos_layer"][f"after L{l}"] for l in L]
ax[0].plot(L, cs, "o-", color=BLUE, ms=3, lw=1.2)
ax[0].set_xlabel("after layer"); ax[0].set_ylabel("cosine (COUNT, WIDTH)"); ax[0].set_ylim(-0.05, 1); ax[0].set_xticks([5, 8, 11, 14, 17, 20, 23])
ax[0].set_title("(a) the two inputs align", fontsize=8, loc="left")
blocks = [("layers6-11", "6–11"), ("layers12-17", "12–17"), ("layers18-23", "18–23")]
x = np.arange(len(blocks)); w = 0.36
ax[1].bar(x - w / 2, [S["mediation_sets"]["C"][b] for b, _ in blocks], w, color=BLUE, label="COUNT")
ax[1].bar(x + w / 2, [S["mediation_sets"]["Wd"][b] for b, _ in blocks], w, color=ORANGE, label="WIDTH")
ax[1].axhline(0, color="black", lw=0.6); ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, l in blocks])
ax[1].set_xlabel("layers frozen at the decision token"); ax[1].set_ylabel("share of effect lost"); ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("(b) no single site", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig-decision.pdf"), **meta); plt.close(fig)

# ---------------- geometry
z = np.load(os.path.join(O, "exp03_pythia-410m.npz")); meta3 = z["meta"]; cols = list(z["cols"])
Cm = {c: meta3[:, i] for i, c in enumerate(cols)}
R = np.load(os.path.join(O, "exp03_pythia-410m_resid_L0-12.npy"), mmap_mode="r")
sel = np.where(Cm["W"] == 60)[0]; used = Cm["used"][sel].astype(int)
X = np.asarray(R[6][sel], dtype=np.float32)  # index 6 = after layer 5
classes = [u for u in range(61) if (used == u).sum() >= 20]
M = np.stack([X[used == u].mean(0) for u in classes]); Mc = M - M.mean(0)
_, sv, Vt = np.linalg.svd(Mc, full_matrices=False); P = Mc @ Vt[:2].T
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.4))
sc = ax[0].scatter(P[:, 0], P[:, 1], c=classes, cmap="viridis", s=12)
ax[0].plot(P[:, 0], P[:, 1], color=GREY, lw=0.5, zorder=0)
cb = fig.colorbar(sc, ax=ax[0], fraction=0.05, pad=0.02); cb.set_label("characters used", fontsize=7)
ax[0].set_xlabel("first direction"); ax[0].set_ylabel("second direction"); ax[0].set_title("(a) class means after layer 5", fontsize=8, loc="left")
g = J("p4_03_geometry_conf4-pythia410m.json")["attention"]["60"]
for h, col, ls in (("L6H2", BLUE, "-"), ("L8H15", ORANGE, "-"), ("L17H1", GREEN, "-"), ("L5H6", GREY, "--"), ("L5H7", RED, "--")):
    a = g[h]; prof = np.array(a["profile"]); ax[1].plot(a["remaining"], prof / prof.max(), ls, color=col, lw=1.1, label=h)
ax[1].set_xlabel("characters remaining"); ax[1].set_ylabel("attention to line start\n(relative to its maximum)")
ax[1].invert_xaxis(); ax[1].legend(frameon=False, fontsize=6.5, ncol=2); ax[1].set_title("(b) readers ramp to the edge", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig-geometry.pdf"), **meta); plt.close(fig)

# ---------------- training
steps = [1000, 2000, 4000, 8000, 16000, 48000, 143000]; keys = ["1000", "2000", "4000", "8000", "16000", "48000", "final"]
D = [J(f"exp26_checkpoint_{k}.json") for k in keys]
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.1))
ax[0].plot(steps, [d["intact"]["L5H6_share"] for d in D], "o-", color=BLUE, ms=3, label="L5H6 line-start share")
ax[0].plot(steps, [d["intact"]["auroc"] for d in D], "s-", color=GREY, ms=3, label="AUROC")
ax[0].plot(steps, [d["counters_sep_kept"] for d in D], "^-", color=RED, ms=3, label="separation without COUNT4")
ax[0].set_title("(a) counter", fontsize=8, loc="left"); ax[0].legend(frameon=False, fontsize=6, loc="center", bbox_to_anchor=(0.62, 0.45)); ax[0].set_ylim(0, 1.05)
ax[1].plot(steps, [d["corrector_token_coef_change"] for d in D], "o-", color=ORANGE, ms=3, label="L5H7 correction")
ax[1].plot(steps, [d["intact"]["token_coef"] for d in D], "s-", color=GREY, ms=3, label="intact token weight")
ax[1].axhline(0, color="black", lw=0.5); ax[1].set_title("(b) corrector", fontsize=8, loc="left"); ax[1].legend(frameon=False, fontsize=6)
ax[2].plot(steps, [d["width_share_removed_readers3"] for d in D], "o-", color=GREEN, ms=3, label="readers")
ax[2].plot(steps, [d["width_share_removed_MLP14_16_18"] for d in D], "s-", color=BLUE, ms=3, label="MLPs 14, 16, 18")
ax[2].plot(steps, [d["width_share_removed_MLP21_23"] for d in D], "^-", color=RED, ms=3, label="MLPs 21, 23")
ax[2].axhline(0, color="black", lw=0.5); ax[2].set_title("(c) width effect carried", fontsize=8, loc="left"); ax[2].legend(frameon=False, fontsize=6)
for a in ax: a.set_xscale("log"); a.set_xlabel("training step")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig-training.pdf"), **meta); plt.close(fig)

# ---------------- round 5
V = values.load()
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.2))
for tx, col, mk, lab in (("en", BLUE, "o", "English"), ("code", ORANGE, "s", "code"), ("de", GREEN, "^", "German")):
    ws = (20, 30, 40, 60, 80, 100, 120) if tx == "en" else (40, 60)
    xs = [V[f"r5.t.{tx}.{w}.ntok"] for w in ws]; ys = [V[f"r5.t.{tx}.{w}.intact"] for w in ws]
    ax[0].errorbar(xs, [y[0] for y in ys], yerr=[[y[0] - y[1] for y in ys], [y[2] - y[0] for y in ys]], fmt=mk, color=col, ms=3.5, lw=0.8, capsize=1.5, label=lab)
    if tx == "en":
        for x_, y_, w in zip(xs, ys, ws): ax[0].annotate(str(w), (x_, y_[0]), xytext=(3, 3), textcoords="offset points", fontsize=5.5, color=GREY)
ax[0].axhline(0, color="black", lw=0.5); ax[0].set_xlabel("mean tokens on the line"); ax[0].set_ylabel("leftover token weight")
ax[0].legend(frameon=False, fontsize=6); ax[0].set_title("(a) token bias", fontsize=8, loc="left")
nz = np.load(os.path.join(O, "p5_03_damping_reserve-pythia410m_natural.npz")); R = nz["rec"]; RC = {c: i for i, c in enumerate(nz["cols"])}
ci_ = list(nz["conds"]).index("top2_abl"); bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]; mid = [(a + b) / 2 for a, b in zip(bins[:-1], bins[1:])]
for isnl, col, lab in ((1, RED, "newline"), (0, GREY, "other tokens")):
    sub = R[(R[:, RC["cond"]] == ci_) & (R[:, RC["is_newline"]] == isnl)]
    ys = [sub[(sub[:, RC["p_top"]] >= a) & (sub[:, RC["p_top"]] < b), RC["dlogp"]].mean() for a, b in zip(bins[:-1], bins[1:])]
    ax[1].plot(mid, ys, "o-", color=col, ms=3, lw=1.1, label=lab)
ax[1].axhline(0, color="black", lw=0.5); ax[1].set_xlabel("intact probability of the likely token"); ax[1].set_ylabel("change in log-probability")
ax[1].legend(frameon=False, fontsize=6); ax[1].set_title("(b) two neurons ablated", fontsize=8, loc="left")
ks = [0, 4, 8, 16, 24]; nmk = ["circuit", "plus4", "plus8", "plus16", "plus24"]
for mode, col, mk in (("mean", BLUE, "o"), ("resample", ORANGE, "s")):
    ax[2].plot(ks, [V[f"r5.s.{n_}.{mode}.fid"] for n_ in nmk], mk + "-", color=col, ms=3, lw=1.1, label=f"{mode} ablation")
    lo_, hi_ = V[f"r5.s.rand.{mode}.fid"]; ax[2].plot([24, 24], [lo_, hi_], color=col, lw=3, alpha=0.35)
ax[2].axhline(0, color="black", lw=0.5); ax[2].set_xlabel("components added"); ax[2].set_ylabel("fidelity"); ax[2].set_ylim(-1, 1)
ax[2].legend(frameon=False, fontsize=6, loc="lower right"); ax[2].set_title("(c) fuller circuit", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig-round5.pdf"), **meta); plt.close(fig)
print("figures written:", sorted(os.listdir(FIG)))
