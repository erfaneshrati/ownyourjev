import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

F = json.load(open("paper/facts.json"))
T = F["tasks"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED, GRID, INK, INK2, SURF = "#898781", "#e1e0d9", "#0b0b0b", "#52514e", "#ffffff"
NAMES = {"sms_spam": "SMS Spam", "paws": "PAWS", "prompt_injections": "Prompt Injections", "toxic_chat": "ToxicChat",
         "banking77": "Banking77", "clinc150": "CLINC150", "ag_news": "AG News", "financial_phrasebank": "Financial PhraseBank",
         "emotion": "Emotion", "stsb": "STS-B", "sst5": "SST-5", "toxigen": "ToxiGen"}
METRIC = {"f1": "F1", "accuracy": "acc.", "spearman": "Spearman", "argmax_accuracy": "acc."}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK})

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ---- Figure 1: results forest plot
order = ["sms_spam", "paws", "prompt_injections", "toxic_chat", "banking77", "clinc150", "ag_news", "financial_phrasebank", "emotion", "stsb", "sst5", "toxigen"]
fig, ax = plt.subplots(figsize=(6.4, 4.1))
style(ax)
ys = list(range(len(order)))[::-1]
for y, t in zip(ys, order):
    x = T[t]
    ax.plot(x["ci"], [y, y], color=BLUE, lw=1.6, solid_capstyle="round")
    ax.plot([x["ours"]], [y], "o", color=BLUE, ms=5.5, mec=SURF, mew=0.8, zorder=3)
    ax.plot([x["jev"]], [y], "D", color=MUTED, ms=5, mec=SURF, mew=0.8, zorder=3)
    lab = "win" if x["verdict"] == "win" else "tie"
    ax.text(1.005, y, lab, transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=INK if lab == "win" else INK2, fontweight="bold" if lab == "win" else "normal")
ax.set_yticks(ys)
ax.set_yticklabels([f"{NAMES[t]} ({METRIC[T[t]['metric']]})" for t in order])
ax.set_xlim(0.5, 1.0)
ax.grid(axis="x", color=GRID, lw=0.6)
for yb in (7.5, 2.5):
    ax.axhline(yb, color=GRID, lw=0.8)
ax.set_xlabel("score on the test split (higher is better)")
ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="-", label="ours: 150M encoder on CPU (95% CI)"),
                   Line2D([], [], marker="D", color=MUTED, ls="", label="Jev-1.13 on the same examples")],
          loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig("paper/figures/results.pdf", bbox_inches="tight"); fig.savefig("paper/figures/results.png", dpi=200, bbox_inches="tight")

# ---- Figure 2: cost and latency
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.3))
for ax in (a, b):
    style(ax)
yy = list(range(len(order)))[::-1]
for y, t in zip(yy, order):
    x = T[t]
    a.plot([x["ours_usd_per_m"], x["jev_usd_per_m"]], [y, y], color=GRID, lw=1.2, zorder=1)
    a.plot([x["jev_usd_per_m"]], [y], "D", color=MUTED, ms=4.5, zorder=2)
    a.plot([x["ours_usd_per_m"]], [y], "o", color=BLUE, ms=4.5, zorder=2)
a.set_xscale("log"); a.set_yticks(yy); a.set_yticklabels([NAMES[t] for t in order])
a.set_xlabel("USD per 1M decisions (log scale)")
a.set_title("cost per decision", loc="left", fontsize=9)
a.grid(axis="x", color=GRID, lw=0.6, which="both")
b.axvspan(70, 500, color=MUTED, alpha=0.18, lw=0)
b.text(72, yy[0] + 0.6, "Jev published\nend-to-end range\n70–500 ms", fontsize=7, color=INK2, va="top")
for y, t in zip(yy, order):
    x = T[t]
    b.plot([x["p50"], x["p95"]], [y, y], color=BLUE, lw=1.4)
    b.plot([x["p50"]], [y], "o", color=BLUE, ms=4)
    b.plot([x["p95"]], [y], "|", color=BLUE, ms=7, mew=1.4)
b.set_xscale("log"); b.set_xlim(8, 600); b.set_yticks(yy); b.set_yticklabels([])
b.set_xlabel("ms per request, CPU 4 threads (p50 to p95)")
b.set_title("latency", loc="left", fontsize=9)
b.grid(axis="x", color=GRID, lw=0.6, which="both")
fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", label="ours: CPU, 4 threads ($0.04 per vCPU-hour)"), Line2D([], [], marker="D", color=MUTED, ls="", label="Jev ($0.042 per 1M input tokens)")],
         loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("paper/figures/cost_latency.pdf", bbox_inches="tight"); fig.savefig("paper/figures/cost_latency.png", dpi=200, bbox_inches="tight")

# ---- Figure 3: labeling guidelines (dev agreement of the labeler with gold, before -> after), from the ledgers/NOTES
G = [("Emotion (acc.)", 0.49, 0.785), ("Banking77 (acc.)", 0.735, 0.84), ("SST-5 (acc.)", 0.53, 0.62),
     ("ToxicChat toxic (F1)", 0.774, 0.857), ("Prompt Inj. (acc.)", 0.895, 0.965), ("CLINC150 (acc.)", 0.90, 0.965), ("STS-B (Spearman)", 0.844, 0.919)]
fig, ax = plt.subplots(figsize=(5.2, 2.9))
style(ax)
yy = list(range(len(G)))[::-1]
for y, (n, a0, a1) in zip(yy, G):
    ax.annotate("", xy=(a1, y), xytext=(a0, y), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4, shrinkA=3, shrinkB=3))
    ax.plot([a0], [y], "o", color=MUTED, ms=4.5)
    ax.plot([a1], [y], "o", color=ORANGE, ms=5)
    ax.text(a1 + 0.01, y, f"+{a1 - a0:.2f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(yy); ax.set_yticklabels([g[0] for g in G]); ax.set_xlim(0.45, 1.02)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_xlabel("labeler agreement with gold on dev (200 examples)")
fig.legend(handles=[Line2D([], [], marker="o", color=MUTED, ls="", label="plain zero-shot prompt (v1)"), Line2D([], [], marker="o", color=ORANGE, ls="", label="v2: guidelines, seeds, batching or voting")],
          loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("paper/figures/labeling.pdf", bbox_inches="tight"); fig.savefig("paper/figures/labeling.png", dpi=200, bbox_inches="tight")
print("figures written")
