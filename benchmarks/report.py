"""Build REPORT.md and figures from benchmarks/results/<task>/result.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.tasks import MODES

# palette (dataviz reference instance; categorical slots in fixed order, Jev is a muted reference, not a series)
SERIES = {"ultrabert": "#2a78d6", "ultrabert-embed": "#eb6834", "gte-modernbert": "#1baf7a"}
VARIANT = {"silver": "#2a78d6", "seeded": "#eb6834", "gold": "#1baf7a"}
MUTED, GRID, INK, INK2, SURFACE = "#898781", "#e1e0d9", "#0b0b0b", "#52514e", "#fcfcfb"
METRIC_LABEL = {"accuracy": "accuracy", "f1": "F1", "spearman": "Spearman ρ", "group_spearman": "mean per-doc Spearman ρ", "argmax_accuracy": "argmax accuracy", "mean_spearman": "mean Spearman ρ"}


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def labeler_quality(name: str, metric: str) -> dict[str, float | None]:
    """The LLM labels' own agreement with gold on the training pool (the ceiling for silver training)."""
    import numpy as np

    from benchmarks.tasks import get_task
    from ownyourjev.data import read_jsonl
    from ownyourjev.evaluate import evaluate

    out: dict[str, float | None] = {}
    for variant, fn in (("silver", "silver.jsonl"), ("seeded", "silver_seeded.jsonl")):
        p = Path("data/silver") / name / fn
        if not p.is_file():
            out[variant] = None
            continue
        try:
            t = get_task(name)
            h = t.spec.primary_head
            q = t.spec.parsed[h]
            exs = [e for e in read_jsonl(p) if e.valid and e.gold and h in e.gold]
            if not exs:
                out[variant] = None
                continue
            golds = [e.gold[h] for e in exs]
            if q.type == "noul":
                probs = np.array([1.0 if e.labels[h] else 0.0 for e in exs])
            else:
                probs = np.zeros((len(exs), q.num_outputs))
                for i, e in enumerate(exs):
                    probs[i, int(e.labels[h]) if q.type == "score" else q.options.index(e.labels[h])] = 1.0
            m = evaluate(q, golds, probs)
            out[variant] = m.get(metric if metric in m else {"group_spearman": "spearman", "mean_spearman": "spearman"}.get(metric, "accuracy"))
        except Exception:
            out[variant] = None
    return out


def load_results(results: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(results.glob("*/result.json")):
        out[p.parent.name] = json.loads(p.read_text())
    return out


def fmt(v, nd=3):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


MULTI_DIM = {"summeval": "group_spearman", "helpsteer2": "spearman"}  # Jev reports the mean over all rubric dimensions


def multi_dim_mean(r: dict, variant: str = "silver") -> tuple[float | None, dict[str, float]]:
    """Mean metric over the primary head and all secondary heads (silver variant trains every head)."""
    name = r["task"]
    if name not in MULTI_DIM or variant not in r["variants"]:
        return None, {}
    key = MULTI_DIM[name]
    v = r["variants"][variant]
    per = {r["head"]: v["final"].get(key)}
    for h, m in (v.get("other_heads") or {}).items():
        per[h] = m.get(key)
    vals = [x for x in per.values() if x is not None and x == x]
    return (sum(vals) / len(vals) if vals else None), per


def labels_needed(r: dict, variant: str = "silver") -> int | None:
    """Smallest training-set size at which any (backbone, method) learning curve reaches Jev's published value.
    Only reported when the final silver model itself beats Jev (curves are scored on a subset of the eval split)."""
    jev = r["jev_published"].get("jev")
    if jev is None or variant not in r["variants"]:
        return None
    mean, _ = multi_dim_mean(r, variant)
    final = mean if mean is not None else r["variants"][variant]["final"]["primary_value"]
    if final is None or final < jev:
        return None
    best = None
    for curve in r["variants"][variant]["learning_curves"].values():
        for pt in curve:
            if pt["test"] is not None and pt["test"] >= jev and (best is None or pt["n"] < best):
                best = pt["n"]
    return best


def plot_learning_curves(name: str, r: dict, out: Path) -> Path | None:
    v = r["variants"].get("silver")
    if not v:
        return None
    curves = v["learning_curves"]
    methods = ["frozen", "lora", "full"]
    present = [m for m in methods if any(k.endswith("/" + m) for k in curves)]
    if not present:
        return None
    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 3.4), sharey=True, facecolor=SURFACE)
    axes = [axes] if len(present) == 1 else list(axes)
    jev = r["jev_published"].get("jev")
    ci = r["jev_published"].get("ci95")
    metric = r["metric"]
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    all_n = sorted({pt["n"] for pts in curves.values() for pt in pts})
    for ax, m in zip(axes, present):
        _style(ax, f"{m}", "LLM-labeled examples", METRIC_LABEL.get(metric, metric) + " on eval split")
        ax.set_xscale("log")
        ax.set_xticks(all_n)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelrotation=45)
        if jev is not None:
            if ci:
                ax.axhspan(ci[0], ci[1], color=GRID, alpha=0.6, lw=0)
            ax.axhline(jev, color=MUTED, ls="--", lw=1.2)
            ax.text(0.01, jev, f"Jev {jev:.3f}", color=MUTED, fontsize=8, va="bottom", ha="left", transform=ax.get_yaxis_transform())
        ends = []
        for bb, color in SERIES.items():
            pts = curves.get(f"{bb}/{m}")
            if not pts:
                continue
            xs = [p["n"] for p in pts]
            ys = [p["test"] if p["test"] is not None else float("nan") for p in pts]
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=4, label=bb)
            ends.append((ys[-1], xs[-1], bb))
        # direct labels at the line ends, staggered so they never overlap
        ends.sort()
        last = -1e9
        for y, x, bb in ends:
            y_lab = max(y, last + 0.045 * (ax.get_ylim()[1] - ax.get_ylim()[0])) if ends else y
            ax.annotate(bb, (x, y), xytext=(5, (y_lab - y) * 0), textcoords="offset points", fontsize=8, color=INK2, va="center") if y_lab == y else ax.annotate(bb, (x, y_lab), xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
            last = y_lab
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"{name}: test {METRIC_LABEL.get(metric, metric)} vs. number of LLM-labeled examples", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    p = out / "figures" / f"{name}_learning_curve.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return p


def plot_overview(results: dict[str, dict], out: Path) -> Path:
    rows = [(mode, n) for mode in ("noul", "choice", "score") for n in MODES[mode] if n in results]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.6), facecolor=SURFACE)
    _style(ax, "Published Jev-1.13 vs. ownyourjev (encoder trained on LLM labels)", "primary metric on the published eval split", "")
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    ys = []
    for i, (mode, name) in enumerate(rows):
        r = results[name]
        y = len(rows) - i
        ys.append(y)
        jev = r["jev_published"].get("jev")
        ci = r["jev_published"].get("ci95")
        if ci:
            ax.plot(ci, [y, y], color=MUTED, lw=4, alpha=0.35, solid_capstyle="butt")
        if jev is not None:
            ax.plot([jev], [y], marker="D", color=MUTED, ms=6, ls="")
        for k, (variant, color) in enumerate(VARIANT.items()):
            v = r["variants"].get(variant)
            if not v:
                continue
            val = v["final"]["primary_value"]
            lo, hi = v["final"].get("ci95", [val, val])
            mean, _ = multi_dim_mean(r, variant)
            if mean is not None:  # multi-dimension rubric: plot the mean over dimensions (no CI)
                val, lo, hi = mean, mean, mean
            off = (k - 1) * 0.18
            ax.plot([lo, hi], [y + off, y + off], color=color, lw=1.2, alpha=0.8)
            ax.plot([val], [y + off], marker="o", color=color, ms=5, ls="", mec=SURFACE, mew=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{n}  ({results[n]['metric']})" for _, n in rows], fontsize=8.5, color=INK)
    ax.set_xlim(0, 1.0)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="D", color=MUTED, ls="", label="Jev-1.13 (published, 95% CI band)")] + [Line2D([], [], marker="o", color=c, ls="", label=f"ours: {v} labels") for v, c in VARIANT.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_title("Published Jev-1.13 vs. ownyourjev (encoder trained on LLM labels)", color=INK, fontsize=11, loc="left", pad=44)
    fig.tight_layout()
    p = out / "figures" / "overview.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return p


def write_report(results_dir: Path) -> Path:
    results = load_results(results_dir)
    out = results_dir
    lines = ["# ownyourjev benchmark report", ""]
    lines.append("Jev numbers are the published zero-shot results of jev-1.13.0 (arXiv:2609.37647, Table 2) on the same evaluation splits. "
                 "Our models are 150M-parameter encoders trained on labels produced by `deepseek/deepseek-v4.1-flash`; model selection, early stopping and calibration used only held-out LLM labels. "
                 "`silver` = LLM labels only; `seeded` = LLM labels with 20 gold examples shown in the prompt; `gold` = same recipe on gold labels (upper bound). Bold = our 95% CI lies entirely above Jev's published value.")
    lines.append("")
    if results:
        plot_overview(results, out)
        lines += ["![overview](figures/overview.png)", ""]
    notes = [
        "Notes:",
        "- *LLM labels to beat Jev*: the smallest size on the data-size ladder at which a learning curve passes Jev, shown only when the final silver model beats Jev.",
        "- SMS Spam and SummEval have no separate training split: the eval set is labeled by the LLM and scored with k-fold cross-fitting (no model ever scores texts it was trained on).",
        "- HelpSteer2 `seeded` was skipped by the budget guard: 20 prompt/response pairs in every labeling prompt were estimated at $4.41, over the $3 per-task cap.",
        "- SummEval/HelpSteer2 `seeded` and `gold` train the first dimension only; the silver row is the mean over all dimensions, which is what Jev's number is.",
        "",
    ]
    lines += notes
    spend = max((r.get("spend_usd_total_so_far", 0) for r in results.values()), default=0)
    for mode in ("noul", "choice", "score"):
        names = [n for n in MODES[mode] if n in results]
        if not names:
            continue
        lines += [f"## {mode.capitalize()}", "", "| task | metric | n eval | Jev (95% CI) | ours silver (95% CI) | seeded | gold | LLM labels vs gold (silver / seeded) | recipe (silver) | LLM labels to beat Jev |", "|---|---|---|---|---|---|---|---|---|---|"]
        for n in names:
            r = results[n]
            jev, ci = r["jev_published"].get("jev"), r["jev_published"].get("ci95") or [None, None]
            cells = []
            for variant in ("silver", "seeded", "gold"):
                v = r["variants"].get(variant)
                if not v:
                    cells.append("–")
                    continue
                val, (lo, hi) = v["final"]["primary_value"], v["final"].get("ci95", [None, None])
                s = f"{fmt(val)}" + (f" ({fmt(lo)}–{fmt(hi)})" if variant == "silver" and lo is not None else "")
                if jev is not None and lo is not None and lo > jev:
                    s = f"**{s}**"
                cells.append(s)
            mean, _ = multi_dim_mean(r)
            if mean is not None:
                cells[0] = f"{fmt(mean)} (mean of all dims)" + ("" if mean <= (jev or 0) else "")
                cells[1] = (cells[1] + f" ({r['head']} only)") if cells[1] != "–" else cells[1]
                cells[2] = (cells[2] + f" ({r['head']} only)") if cells[2] != "–" else cells[2]
            sv = r["variants"].get("silver", {})
            recipe = f"{sv['recipe']['backbone']}/{sv['recipe']['method']} n={sv['n_train']}" if sv else "–"
            ln = labels_needed(r)
            lq = labeler_quality(n, r["metric"])
            lines.append(f"| {n} | {r['metric']} | {r['n_eval']} | {fmt(jev)} ({fmt(ci[0])}–{fmt(ci[1])}) | {cells[0]} | {cells[1]} | {cells[2]} | {fmt(lq.get('silver'))} / {fmt(lq.get('seeded'))} | {recipe} | {ln if ln else 'not reached'} |")
        lines.append("")
    # strategy + calibration + overlap
    lines += ["## Training strategy, cost and calibration", "", "| task | selected | search GPU-s | final train s | ECE ours | ECE Jev | Brier ours | we fix Jev errors | we break Jev correct | both wrong |", "|---|---|---|---|---|---|---|---|---|---|"]
    for mode in ("noul", "choice", "score"):
        for n in MODES[mode]:
            if n not in results or "silver" not in results[n]["variants"]:
                continue
            r = results[n]
            v = r["variants"]["silver"]
            f = v["final"]
            o = v.get("overlap_with_jev") or {}
            lines.append(f"| {n} | {v['recipe']['backbone']}/{v['recipe']['method']} n={v['n_train']} | {v['gpu_seconds_search']:.0f} | {f.get('gpu_s_final', float('nan')):.0f} | {fmt(f.get('ece'))} | {fmt(r['jev_published'].get('jev_ece'))} | {fmt(f.get('brier'))} | {fmt(o.get('we_fix_jev_errors'))} | {fmt(o.get('we_break_jev_correct'))} | {fmt(o.get('both_wrong'))} |")
    lines.append("")
    lines += ["## Best result per (backbone, method) on silver labels", ""]
    for n, r in results.items():
        v = r["variants"].get("silver")
        if not v:
            continue
        lines += [f"**{n}** ({r['metric']}, Jev {fmt(r['jev_published'].get('jev'))})", "", "| backbone | method | n | silver-val | test | train s |", "|---|---|---|---|---|---|"]
        for row in v["best_by_setting"]:
            lines.append(f"| {row['backbone']} | {row['method']} | {row['n']} | {fmt(row['val'])} | {fmt(row['test'])} | {row['train_s']:.0f} |")
        lines.append("")
        p = plot_learning_curves(n, r, out)
        if p:
            lines += [f"![{n}](figures/{p.name})", ""]
    md = [(n, r) for n, r in results.items() if n in MULTI_DIM and "silver" in r["variants"]]
    if md:
        lines += ["## Multi-dimension rubric tasks (silver labels, one model per dimension)", "", "| task | dimension | ours | Jev (published, where reported) |", "|---|---|---|---|"]
        for n, r in md:
            mean, per = multi_dim_mean(r)
            jd = r["jev_published"].get("jev_per_dim", {})
            for h, v in per.items():
                lines.append(f"| {n} | {h} | {fmt(v)} | {fmt(jd.get(h))} |")
            lines.append(f"| {n} | **mean** | **{fmt(mean)}** | **{fmt(r['jev_published'].get('jev'))}** |")
        lines.append("")
    syn = {n: json.loads((results_dir / n / "synthetic.json").read_text()) for n in results if (results_dir / n / "synthetic.json").is_file()}
    if syn:
        lines += ["## Synthetic data only (no real texts)", "", "Examples generated by the labeler LLM from the question spec alone; `guided` additionally shows the generator 20 real labeled examples (never trained on).", "", "| task | variant | n train | metric | ours (95% CI) | Jev |", "|---|---|---|---|---|---|"]
        for n, d in syn.items():
            for variant, v in d.items():
                lo, hi = v["final"]["ci95"]
                lines.append(f"| {n} | {variant} | {v['n_train']} | {results[n]['metric']} | {fmt(v['final']['primary_value'])} ({fmt(lo)}–{fmt(hi)}) | {fmt(results[n]['jev_published'].get('jev'))} |")
        lines.append("")
    # other heads
    extra = [(n, h, m) for n, r in results.items() for h, m in (r["variants"].get("silver", {}).get("other_heads") or {}).items()]
    if extra:
        lines += ["## Secondary heads (same recipe as the primary head)", "", "| task | head | accuracy/F1 | Spearman | ECE |", "|---|---|---|---|---|"]
        for n, h, m in extra:
            lines.append(f"| {n} | {h} | {fmt(m.get('f1', m.get('accuracy')))} | {fmt(m.get('spearman'))} | {fmt(m.get('ece'))} |")
        lines.append("")
    lines += ["## Spend", "", f"OpenRouter spend recorded by the ledger so far: **${spend:.2f}** (cap $15; experiments budget $20).", ""]
    p = out / "REPORT.md"
    p.write_text("\n".join(lines))
    return p
