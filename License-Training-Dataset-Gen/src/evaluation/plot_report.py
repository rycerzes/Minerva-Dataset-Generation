"""Figures for the Atarashi engine report.

Every number here is read from the recorded result files or restated from a
measurement in the report, never recomputed by hand — a figure that disagrees with
the table beside it is worse than no figure.

    uv run src/evaluation/plot_report.py
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent.parent / "docs" / "figs"
INK, MUTED, GOOD, BAD = "#1b1b1b", "#8a8a8a", "#2b7a4b", "#b03030"
AGENT, SCAN = "#2c6fbb", "#c0762c"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=140, bbox_inches="tight", facecolor="white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_head_to_head() -> str:
    """The headline. Held-out figures, because the in-sample ones flatter."""
    corpora = ["DEP-5\nprecision", "DEP-5\nR@1", "SPDX-tag\nprecision", "SPDX-tag\nR@1"]
    agent = [0.8831, 0.8331, 0.9651, 0.9361]
    scancode = [0.8404, 0.8316, 0.9727, 0.9361]
    x = range(len(corpora))
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar([i - 0.19 for i in x], agent, 0.38, label="Atarashi (held-out)", color=AGENT)
    ax.bar([i + 0.19 for i in x], scancode, 0.38, label="ScanCode", color=SCAN)
    for i, (a, s) in enumerate(zip(agent, scancode)):
        d = a - s
        ax.text(i, max(a, s) + 0.012, f"{d:+.4f}", ha="center", fontsize=8.5,
                color=GOOD if d > 0.0005 else (MUTED if abs(d) <= 0.0005 else BAD))
    ax.set_xticks(list(x)); ax.set_xticklabels(corpora, fontsize=9)
    ax.set_ylim(0.75, 1.06); ax.set_ylabel("score", fontsize=9)
    ax.set_title("Head to head, held out — ahead on DEP-5, level on SPDX-tag R@1",
                 fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    _style(ax)
    return save(fig, "head-to-head")


def fig_trajectory() -> str:
    """Where the gain actually came from."""
    steps = ["first\nmeasurement", "GPL naming\nbridge", "corroboration\nfeatures",
             "learned\nreject option"]
    prec = [0.652, 0.795, 0.853, 0.873]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(range(len(steps)), prec, "-o", color=AGENT, lw=2, ms=7,
            label="Atarashi, DEP-5 precision")
    ax.axhline(0.8404, color=SCAN, ls="--", lw=1.6, label="ScanCode (0.8404)")
    for i, v in enumerate(prec):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=INK)
    ax.annotate("defects, not modelling", xy=(0.5, 0.72), fontsize=9, color=MUTED,
                ha="center", style="italic")
    ax.annotate("learned ranker", xy=(2.5, 0.90), fontsize=9, color=MUTED,
                ha="center", style="italic")
    ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, fontsize=8.5)
    ax.set_ylim(0.60, 0.94); ax.set_ylabel("precision", fontsize=9)
    ax.set_title("DEP-5 precision over the rebuild", fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    return save(fig, "trajectory")


def fig_conformal() -> str:
    """Coverage tracks the target; set size is the price."""
    alpha = [0.20, 0.10, 0.05, 0.02]
    target = [1 - a for a in alpha]
    realised = [0.794, 0.931, 0.968, 0.995]
    size = [0.82, 0.99, 1.38, 2.93]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax.plot(alpha, target, "--", color=MUTED, lw=1.5, label="target 1−α")
    ax.plot(alpha, realised, "-o", color=AGENT, lw=2, ms=6, label="realised")
    ax.invert_xaxis(); ax.set_xlabel("α", fontsize=9)
    ax.set_ylabel("coverage", fontsize=9); ax.set_ylim(0.75, 1.01)
    ax.set_title("Coverage tracks the target", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left"); _style(ax)

    ax2.plot(alpha, size, "-o", color=BAD, lw=2, ms=6)
    ax2.axhline(1.0, color=MUTED, ls=":", lw=1.2)
    ax2.annotate("shipped α=0.10", (0.10, 0.99), textcoords="offset points",
                 xytext=(-6, 34), ha="right", fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax2.invert_xaxis(); ax2.set_xlabel("α", fontsize=9)
    ax2.set_ylabel("mean set size", fontsize=9)
    ax2.set_title("…and set size is what it costs", fontsize=10, color=INK)
    _style(ax2)
    fig.suptitle("Conformal calibration of the accept threshold", fontsize=10.5,
                 color=INK, y=1.03)
    return save(fig, "conformal")


def fig_generalisation() -> str:
    """Tree depth trades in-distribution gain against out-of-distribution safety."""
    depth = [1, 2, 3]
    in_dist = [0.021, 0.059, 0.068]
    ood = [0.008, 0.000, -0.013]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(depth, in_dist, "-o", color=AGENT, lw=2, ms=7, label="in-distribution (grouped CV)")
    ax.plot(depth, ood, "-s", color=BAD, lw=2, ms=7, label="out-of-distribution (leave-family-out)")
    ax.axhline(0, color=MUTED, lw=1)
    ax.annotate("shipped", (2, 0.059), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=8.5, color=INK)
    ax.annotate("memorises families", (3, -0.013), textcoords="offset points",
                xytext=(-6, -16), ha="right", fontsize=8.5, color=MUTED, style="italic")
    ax.set_xticks(depth); ax.set_xlabel("tree depth", fontsize=9)
    ax.set_ylabel("Δ top-1 over hand-tuned ranking", fontsize=9)
    ax.set_ylim(-0.024, 0.092)
    ax.set_title("Capacity trades in-distribution gain against generalisation\n"
                 "(after within-query normalisation)", fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left"); _style(ax)
    return save(fig, "generalisation")


def main() -> None:
    figs = {
        "head-to-head": fig_head_to_head(),
        "trajectory": fig_trajectory(),
        "conformal": fig_conformal(),
        "generalisation": fig_generalisation(),
    }
    payload = OUT / "figures.json"
    payload.write_text(json.dumps(figs))
    for name, data in figs.items():
        print(f"  {name:16s} {len(data)/1024:6.0f} KB base64")
    print(f"\nwrote {payload} and PNGs alongside")


if __name__ == "__main__":
    main()
