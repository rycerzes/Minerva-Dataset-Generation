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
    """Nested CV at matched coverage. The earlier version of this figure showed the
    engine ahead of ScanCode; that came from predictions whose accept threshold had
    been calibrated on the queries it was scored against, and it did not survive
    nested cross-validation. Level is the honest picture."""
    labels = ["DEP-5\nprecision", "DEP-5\nR@1", "SPDX-tag\nprecision", "SPDX-tag\nR@1"]
    agent = [0.8772, 0.8301, 0.9533, 0.9211]      # nested CV, tau 0.20
    scan = [0.8404, 0.8316, 0.9727, 0.9361]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar([i - 0.19 for i in x], agent, 0.38, label="Atarashi (nested CV, τ=0.20)", color=AGENT)
    ax.bar([i + 0.19 for i in x], scan, 0.38, label="ScanCode", color=SCAN)
    for i, (a, b) in enumerate(zip(agent, scan)):
        ax.annotate(f"{a:.3f}", (i - 0.19, a), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5, color=INK)
        ax.annotate(f"{b:.3f}", (i + 0.19, b), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.75, 1.06); ax.set_ylabel("score", fontsize=9)
    ax.set_title("Head to head, nested CV at matched coverage — level, not ahead\n"
                 "McNemar p = 1.000 (DEP-5 R@1), 0.454 (SPDX-tag R@1)",
                 fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    _style(ax)
    return save(fig, "head-to-head")


def fig_prevalence() -> str:
    """What the prevalence pool found: a defect no notice-regime benchmark could see."""
    labels = ["overall R@1", "precision", "exact-set", "Apache-2.0\n(n=639)"]
    before = [0.9468, 0.9558, 0.9459, 0.48]
    after = [0.9752, 0.9844, 0.9742, 0.95]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar([i - 0.19 for i in x], before, 0.38, label="leader-gated", color=MUTED)
    ax.bar([i + 0.19 for i in x], after, 0.38, label="list-gated", color=GOOD)
    for i, (a, b) in enumerate(zip(before, after)):
        ax.annotate(f"{a:.3f}", (i - 0.19, a), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5, color=INK)
        ax.annotate(f"{b:.3f}", (i + 0.19, b), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.0, 1.08); ax.set_ylabel("score", fontsize=9)
    ax.set_title("Prevalence pool, n=11,680 (±0.0028): 286 of 639 Apache-2.0 files\n"
                 "were reported as ImageMagick until the ranker gate was fixed",
                 fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax)
    return save(fig, "prevalence")


def fig_trajectory() -> str:
    """Where the gain actually came from."""
    steps = ["first\nmeasurement", "GPL naming\nbridge", "corroboration\nfeatures",
             "learned\nreject option", "compound layer\n+ list gate"]
    prec = [0.652, 0.795, 0.853, 0.873, 0.8622]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(range(len(steps)), prec, "-o", color=AGENT, lw=2, ms=7,
            label="Atarashi, DEP-5 precision")
    ax.axhline(0.8404, color=SCAN, ls="--", lw=1.6, label="ScanCode (0.8404)")
    for i, v in enumerate(prec):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=INK)
    ax.annotate("defects, not modelling", xy=(0.5, 0.72), fontsize=9, color=MUTED,
                ha="center", style="italic")
    ax.annotate("learned ranker", xy=(2.5, 0.92), fontsize=9, color=MUTED,
                ha="center", style="italic")
    ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, fontsize=8.5)
    ax.set_ylim(0.60, 0.96); ax.set_ylabel("precision", fontsize=9)
    ax.set_title("DEP-5 precision over the rebuild", fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    return save(fig, "trajectory")


def fig_riskcoverage() -> str:
    """The curve, not a point. Comparing two engines at different coverages is what
    made the earlier read confusing; alpha answers "is the truth in the set", which is
    not the question the accept decision asks."""
    tau = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    deb_cov = [0.975, 0.967, 0.952, 0.946, 0.893, 0.858, 0.824]
    deb_prec = [0.8547, 0.8582, 0.8717, 0.8772, 0.8948, 0.9132, 0.9241]
    spdx_cov = [0.981, 0.977, 0.974, 0.966, 0.947, 0.902, 0.857]
    spdx_prec = [0.9464, 0.9462, 0.9498, 0.9533, 0.9563, 0.9667, 0.9737]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for a, cov, prec, sc_c, sc_p, name in (
            (ax, deb_cov, deb_prec, 0.990, 0.8404, "DEP-5"),
            (ax2, spdx_cov, spdx_prec, 0.962, 0.9727, "SPDX-tag")):
        a.plot(cov, prec, "-o", color=AGENT, lw=2, ms=5, label="Atarashi (nested CV)")
        a.plot([sc_c], [sc_p], "*", color=SCAN, ms=16, label="ScanCode")
        for t, c, pr in zip(tau, cov, prec):
            if t in (0.05, 0.20, 0.50):
                a.annotate(f"τ={t}", (c, pr), textcoords="offset points",
                           xytext=(4, -10), fontsize=8, color=MUTED)
        a.set_xlabel("coverage", fontsize=9); a.set_title(name, fontsize=10, color=INK)
        _style(a)
    ax.set_ylabel("precision", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Risk-coverage curve — ScanCode is one point on it",
                 fontsize=10.5, color=INK, y=1.03)
    fig.tight_layout()
    return save(fig, "risk-coverage")


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
        "prevalence": fig_prevalence(),
        "trajectory": fig_trajectory(),
        "risk-coverage": fig_riskcoverage(),
        "generalisation": fig_generalisation(),
    }
    payload = OUT / "figures.json"
    payload.write_text(json.dumps(figs))
    for name, data in figs.items():
        print(f"  {name:16s} {len(data)/1024:6.0f} KB base64")
    print(f"\nwrote {payload} and PNGs alongside")


if __name__ == "__main__":
    main()
