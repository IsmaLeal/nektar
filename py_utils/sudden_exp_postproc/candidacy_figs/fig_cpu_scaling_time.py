#!/usr/bin/env python3
"""House-style figure: strong-scaling sweep of the DO solver on JED.

Time-integration wall time of a fixed 200-step phi22 DO run at 4-72 MPI
ranks, against the ideal 1/n reference through the 4-rank point, with the
parallel efficiency relative to 4 ranks at each point. The stepping time
is the production-relevant number: Total Computation Time is dominated by
the one-off POD snapshot load (~200 s floor), which is negligible over a
production run's ~10^5-10^6 steps.

Data: the Time-integration lines of slurm-66317035/66317039/66317040
(n4/n8/n12, 2026-09-01) and slurm-66126008/66126097/66126098
(n18/n36/n72, 2026-08) in /scratch/lealaske/se_runs/do_phi22_n*/ on JED;
identical casefiles, only the rank count differs.

Usage:
    python3 fig_cpu_scaling_time.py --out FIG.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

RANKS = np.array([4, 8, 12, 18, 36, 72])
TINT = np.array([33.4003, 18.6686, 12.7399, 8.93453, 5.70279, 4.52889])
EFF = TINT[0] * RANKS[0] / (RANKS * TINT)

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

fig, ax = plt.subplots(figsize=(3.2, 2.35), constrained_layout=True)

ax.loglog(RANKS, TINT, "o-", color="#0072B2", lw=1.0, ms=3.5)
ref = TINT[0] * RANKS[0] / RANKS
ax.loglog(RANKS, ref, "--", color="0.45", lw=0.8)

for n, t, e in zip(RANKS[1:], TINT[1:], EFF[1:]):
    ax.annotate(f"{100 * e:.0f}\\%", (n, t), xytext=(0, 4),
                textcoords="offset points", fontsize=7, color="#0072B2",
                ha="center", va="bottom")
ax.annotate("measured, with parallel\nefficiency vs 4 ranks",
            (4.3, 33.4), xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="#0072B2", ha="left", va="bottom")
ax.annotate(r"ideal $\propto 1/n$", (36, TINT[0] * 4 / 36), xytext=(-2, -12),
            textcoords="offset points", fontsize=8, color="0.45",
            ha="center", va="top")

ax.set_xlabel(r"MPI ranks $n$", fontsize=10)
ax.set_ylabel(r"$T_{\mathrm{int}}$, 200 steps (s)", fontsize=10)
ax.set_xticks([4, 8, 12, 18, 36, 72])
ax.set_xticklabels(["4", "8", "12", "18", "36", "72"])
ax.set_yticks([2, 5, 10, 20, 50])
ax.set_yticklabels(["2", "5", "10", "20", "50"])
ax.set_ylim(1.5, 60)
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out)
fig.savefig(args.out.with_suffix(".png"), dpi=200)
print(f"Saved: {args.out} (+.png)")
