#!/usr/bin/env python3
"""House-style figure: rank-count selection for the DO solver on JED.

One curve: wall time times CPU cost, n*T_int^2, of a fixed 200-step
phi22 DO run at 4-72 MPI ranks, normalised to its minimum. The minimum
is the rank count that balances hours waited against CPU-hours burned
with equal weight; a cheaper-leaning choice moves left along the basin,
a faster-leaning one right.

The stepping time T_int is the production-relevant number: Total
Computation Time is dominated by the one-off POD snapshot load (~200 s
floor), negligible over a production run's ~10^5-10^6 steps.

Data: the Time-integration lines of slurm-66317035/66317039/66317040
(n4/n8/n12, 2026-09-01) and slurm-66126008/66126097/66126098
(n18/n36/n72, 2026-08) in /scratch/lealaske/se_runs/do_phi22_n*/ on JED;
identical casefiles, only the rank count differs.

Usage:
    python3 fig_cpu_scaling.py --out FIG.pdf
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
PROD = RANKS * TINT ** 2
#PROD = PROD / PROD.min()

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

fig, ax = plt.subplots(figsize=(3.2, 2.35), constrained_layout=True)

ax.semilogx(RANKS, PROD, "o-", color="#0072B2", lw=1.0, ms=3.5)
i = int(np.argmin(PROD))
ax.plot(RANKS[i], PROD[i], "o", mfc="none", mec="#0072B2", ms=7, mew=1.0)
ax.annotate("minimum", (RANKS[i], PROD[i]), xytext=(0, 9),
            textcoords="offset points", fontsize=8, color="#0072B2",
            ha="center", va="bottom")

ax.set_xlabel(r"MPI ranks $n$", fontsize=10)
ax.set_ylabel(r"time $\times$ cost $= n\,T_{\mathrm{int}}^{2}$",
              fontsize=9)
ax.set_xticks([4, 8, 12, 18, 36, 72])
ax.set_xticklabels(["4", "8", "12", "18", "36", "72"])
#ax.set_ylim(0, 4.2)
ax.xaxis.set_minor_formatter(NullFormatter())

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out)
fig.savefig(args.out.with_suffix(".png"), dpi=200)
print(f"Saved: {args.out} (+.png)")
