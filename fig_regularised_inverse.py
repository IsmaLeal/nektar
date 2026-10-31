#!/usr/bin/env python3
"""House-style inset for the solver slide of the candidacy talk: the
regularised covariance inverse of Eq. (33) of the plan,
    C_ii/(C_ii^2 + delta^2),
against C_ii on linear axes, with the bare 1/C_ii as the grey reference. Axes
are plotted in units of delta (ticks labelled delta, 2delta, ... and 1/delta,
1/(2delta)), so the figure carries no case-specific number: the curve follows
1/C_ii above the floor, peaks at C_ii = delta with value 1/(2 delta), and goes
to zero below it instead of diverging.

Usage:
    python3 fig_regularised_inverse.py --out FIG.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8,
})

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

x = np.linspace(0.0, 4.0, 400)          # C_ii in units of delta
fig, ax = plt.subplots(figsize=(1.45, 1.15), constrained_layout=True)
xb = x[x > 0.05]
ax.plot(xb, 1.0 / xb, "--", color="0.45", lw=0.8)          # bare inverse, times delta
ax.plot(x, x / (1.0 + x**2), "-", color="#0072B2", lw=1.0)  # regularised, times delta
ax.annotate(r"$1/C_{ii}$", (1.2, 0.92), fontsize=7, color="0.45",
            ha="left", va="bottom")
ax.annotate(r"$C_{ii}/(C_{ii}^2+\delta^2)$", (1.45, 0.20), fontsize=7,
            color="#0072B2", ha="left", va="top")
ax.set_xlabel(r"$C_{ii}$", fontsize=8, labelpad=1)
ax.set_xlim(0.0, 4.0)
ax.set_ylim(0.0, 1.1)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(["0", r"$\delta$", r"$2\delta$", r"$3\delta$", r"$4\delta$"])
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels(["0", r"$1/2\delta$", r"$1/\delta$"])
ax.tick_params(labelsize=7, pad=1.5)
fig.savefig(args.out)
print("wrote", args.out)
