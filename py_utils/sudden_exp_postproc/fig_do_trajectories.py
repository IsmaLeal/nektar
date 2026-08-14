#!/usr/bin/env python3
"""Research-plan figure for subsection 3.4: asymmetry histories M_p(t) of
a few transitioning realisations of the DO ensemble, so that several
crossings between the two attached states are visible in both directions.

python3 py_utils/sudden_exp_postproc/fig_do_trajectories.py \
  --run cases/se/runs/2026_07_17_do_total_sig0p0144_t1000 \
  --out ~/todisimo/studies/7.0_epfl/year1/academic/candidacy_exam/tex/figures/do_trajectories.pdf
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from do_quicklook import mp_series
from do_stats_plots import transitions_per_particle

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--mbar", type=float, default=0.1935)
ap.add_argument("--c", type=float, default=0.8)
ap.add_argument("--tmin", type=float, default=30.0)
ap.add_argument("--nshow", type=int, default=4)
args = ap.parse_args()

t, yi, Mp, _ = mp_series(args.run)
trans = transitions_per_particle(t, Mp, args.mbar, args.c, args.tmin)

# The realisations with the most crossings, so several jumps are visible.
order = np.argsort([-len(tr) for tr in trans])
show = order[: args.nshow]

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

fig, ax = plt.subplots(figsize=(4.1, 1.9), constrained_layout=True)
for q, col in zip(show, COLORS):
    ax.plot(t, Mp[:, q], lw=0.7, color=col)
for s in (1, -1):
    ax.axhline(s * args.mbar, ls=":", color="0.45", lw=0.7)
ax.text(1.012, args.mbar, "$+\\bar{M}$", color="0.35", fontsize=8,
        transform=ax.get_yaxis_transform(), ha="left", va="center",
        clip_on=False)
ax.text(1.012, -args.mbar, "$-\\bar{M}$", color="0.35", fontsize=8,
        transform=ax.get_yaxis_transform(), ha="left", va="center",
        clip_on=False)
ax.set_xlim(t[0], t[-1])
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("$t$")
ax.set_ylabel("$M_p$", fontsize=11)

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out)
png = args.out.with_suffix(".png")
fig.savefig(png, dpi=200)
print(f"Saved: {args.out} and {png}")
print("realisations:", ", ".join(f"{q} ({len(trans[q])} crossings)"
                                 for q in show))
