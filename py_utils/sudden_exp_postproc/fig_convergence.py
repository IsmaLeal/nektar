#!/usr/bin/env python3
"""House-style figure for the research plan, subsubsection 3.3.1:
(a) successive-pair distances of the gauge-invariant observable vs dt,
    with slope guides: whole-domain differences follow the sqrt(nu*dt)
    projection wall layer of the base scheme (identical with and without
    the DO layer); interior mean-field differences converge at the design
    order 2; the coefficient term at the designed first order;
(b) the mean-field difference between the two coarsest resolutions on the
    extraction plane, showing the wall layer the interior box excludes.

Data: the 2026_07_19_conv_* families through convergence_orders.py's own
loaders (cached gdT.dat extractions; no FieldConvert call needed).

Usage:
    python3 fig_convergence.py --out FIG.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter

from convergence_orders import CELL_AREA, FC_DEFAULT, do_state, load_dat, obs_dist, s0_state

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

RUNS = Path("/home/isma/nektar_src_full/cases/se/runs")
DO_TAGS = ["2026_07_19_conv_dt100", "2026_07_19_conv_dt050",
           "2026_07_19_conv_dt025", "2026_07_19_conv_dt0125"]
S0_TAGS = ["2026_07_19_conv_s0dt100", "2026_07_19_conv_s0dt050",
           "2026_07_19_conv_s0dt025", "2026_07_19_conv_s0dt0125"]
DTS = np.array([0.01, 0.005, 0.0025, 0.00125])
T = 2.0
XMIN, XMAX, YMAX = 4.0, 30.0, 0.9

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

DO = [do_state(RUNS / t, dt, T, FC_DEFAULT) for t, dt in zip(DO_TAGS, DTS)]
S0 = [s0_state(RUNS / t, FC_DEFAULT) for t in S0_TAGS]

x, y = DO[0][0], DO[0][1]
mask2 = np.concatenate([(x >= XMIN) & (x <= XMAX) & (np.abs(y) <= YMAX)] * 2)
do_obs = [(s[2], s[3], s[4]) for s in DO]
s0_obs = [(s[2], None, None) for s in S0]

rows = {}
for name, obs, msk in (("full", do_obs, None), ("int", do_obs, mask2),
                       ("s0full", s0_obs, None)):
    dm, dp = zip(*[obs_dist(obs[i], obs[i + 1], msk) for i in range(3)])
    rows[name] = (np.array(dm), np.array(dp))
dcomb_full = np.sqrt(rows["full"][0] ** 2 + rows["full"][1] ** 2)

fig, (axa, axb) = plt.subplots(
    1, 2, figsize=(6.5, 2.35), width_ratios=[1, 1.5],
    constrained_layout=True)

# (a) successive-pair distances
dpair = DTS[:3]
# the S=0 control lies within 4% of the full-norm curve, so it is drawn as
# markers alone: two overlaid lines of similar luminance merge in greyscale
series = [
    (dcomb_full,        "#D55E00", "o-", 0.5),
    (rows["s0full"][0], "0.30",    "s",  None),
    (rows["int"][0],    "#0072B2", "o-", 2.0),
    (rows["int"][1],    "#009E73", "^-", 1.0),
]
for d, c, fmt, pref in series:
    axa.loglog(dpair, d, fmt, color=c, lw=1.0, ms=3.5)
    if pref is not None:
        ref = d[-1] * (dpair / dpair[-1]) ** pref * 1.45
        axa.loglog(dpair, ref, "--", color=c, lw=0.8)
        expo = {0.5: "1/2", 2.0: "2", 1.0: "1"}[pref]
        axa.annotate(f"$\\Delta t^{{{expo}}}$", (dpair[-1], ref[-1]),
                     xytext=(1, 3), textcoords="offset points",
                     fontsize=8, color=c, ha="left", va="bottom")
axa.annotate("full norm,\nwhole domain", (2.6e-3, 2.45e-4), fontsize=7,
             color="#D55E00", ha="left", va="top")
axa.annotate("mean term, interior", (5.6e-3, 5.0e-5), fontsize=7,
             color="#0072B2", ha="left", va="top")
axa.annotate("fluctuation term, interior", (7.2e-3, 1.05e-5), fontsize=7,
             color="#009E73", ha="center", va="top")
axa.set_xlabel(r"$\Delta t$", fontsize=10)
axa.set_ylabel(r"$\Vert\mathcal{Q}_{\Delta t}-\mathcal{Q}_{\Delta t/2}\Vert$ and its terms",
               fontsize=9)
axa.set_ylim(4e-6, 2e-3)
axa.set_xticks([2.5e-3, 5e-3, 1e-2])
axa.set_xticklabels([r"$2.5\times10^{-3}$", r"$5\times10^{-3}$",
                     r"$10^{-2}$"])
axa.xaxis.set_minor_formatter(NullFormatter())
axa.text(-0.02, 1.07, "(a)", transform=axa.transAxes, va="top", ha="right",
         fontsize=11)

# (b) coarsest-pair mean-field difference on the plane, wall layer visible
d1 = load_dat(RUNS / DO_TAGS[0] / "gdT.dat", 6)
d2 = load_dat(RUNS / DO_TAGS[1] / "gdT.dat", 6)
X = d1[:, 0].reshape(121, 901)
Y = d1[:, 1].reshape(121, 901)
dmag = np.sqrt((d1[:, 3] - d2[:, 3]) ** 2
               + (d1[:, 4] - d2[:, 4]) ** 2).reshape(121, 901)
dmag = np.where((X < 0) & (np.abs(Y) > 0.5), np.nan, dmag)
vmax = np.nanmax(dmag)
im = axb.pcolormesh(X, Y, np.clip(dmag, vmax * 1e-3, None),
                    norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax),
                    cmap="Blues", rasterized=True)
axb.plot([-5, 0], [0.5, 0.5], "k-", lw=1.0)
axb.plot([-5, 0], [-0.5, -0.5], "k-", lw=1.0)
axb.plot([0, 0], [0.5, 1.5], "k-", lw=1.0)
axb.plot([0, 0], [-0.5, -1.5], "k-", lw=1.0)
axb.plot([0, 40], [1.5, 1.5], "k-", lw=1.0)
axb.plot([0, 40], [-1.5, -1.5], "k-", lw=1.0)
axb.add_patch(Rectangle((XMIN, -YMAX), XMAX - XMIN, 2 * YMAX,
                        fill=False, edgecolor="#009E73", lw=1.2))
axb.text(0.5 * (XMIN + XMAX), 0.0, "interior box", color="#009E73",
         fontsize=8, ha="center", va="center")
axb.set_xlim(-5, 40)
axb.set_ylim(-1.6, 1.6)
axb.set_box_aspect(0.28)
axb.set_xlabel("$x$", fontsize=10)
axb.set_ylabel("$y$", fontsize=10)
axb.text(-0.02, 1.07, "(b)", transform=axb.transAxes, va="top", ha="right",
         fontsize=11)
cb = fig.colorbar(im, ax=axb, pad=0.015, aspect=8, shrink=0.46)
cb.set_label(r"$|\Delta\bar{\mathbf{u}}|$", fontsize=9)
cb.ax.tick_params(labelsize=7)

# the wide strip of (b) is shorter than (a); freeze the constrained layout
# and lift (b) with its colorbar so the two panels share a top edge
fig.draw_without_rendering()
fig.set_layout_engine("none")
dy = axa.get_position().y1 - axb.get_position().y1
for a in (axb, cb.ax):
    pos = a.get_position()
    a.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out)
fig.savefig(args.out.with_suffix(".png"), dpi=200)
p_mean = np.log2(rows["int"][0][1] / rows["int"][0][2])
p_coef = np.log2(rows["int"][1][1] / rows["int"][1][2])
print(f"Saved: {args.out} (+.png)  finest-pair orders: "
      f"mean {p_mean:.2f}, coefficients {p_coef:.2f}")
