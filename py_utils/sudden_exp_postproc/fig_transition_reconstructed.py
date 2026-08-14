#!/usr/bin/env python3
"""Research-plan figure "a transition, reconstructed" for subsection 3.4:
(a) asymmetry histories M_p(t) of a few transitioning realisations, the
reconstructed one in blue with three instants marked on its curve, and
(b)-(d) that realisation's reconstructed velocity field
ubar + sum_i Y_i u_i at those instants, stacked to the right: attached,
mid-crossing, attached to the opposite wall.

Traces come from do_quicklook.mp_series (axis-cache-aware); fields are
reconstructed from the run's gridded archive (output/out_grid.h5), which
stores mean, modes and coefficients on a regular grid at every archive
instant, so the reconstruction is a plain weighted sum. Example:

python3 py_utils/sudden_exp_postproc/fig_transition_reconstructed.py \
  --run cases/se/runs/2026_07_17_do_total_sig0p0144_t1000 \
  --out ~/todisimo/studies/7.0_epfl/year1/academic/candidacy_exam/tex/figures/transition_reconstructed.pdf
"""
import argparse
from pathlib import Path

import h5py
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
ap.add_argument("--settle", type=float, default=60.0,
                help="Window (t.u.) the realisation must spend attached "
                     "on each side of the chosen crossing.")
ap.add_argument("--xmax", type=float, default=25.0)
ap.add_argument("--xmin", type=float, default=-3.0)
ap.add_argument("--pct", type=float, default=99.5)
ap.add_argument("--nshow", type=int, default=4,
                help="Curves in panel (a), the reconstructed one included.")
args = ap.parse_args()

# ---------------- traces ----------------
t, yi, Mp, _ = mp_series(args.run)
trans = transitions_per_particle(t, Mp, args.mbar, args.c, args.tmin)

# Exemplar: the crossing with the most settled plateaus on both sides,
# away from the run ends.
best = None
for p, tr in enumerate(trans):
    for ttr in tr:
        if not (args.tmin + 2 * args.settle < ttr < t[-1] - 2 * args.settle):
            continue
        wb = (t > ttr - 1.5 * args.settle) & (t < ttr - 0.5 * args.settle)
        wa = (t > ttr + 0.5 * args.settle) & (t < ttr + 1.5 * args.settle)
        mb, ma = Mp[wb, p].mean(), Mp[wa, p].mean()
        if mb * ma >= 0:
            continue
        score = min(abs(mb), abs(ma))
        if best is None or score > best[0]:
            best = (score, p, ttr, mb, ma)
assert best is not None, "no settled crossing found"
_, p, ttr, mb, ma = best

# Crossing instant = zero crossing of M_p nearest the detected transition;
# the attached instants sit one settle-window to each side.
near = np.abs(t - ttr) < args.settle
k2 = np.where(near)[0][np.argmin(np.abs(Mp[near, p]))]
k1 = np.argmin(np.abs(t - (t[k2] - args.settle)))
k3 = np.argmin(np.abs(t - (t[k2] + args.settle)))
kk = [k1, k2, k3]

# Companions for panel (a): the most-crossing realisations, as in the
# retired fig_do_trajectories.py, with the exemplar drawn in blue on top.
order = [int(q) for q in np.argsort([-len(tr) for tr in trans]) if q != p]
show = [p] + order[: args.nshow - 1]

# ---------------- fields ----------------
with h5py.File(args.run / "output/out_grid.h5", "r") as f:
    tg = np.asarray(f["t"])
    x = np.asarray(f["x"])
    y = np.asarray(f["y"])
    fields = []
    for k in kk:
        kg = int(np.argmin(np.abs(tg - t[k])))
        u = np.asarray(f["u"][:, :, kg])
        for i in range(f["mode_u"].shape[0]):
            u += f["yi"][p, i, kg] * f["mode_u"][i, :, :, kg]
        fields.append(u)

outside = (x < 0.0) & (np.abs(y) > 0.5)
fields = [np.ma.masked_where(outside, u) for u in fields]

allu = np.concatenate([u.compressed() for u in fields])
lim = np.nanpercentile(np.abs(allu), args.pct)
nlim = max(1e-6, np.nanpercentile(np.abs(allu[allu < 0]), args.pct))
f0 = nlim / (nlim + lim)
n0 = max(2, int(round(2048 * f0)))
cols = np.vstack([plt.cm.RdBu_r(np.linspace(0.0, 0.5, n0)),
                  plt.cm.RdBu_r(np.linspace(0.5, 1.0, 2048 - n0))])
cmap = plt.matplotlib.colors.ListedColormap(cols)
cmap.set_bad("0.85")
norm = plt.matplotlib.colors.Normalize(vmin=-nlim, vmax=lim)

# ---------------- figure ----------------
fig = plt.figure(figsize=(6.3, 2.7), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05])
gsl = gs[0, 0].subgridspec(3, 1, height_ratios=[1, 4, 1])
axa = fig.add_subplot(gsl[1])
gsr = gs[0, 1].subgridspec(3, 1)
axf = [fig.add_subplot(gsr[r]) for r in range(3)]

# (a) transitioning realisations, the reconstructed one in blue on top
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
for q, col in list(zip(show, COLORS))[::-1]:
    axa.plot(t, Mp[:, q], lw=0.7, color=col, zorder=3 if q == p else 2)
for k, lab in zip(kk, ["(b)", "(c)", "(d)"]):
    axa.plot(t[k], Mp[k, p], "o", ms=3.5, color="#0072B2",
             mec="white", mew=0.5, zorder=4)
    if lab == "(c)":
        off, ha, va = (6, 0), "left", "center"
    else:
        off, ha, va = (0, 7), "center", "baseline"
    axa.annotate(lab, (t[k], Mp[k, p]), xytext=off,
                 textcoords="offset points", ha=ha, va=va, fontsize=8,
                 zorder=5)
axa.set_xlim(t[0], t[-1])
axa.set_ylim(-0.5, 0.5)
axa.set_xlabel("$t$")
axa.set_ylabel("$M_p$", fontsize=11)
axa.text(-0.075, 1.0, "(a)", transform=axa.transAxes, va="top", ha="right",
         fontsize=11)

# (b)-(d) reconstructed fields with streamfunction contours
for ax, u, k, lab in zip(axf, fields, kk, ["(b)", "(c)", "(d)"]):
    im = ax.imshow(u, origin="lower",
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   cmap=cmap, norm=norm,
                   aspect="auto", interpolation="bilinear")
    dy = y[1, 0] - y[0, 0]
    uf = np.ma.filled(u, 0.0)
    psi = np.zeros_like(uf)
    psi[1:, :] = np.cumsum(0.5 * (uf[1:, :] + uf[:-1, :]) * dy, axis=0)
    q = np.median(psi[-1, :])
    lev_main = np.linspace(0.0, q, 11)[1:-1]
    lev_neg = np.linspace(psi.min(), 0.0, 6)[1:-1]
    lev_pos = np.linspace(q, psi.max(), 6)[1:-1]
    levels = np.unique(np.concatenate([lev_neg, lev_main, lev_pos]))
    ax.contour(x, y, np.ma.masked_where(u.mask, psi), levels=levels,
               colors="0.35", linewidths=0.45, linestyles="solid")
    ax.set_facecolor("0.85")
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(y.min(), y.max())
    ax.set_yticks([-1.5, 0, 1.5])
    ax.set_ylabel("$y$", fontsize=11)
    ax.text(0.995, 0.92, f"{lab} $t={t[k]:.0f}$", transform=ax.transAxes,
            va="top", ha="right", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                      pad=1.5))
for ax in axf[:-1]:
    ax.set_xticklabels([])
axf[-1].set_xlabel("$x$", fontsize=11)
cb = fig.colorbar(im, ax=axf, shrink=0.92, pad=0.015)
cb.set_label("$u$", fontsize=11)

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out)
png = args.out.with_suffix(".png")
fig.savefig(png, dpi=200)
print(f"Saved: {args.out} and {png}")
print(f"realisation {p}; crossing at t={t[k2]:.1f}; instants "
      + ", ".join(f"t={t[k]:.0f} (M={Mp[k, p]:+.3f})" for k in kk))
print("panel (a) curves:", ", ".join(f"{q} ({len(trans[q])} crossings)"
                                     for q in show))
