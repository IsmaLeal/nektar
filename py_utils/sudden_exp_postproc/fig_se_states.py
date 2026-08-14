#!/usr/bin/env python3
"""Three-panel figure of the sudden-expansion states for the research plan:
(a) unstable symmetric base flow at Re=100, (b)/(c) the two mirror-image
stable asymmetric states. Crosswise velocity v on a shared symmetric color
scale, so the mirror antisymmetry of (b) and (c) is visible directly.

Inputs are gridded .h5 files produced by py_utils/load_chk.py (single
snapshot each). Example:

python3 py_utils/sudden_exp_postproc/fig_se_states.py \
  --base $TMP/state_base.h5 --pos $TMP/state_pos.h5 --neg $TMP/state_neg.h5 \
  --out figures/se_states.pdf
"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

ap = argparse.ArgumentParser()
ap.add_argument("--base", type=Path, required=True)
ap.add_argument("--pos", type=Path, required=True)
ap.add_argument("--neg", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--xmax", type=float, default=15.0)
ap.add_argument("--xmin", type=float, default=-3.0)
ap.add_argument("--pct", type=float, default=99.5,
                help="Percentile of |v| over the two asymmetric states "
                     "setting the shared symmetric color limit.")
ap.add_argument("--res", type=float, nargs=3, default=[70.0, 100.0, 100.0],
                help="Reynolds numbers printed in the panel labels "
                     "(base, pos, neg).")
args = ap.parse_args()


def load(path):
    with h5py.File(path, "r") as f:
        x = np.asarray(f["x"])
        y = np.asarray(f["y"])
        u = np.asarray(f["u"])[..., -1]
        v = np.asarray(f["v"])[..., -1]
    # Solid region of the sudden-expansion geometry: the two blocks upstream
    # of the step (x<0, |y|>h) with inlet half-height h=0.5. Masking on the
    # geometry directly avoids painting wall nodes (u=v=0) as solid.
    outside = (x < 0.0) & (np.abs(y) > 0.5)
    return (x, y, np.ma.masked_where(outside, u),
            np.ma.masked_where(outside, v))


def observable(x, y, vfield):
    iy = np.argmin(np.abs(y[:, 0]))
    xs = x[0, :]
    vline = np.ma.filled(vfield[iy, :], 0.0)
    m = xs >= 0
    ix2 = np.argmin(np.abs(xs - 2.0))
    return np.sign(vline[ix2]) * np.sqrt(np.trapz(vline[m] ** 2, xs[m]))


states = [load(args.base), load(args.pos), load(args.neg)]
Ms = [observable(x, y, v) for (x, y, u, v) in states]
allu = np.concatenate([s[2].compressed() for s in states[1:]])
lim = np.nanpercentile(np.abs(allu), args.pct)
# Backflow is an order of magnitude weaker than the jet. Rather than a
# two-slope norm (whose colorbar stretches the blue segment and makes the
# ticks uneven), remap the colormap itself so white sits at u=0 under a
# plain linear norm: the bar stays linear with evenly spaced ticks, while
# the full blue range is spent on the actual u<0 values.
nlim = max(1e-6, np.nanpercentile(np.abs(allu[allu < 0]), args.pct))
f0 = nlim / (nlim + lim)
n0 = max(2, int(round(2048 * f0)))
cols = np.vstack([plt.cm.RdBu_r(np.linspace(0.0, 0.5, n0)),
                  plt.cm.RdBu_r(np.linspace(0.5, 1.0, 2048 - n0))])
cmap = plt.matplotlib.colors.ListedColormap(cols)
cmap.set_bad("0.85")
norm = plt.matplotlib.colors.Normalize(vmin=-nlim, vmax=lim)

labels = ["(a) $\\mathrm{{Re}}={:g}$".format(args.res[0]),
          "(b) $\\mathrm{{Re}}={:g}$".format(args.res[1]),
          "(c) $\\mathrm{{Re}}={:g}$".format(args.res[2])]

fig, axes = plt.subplots(3, 1, figsize=(6.3, 3.4), sharex=True,
                         constrained_layout=True)
for ax, (x, y, u, v), lab in zip(axes, states, labels):
    im = ax.imshow(u, origin="lower",
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   cmap=cmap, norm=norm,
                   aspect="auto", interpolation="bilinear")
    # Streamlines as level sets of the streamfunction psi (u = dpsi/dy),
    # integrated upward from the bottom boundary. Level curves are exact,
    # continuous, closed streamlines -- streamplot's seed-based integration
    # overdraws closed orbits in the recirculation bubbles.
    dy = y[1, 0] - y[0, 0]
    uf = np.ma.filled(u, 0.0)
    psi = np.zeros_like(uf)
    psi[1:, :] = np.cumsum(0.5 * (uf[1:, :] + uf[:-1, :]) * dy, axis=0)
    q = np.median(psi[-1, :])                    # top-wall streamline = flux
    lev_main = np.linspace(0.0, q, 11)[1:-1]     # through-flow
    lev_neg = np.linspace(psi.min(), 0.0, 6)[1:-1]   # lower bubbles
    lev_pos = np.linspace(q, psi.max(), 6)[1:-1]     # upper bubbles
    levels = np.unique(np.concatenate([lev_neg, lev_main, lev_pos]))
    ax.contour(x, y, np.ma.masked_where(u.mask, psi), levels=levels,
               colors="0.35", linewidths=0.45, linestyles="solid")
    ax.set_facecolor("0.85")
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(y.min(), y.max())
    ax.set_yticks([-1.5, 0, 1.5])
    ax.set_ylabel("$y$", fontsize=13)
    ax.text(0.995, 0.92, lab, transform=ax.transAxes, va="top", ha="right",
            fontsize=11, bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, pad=1.5))
axes[-1].set_xlabel("$x$", fontsize=13)
cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015)
cb.set_label("$u$", fontsize=13)
cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8])

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out)
png = args.out.with_suffix(".png")
fig.savefig(png, dpi=200)
print(f"Saved: {args.out} and {png}")
print("M values (a,b,c):", ", ".join(f"{m:+.4f}" for m in Ms))
