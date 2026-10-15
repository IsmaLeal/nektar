#!/usr/bin/env python3
"""Candidacy-talk version of fig_mowlavi_comparison.py (validation frame).

The report composites stack the paper's panels above the DOVCS ones inside
one figure, which the slide then scales to a third of its size: titles and
tick labels become unreadable. Here the four blocks are written as separate
files, each drawn at the size it occupies on the slide (1:1, no scaling in
LaTeX), so the fonts are true slide points:

  mowlavi_fields_paper.png   Mowlavi & Sapsis (2018) Fig. 10 crop (300 dpi)
  mowlavi_fields_dovcs.pdf   DOVCS mean / modes / realization, same layout
  mowlavi_coeffs_paper.png   their Fig. 11 crop (300 dpi)
  mowlavi_coeffs_dovcs.pdf   DOVCS coefficient marginals

Row headers ("published" / "DOVCS", t = 2.5) are typeset by the slide itself. Data,
crops, colour limits and the realization index are imported unchanged from
fig_mowlavi_comparison.py.

python3 py_utils/sudden_exp_postproc/candidacy_figs/fig_mowlavi_comparison_slide.py \
  --out ~/todisimo/studies/7.0_epfl/year1/academic/candidacy_exam/slides/figures
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # shared readers live one level up

from fig_mowlavi_comparison import PARTICLE, load_run, paper_crops, vort

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 7,
})

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path,
                default=Path("~/todisimo/studies/7.0_epfl/year1/academic/"
                             "candidacy_exam/slides/figures").expanduser())
ap.add_argument("--height", type=float, default=78 / 72,
                help="Height of each block on the slide (in); the slide "
                     "includes the files at this height.")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

TITLE, TICK, CBTICK, LABEL = 7, 5.5, 5, 7   # pt, as displayed

fig10_img, fig11_img = paper_crops()
fig10_img.save(args.out / "mowlavi_fields_paper.png")
fig11_img.save(args.out / "mowlavi_coeffs_paper.png")

x, y, u, v, mu, mv, yi = load_run()
dx = np.mean(np.diff(x[0, :]))
dy = np.mean(np.diff(y[:, 0]))
ur = u + np.tensordot(yi[PARTICLE], mu, axes=1)
vr = v + np.tensordot(yi[PARTICLE], mv, axes=1)

# ---------------- fields: 2 x 3 panels, paper arrangement ----------------
# colour limits as in fig_mowlavi_comparison.py (Mowlavi & Sapsis Fig. 10,
# panel by panel); colorbar ticks thinned to three so they stay legible
panels = [
    (u, v, "Mean", (-0.75, 2.75), [-0.5, 1, 2.5]),
    (mu[0], mv[0], "Mode 1", (-0.7, 0.7), [-0.6, 0, 0.6]),
    (mu[1], mv[1], "Mode 2", (-1.25, 1.75), [-1, 0, 1.5]),
    (ur, vr, "Realization", (-1, 9), [0, 4, 8]),
    (mu[2], mv[2], "Mode 3", (-2.5, 2.5), [-2, 0, 2]),
    (mu[3], mv[3], "Mode 4", (-2.5, 2.5), [-2, 0, 2]),
]
H = args.height
fig, axs = plt.subplots(2, 3, figsize=(1.95, H), sharex=True, sharey=True,
                        constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.04,
                                hspace=0.04)
s = 10
for ax, (uu, vv, ttl, clim, ticks) in zip(axs.flat, panels):
    w = vort(uu, vv, dx, dy)
    pc = ax.pcolormesh(x, y, w, cmap="jet", shading="gouraud",
                       vmin=clim[0], vmax=clim[1], rasterized=True)
    ax.quiver(x[::s, ::s], y[::s, ::s], uu[::s, ::s], vv[::s, ::s],
              color="white", width=0.008, headwidth=3.5, headlength=4,
              scale_units="width", scale=None)
    ax.set_title(ttl, fontsize=TITLE, pad=2)
    ax.set_aspect("equal")
    ax.set_xticks([0, 3, 6])
    ax.set_yticks([0, 3, 6])
    ax.tick_params(labelsize=TICK, pad=1, length=1.5)
    cb = fig.colorbar(pc, ax=ax, fraction=0.05, pad=0.03)
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    cb.ax.tick_params(labelsize=CBTICK, pad=1, length=1.5)
    cb.outline.set_linewidth(0.4)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
fig.savefig(args.out / "mowlavi_fields_dovcs.pdf", dpi=300)
plt.close(fig)

# ---------------- coefficients: (Y1, Y2) and (Y3, Y4) marginals ----------------
fig, axs = plt.subplots(1, 2, figsize=(1.75, H), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.08)
for ax, (i, j) in zip(axs, [(0, 1), (2, 3)]):
    ax.scatter(yi[:, i], yi[:, j], s=3, alpha=0.5, color="#0072B2",
               linewidths=0, rasterized=True)
    ax.set_xlabel(f"$Y_{i + 1}$", fontsize=LABEL, labelpad=1)
    ax.set_ylabel(f"$Y_{j + 1}$", fontsize=LABEL, labelpad=1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.tick_params(labelsize=TICK, pad=1, length=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
fig.savefig(args.out / "mowlavi_coeffs_dovcs.pdf", dpi=300)
plt.close(fig)

for name in ("mowlavi_fields_paper.png", "mowlavi_coeffs_paper.png"):
    im = fig10_img if "fields" in name else fig11_img
    print(f"{name}: {im.width}x{im.height} px, aspect {im.width / im.height:.3f}")
print("saved to", args.out)
