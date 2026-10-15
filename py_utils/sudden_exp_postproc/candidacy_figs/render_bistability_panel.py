#!/usr/bin/env python3
"""Animated bistability panel of a sudden-expansion DNS for the exam slides.

Three stacked rows, one video:
  top    -- instantaneous streamwise velocity u(x, t)
  middle -- running time average of u from t = 0 to the current t
  bottom -- asymmetry signal M(t) drawn up to the current t (left) and the
            probability density of M over [0, t] populating itself (right)

Inputs are per-checkpoint plane interpolations (FieldConvert interppoints,
csv columns x y u v p on a structured ny x nx grid, one file per checkpoint,
gd_<STEP>.csv) plus the run's M_of_t.npy. The parsed fields are cached to an
.h5 next to the output so re-renders skip the csv pass.

Example:
python3 py_utils/sudden_exp_postproc/candidacy_figs/render_bistability_panel.py \
  --csv-dir $SCRATCH/csv --m-of-t RUN/M_of_t.npy \
  --chk-dt 4.0 --out se_bistability_panel.mp4
"""
import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

ap = argparse.ArgumentParser()
ap.add_argument("--csv-dir", type=Path, required=True)
ap.add_argument("--m-of-t", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--chk-dt", type=float, default=4.0,
                help="Time units per checkpoint index (chk stride x TimeStep).")
ap.add_argument("--nx", type=int, default=901)
ap.add_argument("--ny", type=int, default=121)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--dpi", type=int, default=150)
ap.add_argument("--xmax", type=float, default=15.0)
ap.add_argument("--xmin", type=float, default=-3.0)
ap.add_argument("--pct", type=float, default=99.5,
                help="Percentile of |u| setting the color limits.")
ap.add_argument("--max-frames", type=int, default=None,
                help="Limit frames for quick previews.")
ap.add_argument("--font-size", type=float, default=9.0,
                help="Base font size in pt; figure margins scale with it.")
ap.add_argument("--colorbar", action=argparse.BooleanOptionalAction,
                default=True,
                help="Thin colorbar to the right of the u and mean-u rows.")
ap.add_argument("--poster", type=Path, default=None,
                help="Also save frame 0 as a PNG (same dpi) for the slide.")
ap.add_argument("--no-video", action="store_true",
                help="Skip the mp4 encode (layout checks with --poster).")
args = ap.parse_args()
plt.rcParams["font.size"] = args.font_size
# The 9 pt values below reproduce the original layout; margins and row
# spacing grow with the label size so nothing clips at the figure edges.
lfs = max(0.0, args.font_size - 9.0)


def load_csv_fields():
    files = sorted(args.csv_dir.glob("gd_*.csv"),
                   key=lambda p: int(re.search(r"gd_(\d+)", p.name).group(1)))
    steps = [int(re.search(r"gd_(\d+)", f.name).group(1)) for f in files]
    ny, nx = args.ny, args.nx
    u = np.empty((len(files), ny, nx))
    x = y = None
    for k, f in enumerate(files):
        d = pd.read_csv(f, header=0).to_numpy()
        if x is None:
            x = d[:, 0].reshape(ny, nx)
            y = d[:, 1].reshape(ny, nx)
        u[k] = d[:, 2].reshape(ny, nx)
    t = np.asarray(steps, dtype=float) * args.chk_dt
    return x, y, u, t


cache = args.out.with_suffix(".fields.h5")
if cache.exists():
    with h5py.File(cache, "r") as f:
        x, y, u, t = f["x"][:], f["y"][:], f["u"][:], f["t"][:]
else:
    x, y, u, t = load_csv_fields()
    with h5py.File(cache, "w") as f:
        for k, v in [("x", x), ("y", y), ("u", u), ("t", t)]:
            f.create_dataset(k, data=v, compression="gzip", compression_opts=2)

if args.max_frames is not None:
    u, t = u[: args.max_frames], t[: args.max_frames]

mt = np.load(args.m_of_t)
tm, M = mt[:, 0], mt[:, 1]

# Solid region of the sudden-expansion geometry: the two blocks upstream of
# the step (x<0, |y|>h) with inlet half-height h=0.5. Masking on the geometry
# directly avoids painting wall nodes (u=v=0) as solid.
outside = (x < 0.0) & (np.abs(y) > 0.5)
umask = np.broadcast_to(outside, u.shape)
u = np.ma.masked_where(umask, u)
umean = np.ma.masked_where(umask, np.cumsum(u.filled(0.0), axis=0)
                           / np.arange(1, u.shape[0] + 1)[:, None, None])

# Backflow is an order of magnitude weaker than the jet. Remap the colormap
# so white sits at u=0 under a plain linear norm: the full blue range is
# spent on the actual u<0 values (same construction as fig_se_states.py).
sample = u[:: max(1, u.shape[0] // 50)].compressed()
lim = np.nanpercentile(np.abs(sample), args.pct)
nlim = max(1e-6, np.nanpercentile(np.abs(sample[sample < 0]), args.pct))
f0 = nlim / (nlim + lim)
n0 = max(2, int(round(2048 * f0)))
cols = np.vstack([plt.cm.RdBu_r(np.linspace(0.0, 0.5, n0)),
                  plt.cm.RdBu_r(np.linspace(0.5, 1.0, 2048 - n0))])
cmap = plt.matplotlib.colors.ListedColormap(cols)
cmap.set_bad("0.85")

mlim = 1.05 * np.max(np.abs(M))
bins = np.linspace(-mlim, mlim, 61)
dens_full, _ = np.histogram(M, bins=bins, density=True)
dlim = 1.1 * dens_full.max()

fig = plt.figure(figsize=(12.8, 7.2))
gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.05, 1.0],
                      width_ratios=[4.2, 1.0],
                      hspace=0.28 + 0.0130 * lfs, wspace=0.06 + 0.0045 * lfs,
                      left=0.055 + 0.0022 * lfs, right=0.985,
                      top=0.96, bottom=0.09 + 0.0022 * lfs)
ax_top = fig.add_subplot(gs[0, :])
ax_mid = fig.add_subplot(gs[1, :], sharex=ax_top)
ax_sig = fig.add_subplot(gs[2, 0])
ax_pdf = fig.add_subplot(gs[2, 1], sharey=ax_sig)

extent = [x[0, 0], x[0, -1], y[0, 0], y[-1, 0]]
ims = []
for ax, field, tag in [(ax_top, u[0], r"$u(\mathbf{x},t)$"),
                       (ax_mid, umean[0], r"$\overline{u}^{\,[0,t]}$")]:
    im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap,
                   vmin=-nlim, vmax=lim, aspect="equal",
                   interpolation="bilinear")
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylabel(r"$y$")
    ax.text(0.008, 0.93, tag, transform=ax.transAxes, ha="left", va="top")
    ims.append(im)
ax_top.tick_params(labelbottom=False)
ax_mid.set_xlabel(r"$x$")
if args.colorbar:
    # Inset to the right of each field: the axes box shrinks to the image
    # under aspect="equal", and inset_axes follows that box at draw time,
    # so the bar always sits flush with the image edge. Three ticks: the
    # colour-limit ends and the u=0 white.
    for ax, im in zip((ax_top, ax_mid), ims):
        cax = ax.inset_axes([1.012, 0.04, 0.011, 0.92])
        cb = fig.colorbar(im, cax=cax, ticks=[-nlim, 0.0, lim])
        cb.ax.set_yticklabels([f"${-nlim:.2g}$", "$0$", f"${lim:.2g}$"])
        cb.ax.tick_params(labelsize=0.6 * args.font_size, length=2.0,
                          width=0.6, pad=1.5)
        # u=0 sits ~10% above the bottom end: hang the end label below its
        # tick and seat the 0 label above its own so the two do not touch.
        for lab, va in zip(cb.ax.get_yticklabels(), ("top", "bottom", "center")):
            lab.set_va(va)
        cb.outline.set_linewidth(0.5)
clock = ax_top.text(0.992, 0.93, "", transform=ax_top.transAxes,
                    ha="right", va="top", color="0.45")

ax_sig.axhline(0.0, color="0.45", lw=0.6)
(line,) = ax_sig.plot([], [], color="#0072B2", lw=0.8)
(dot,) = ax_sig.plot([], [], "o", color="#0072B2", ms=3.5)
ax_sig.set_xlim(0.0, tm[-1])
ax_sig.set_ylim(-mlim, mlim)
ax_sig.set_xlabel(r"$t$")
ax_sig.set_ylabel(r"$M$")

pdf_poly = [None]
ax_pdf.set_xlim(0.0, dlim)
ax_pdf.set_xlabel(r"$p(M)$")
ax_pdf.tick_params(labelleft=False)


def draw(k):
    ims[0].set_data(u[k])
    ims[1].set_data(umean[k])
    clock.set_text(rf"$t = {t[k]:.0f}$")
    it = int(np.searchsorted(tm, t[k], side="right"))
    line.set_data(tm[:it], M[:it])
    dot.set_data(tm[it - 1 : it], M[it - 1 : it])
    dens, _ = np.histogram(M[:it], bins=bins, density=True)
    if pdf_poly[0] is not None:
        pdf_poly[0].remove()
    pdf_poly[0] = ax_pdf.fill_betweenx(
        0.5 * (bins[:-1] + bins[1:]), 0.0, dens,
        step="mid", color="#0072B2", alpha=0.85, lw=0)
    return ims + [clock, line, dot, pdf_poly[0]]


if args.poster is not None:
    draw(0)
    fig.savefig(args.poster, dpi=args.dpi)
    print(f"wrote {args.poster}")
if args.no_video:
    raise SystemExit(0)

anim = animation.FuncAnimation(fig, draw, frames=u.shape[0], blit=False)
writer = animation.FFMpegWriter(
    fps=args.fps, extra_args=["-crf", "18", "-pix_fmt", "yuv420p"])
anim.save(args.out, writer=writer, dpi=args.dpi)
print(f"wrote {args.out}: {u.shape[0]} frames at {args.fps} fps "
      f"({u.shape[0] / args.fps:.1f} s), t = {t[0]:.0f}..{t[-1]:.0f}")
