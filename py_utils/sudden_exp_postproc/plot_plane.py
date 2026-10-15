#!/usr/bin/env python3
r"""Plot fields stored on the 901 x 121 extraction plane of se.py / extract_lib.py
(x in [-5, 40], y in [-1.5, 1.5]): one panel per key of an
.npz written by that module.

  make-phi          phi_u, phi_v                    (phi_Re<Re>.npz)
  reduce-dns        u, v                            (one checkpoint)
  reduce-do --keep-grid   mean_u, mean_v, mode_u, mode_v   (mode_* are (S, NY, NX): pick one with --mode K)

Style copied from candidacy_figs/fig_se_states.py: imshow on the plane, the
inlet walls (x < 0, |y| > 0.5) in grey, one colour scale shared by every
panel, white at zero (the blue range is spent on the actual negative values,
so a field of one sign still shows its weak backflow). Writes the figure and
a .cmd file with the command line next to it (figsave.py).

Window: every field figure under cases/se/figures/ shows the same x range,
XMIN..XMAX = -5..15 (the whole inlet channel and the region where phi, the
adjoint mode and the modes live); --full shows the whole domain -5..40.

Examples:
  python3 plot_plane.py cases/se/templates/phi_Re100.npz --title 'Re 100' \
      --out cases/se/figures/Re100/phi.pdf
  python3 plot_plane.py run/output/extract/do_00100.npz --keys mode_u mode_v --mode 0 \
      --out cases/se/figures/Re100_sig0.0144/RUN/mode0.pdf
  python3 plot_plane.py cases/se/templates/qdagger_Re79p3.npz --keys u v \
      --labels '$q^\dagger_u$' '$q^\dagger_v$' --title 'Re 79.3' --out cases/se/figures/Re79p3/qdagger.pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figsave import save  # noqa: E402
from se import INSIDE, x, y  # noqa: E402

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

SYMBOL = {"phi_u": r"$\phi_u$", "phi_v": r"$\phi_v$", "u": "$u$", "v": "$v$",
          "mean_u": r"$\bar u$", "mean_v": r"$\bar v$", "mode_u": "$u_{{{k}}}$", "mode_v": "$v_{{{k}}}$"}
XMIN, XMAX = -5.0, 15.0          # the window shared by every field figure

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("npz", type=Path)
ap.add_argument("--out", type=Path, required=True, help="figure path; .pdf also writes the .png")
ap.add_argument("--keys", nargs="+", help="npz keys, one panel each (default: phi_u phi_v, else u v, else mean_u mean_v)")
ap.add_argument("--mode", type=int, default=0, help="index K for 3-D keys such as mode_u")
ap.add_argument("--labels", nargs="+", help="panel labels (LaTeX), one per key; default from the key names")
ap.add_argument("--title", default="", help="text appended to the panel labels, e.g. 'Re 100'")
ap.add_argument("--full", action="store_true", help="whole domain, x in [-5, 40], instead of the shared window")
ap.add_argument("--pct", type=float, default=99.5, help="percentile of |field| setting the colour limits")
args = ap.parse_args()

z = np.load(args.npz)
keys = args.keys
if args.labels and keys and len(args.labels) != len(keys):
    sys.exit("--labels needs one label per key")
if keys is None:
    for cand in (["phi_u", "phi_v"], ["u", "v"], ["mean_u", "mean_v"]):
        if all(k in z for k in cand):
            keys = cand
            break
    else:
        sys.exit(f"{args.npz}: none of phi_u/phi_v, u/v, mean_u/mean_v found; keys: {list(z.keys())}")

fields = []
for k in keys:
    f = np.asarray(z[k], dtype=float)
    if f.ndim == 3:
        f = f[args.mode]
    if f.shape != INSIDE.shape:
        sys.exit(f"{k}: shape {f.shape}, expected {INSIDE.shape}")
    fields.append(np.ma.masked_where(~INSIDE, f))

allv = np.concatenate([f.compressed() for f in fields])
lim = max(1e-12, np.nanpercentile(np.abs(allv), args.pct))
neg = allv[allv < 0]
nlim = max(1e-6, np.nanpercentile(np.abs(neg), args.pct)) if neg.size else 1e-6
nlim = min(nlim, lim)
# Remap the colormap so white sits at zero under a plain linear norm (fig_se_states.py).
f0 = nlim / (nlim + lim)
n0 = max(2, int(round(2048 * f0)))
cols = np.vstack([plt.cm.RdBu_r(np.linspace(0.0, 0.5, n0)),
                  plt.cm.RdBu_r(np.linspace(0.5, 1.0, 2048 - n0))])
cmap = matplotlib.colors.ListedColormap(cols)
cmap.set_bad("0.85")
norm = matplotlib.colors.Normalize(vmin=-nlim, vmax=lim)

fig, axes = plt.subplots(len(fields), 1, figsize=(6.3, 1.15 * len(fields) + 0.6), sharex=True,
                         constrained_layout=True, squeeze=False)
axes = axes[:, 0]
for i, (ax, f, k) in enumerate(zip(axes, fields, keys)):
    im = ax.imshow(f, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                   cmap=cmap, norm=norm, aspect="auto", interpolation="bilinear")
    ax.set_facecolor("0.85")
    ax.set_xlim(XMIN, x.max() if args.full else XMAX)
    ax.set_ylim(y.min(), y.max())
    ax.set_yticks([-1.5, 0, 1.5])
    ax.set_ylabel("$y$", fontsize=13)
    sym = args.labels[i] if args.labels else SYMBOL.get(k, k.replace('_', r'\_')).format(k=args.mode)
    lab = f"({chr(97 + i)}) " + sym
    if args.title:
        lab += f", {args.title}"
    ax.text(0.995, 0.92, lab, transform=ax.transAxes, va="top", ha="right", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
axes[-1].set_xlabel("$x$", fontsize=13)
fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015)

out = save(fig, args.out, dpi=200, also_png=True)
print(f"Saved: {out}" + (f" and {out.with_suffix('.png')}" if out.suffix != ".png" else ""))
