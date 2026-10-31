"""
One DO mode on the plane (panel.py --panel modes): u_k or v_k of one run at one of the instants
where measures.h5 keeps the modes (grid/t, every 1000 t.u.), one colour scale per mode.
field(ax, f, label) draws any field on the plane (inlet walls grey, white at zero, x in [-5, 15]):
plot_plane.py and svd.py draw with it too.

    k=K        mode, 1 = the first (default 1)
    comp=u|v   component (default u)
    t=T        instant, the stored one nearest to T (default the last)

Run as a file for every mode of one run, one panel each (it calls panel.py, which writes the .cmd):

    python3 draw_modes.py S6_Np1000_seed45 [--pair Re100_sig0.0144] [--comp u] [--t T] [--modes N] \
        --out cases/se/figures/Re100_sig0.0144/do_Re100_sig0.0144_S6_Np1000_seed45/modes_u.pdf
"""
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import subprocess
import sys
from pathlib import Path

import se

RC = {"text.usetex": True, "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9}
SIZE, DPI = (6.3, 1.15), 200
XMIN, XMAX = -5.0, 15.0          # the window shared by every field figure (plot_plane.py)
PCT = 99.5                       # percentile of |field| setting the colour limits


def draw(ax, curves, k=1, comp="u", t=None):
    """curves: one DO run, [([run directory], None)] (se.find)."""
    assert len(curves) == 1 and len(curves[0][0]) == 1, "modes: one DO run per panel"
    run = curves[0][0][0]
    k = int(k)
    with h5py.File(se.planes(run), "r") as h:
        tg = h["grid/t"][:]
        kt = len(tg) - 1 if t is None else int(np.argmin(np.abs(tg - float(t))))
        f = h[f"grid/mode_{comp}"][kt, k - 1].astype(float)
    im = field(ax, f, f"${comp}_{{{k}}}$")
    ax.set_xlabel("$x$", fontsize=13)
    ax.set_title(se.label(run) + f", $t = {tg[kt]:g}$", loc="right", fontsize=9)
    ax.figure.colorbar(im, ax=ax, pad=0.015)


def field(ax, f, label, scale=None, pct=PCT, xmax=XMAX):
    """f (NY, NX) on the plane; scale: the values the colour limits come from, default f inside the channel."""
    f = np.ma.masked_where(~se.INSIDE, f)
    v = f.compressed() if scale is None else scale
    lim = max(1e-12, np.nanpercentile(np.abs(v), pct))
    neg = v[v < 0]
    nlim = min(lim, max(1e-6, np.nanpercentile(np.abs(neg), pct)) if neg.size else 1e-6)
    n0 = max(2, int(round(2048 * nlim / (nlim + lim))))       # white at zero
    cmap = matplotlib.colors.ListedColormap(np.vstack([plt.cm.RdBu_r(np.linspace(0.0, 0.5, n0)),
                                                       plt.cm.RdBu_r(np.linspace(0.5, 1.0, 2048 - n0))]))
    cmap.set_bad("0.85")
    im = ax.imshow(f, origin="lower", extent=[se.x.min(), se.x.max(), se.y.min(), se.y.max()], cmap=cmap,
                   norm=matplotlib.colors.Normalize(-nlim, lim), aspect="auto", interpolation="bilinear")
    ax.set_facecolor("0.85")
    ax.set_xlim(XMIN, xmax)
    ax.set_ylim(se.y.min(), se.y.max())
    ax.set_yticks([-1.5, 0, 1.5])
    ax.set_ylabel("$y$", fontsize=13)
    ax.text(0.995, 0.92, label, transform=ax.transAxes, va="top", ha="right", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    return im


if __name__ == "__main__":                 # every mode of one run, one panel each, through panel.py
    ap = argparse.ArgumentParser(description="every mode of one DO run, one panel each (panel.py --panel modes)")
    ap.add_argument("run", help="a DO run as panel.py names it, e.g. S6_Np1000_seed45")
    ap.add_argument("--pair", default="Re100_sig0.0144")
    ap.add_argument("--comp", default="u", choices=("u", "v"))
    ap.add_argument("--t", help="instant, the stored one nearest to T; default the last")
    ap.add_argument("--modes", type=int, help="the first N modes; default every mode of the run")
    ap.add_argument("--out", required=True, help="figure path; .pdf also writes the .png")
    a = ap.parse_args()
    runs, _ = se.find(se.pair_dir(a.pair), a.run)
    assert len(runs) == 1, f"{a.run}: one DO run, got {[r.name for r in runs]}"
    with h5py.File(se.planes(runs[0]), "r") as h:
        S = h["grid/mode_u"].shape[1]
    S = min(S, a.modes) if a.modes else S
    opts = [f"comp={a.comp}"] + ([f"t={a.t}"] if a.t else [])
    cmd = [sys.executable, str(Path(__file__).with_name("panel.py")), "--pair", a.pair, "--sharex"]
    for k in range(1, S + 1):
        cmd += ["--panel", "modes", f"k={k}", *opts] + ([a.run] if k == 1 else [])
    sys.exit(subprocess.call(cmd + ["--out", a.out]))
