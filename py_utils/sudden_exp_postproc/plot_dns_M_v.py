#!/usr/bin/env python3
"""Plot the transition diagnostics of a run: M(t) and the centerline sensor.

Two modes, auto-detected per run directory:

1. Single-realization runs (DNS mode, kicked/deterministic VCS): reads
   output/asymmetry/axis_v.his, plots M(t) (Ducimetiere Eq. 25) with the
   hysteresis-counted transitions and the raw sensor v(x=2, y=0, t).

2. DO runs (detected by output/do/ archives): the .his only holds the
   ensemble MEAN, whose M stays near 0 for a balanced population, so
   instead the script reconstructs the PER-PARTICLE measure
       M_p(t) = M[ubar + sum_i Y_pi u_i]
   from the DO archives: it runs FieldConvert line-interpolations of every
   archive onto the symmetry axis (cached in output/py_utils_results/
   axis_cache/), reads Yi from the archive metadata, counts transitions
   per particle (hysteresis, init transient excluded via --tmin), and
   plots sample trajectories (hoppers highlighted) over the ensemble
   M-histogram. The readers and the transition rule are those of se.py
   (read_his, axis_M, mp_series, crossings).

Usage:
    python3 plot_dns_M_v.py RUN_DIR [RUN_DIR ...] [--out fig.png]
        [--mbar 0.1935] [--c 0.8] [--xsensor 2.0] [--tmin 30]
        [--fieldconvert PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import se


# --------------------------------------------------------------------------
# single-realization panel pair (DNS / deterministic)
# --------------------------------------------------------------------------
def plot_single(run: Path, axM, axV, args):
    t, x, f = se.read_his(run / "output/asymmetry/axis_v.his")
    v = f[:, :, 1]
    isen = int(np.argmin(np.abs(x - args.xsensor)))
    M = se.axis_M(x, v, args.xsensor)
    tr = se.crossings(t, M, args.c * args.mbar)
    thr = args.c * args.mbar
    axM.plot(t, M, lw=0.6, color="0.25")
    for y, ccol, ls in [(args.mbar, "tab:blue", ":"),
                        (-args.mbar, "tab:blue", ":"),
                        (thr, "tab:red", "--"), (-thr, "tab:red", "--")]:
        axM.axhline(y, color=ccol, ls=ls, lw=0.8)
    axV.plot(t, v[:, isen], lw=0.5, color="tab:green")
    axV.axhline(0, color="k", lw=0.5)
    for tr_ in tr:
        axM.axvline(tr_, color="tab:orange", alpha=0.7, lw=1.1)
        axV.axvline(tr_, color="tab:orange", alpha=0.4, lw=1.0)
    axM.set_ylabel("M(t)")
    axV.set_ylabel(f"v(x={args.xsensor:g},y=0)")
    axM.set_title(f"{run.name}: {len(tr)} transitions", fontsize=10,
                  loc="left")


# --------------------------------------------------------------------------
# DO run: per-particle reconstruction
# --------------------------------------------------------------------------
def plot_do(run: Path, axTraj, axHist, args):
    t, yi, Mp, _ = se.mp_series(run, args.xsensor, args.fieldconvert)
    S = yi.shape[2]
    P = Mp.shape[1]
    thr = args.c * args.mbar

    # transition count on the steady window
    w = t >= args.tmin
    trans = se.crossings(t[w], Mp[w], thr)
    hops = sum(len(tr) for tr in trans)
    hoppers = [p for p, tr in enumerate(trans) if len(tr)]
    span = t[w][-1] - t[w][0] if w.any() else 0.0
    rate = hops / (P * span) if span > 0 else float("nan")

    # trajectories: a grey sample + every hopper colored
    rng = np.random.default_rng(0)
    sample = rng.choice(P, size=min(50, P), replace=False)
    for p in sample:
        axTraj.plot(t, Mp[:, p], lw=0.4, color="0.75", zorder=1)
    for p in hoppers[:40]:
        axTraj.plot(t, Mp[:, p], lw=0.8, zorder=2)
    for y, ccol, ls in [(args.mbar, "tab:blue", ":"),
                        (-args.mbar, "tab:blue", ":"),
                        (thr, "tab:red", "--"), (-thr, "tab:red", "--")]:
        axTraj.axhline(y, color=ccol, ls=ls, lw=0.8)
    axTraj.axvspan(0, args.tmin, color="0.92", zorder=0)
    axTraj.set_ylabel("M_p(t)")
    axTraj.set_title(
        f"{run.name} S={S}, {P} particles: {hops} transitions by "
        f"{len(hoppers)} particles for t>{args.tmin:g} "
        f"(rate {rate:.2e}/particle-t.u.)", fontsize=10, loc="left")

    h2 = axHist.hist2d(np.repeat(t, P), Mp.ravel(),
                       bins=[len(t), 60], cmap="magma")
    axHist.set_ylabel("M")
    print(f"{run.name}: {hops} transitions by {len(hoppers)} particles "
          f"in t={args.tmin:g}..{t[-1]:.0f}; rate {rate:.2e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("run_M_and_v.png"))
    ap.add_argument("--mbar", type=float, default=0.1935)
    ap.add_argument("--c", type=float, default=0.8)
    ap.add_argument("--xsensor", type=float, default=2.0)
    ap.add_argument("--tmin", type=float, default=30.0,
                    help="DO runs: exclude the init transient before this")
    ap.add_argument("--fieldconvert", default=se.FIELDCONVERT)
    args = ap.parse_args()

    n = len(args.runs)
    fig, axs = plt.subplots(2 * n, 1, figsize=(13, 4.8 * n), sharex=True)
    axs = np.atleast_1d(axs)
    for row, run in enumerate(args.runs):
        is_do = any((run / "output/do").glob("*.do_*.fld")) \
            if (run / "output/do").is_dir() else False
        if is_do:
            plot_do(run, axs[2 * row], axs[2 * row + 1], args)
        else:
            plot_single(run, axs[2 * row], axs[2 * row + 1], args)
    axs[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
