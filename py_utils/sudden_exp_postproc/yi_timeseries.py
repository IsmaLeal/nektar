#!/usr/bin/env python3
"""Plot Y_i vs time for every particle of a DO run.

Each particle is a thin translucent line, coloured by the sign of its
initial Y1 (red for Y1 < 0, blue for Y1 >= 0) so basin transitions stand
out, consistent with phase_space_video.py.

Input is the same as phase_space_video.py: a run directory (the output/*.h5
with the most DO frames is used, falling back to the raw output/do/*.fld
metadata) or an HDF5/NPZ file with `yi` (P, S, T) and `t_do`.

Example:
    python3 yi_timeseries.py cases/se/runs/2026_07_10_do_m02_Re70_bimodal_pod_t500 -i 1
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def _load_phase_space_module():
    mod_path = Path(__file__).resolve().parent / "phase_space_video.py"
    spec = importlib.util.spec_from_file_location("phase_space_video", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load phase_space_video.py from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path,
                    help="Run directory, or HDF5/NPZ file with yi and t_do.")
    ap.add_argument("-i", "--mode", type=int, default=1,
                    help="Mode index i of Y_i to plot, 1-based. Default 1.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Save the figure instead of showing it.")
    ap.add_argument("--alpha", type=float, default=0.08,
                    help="Per-particle line alpha. Default 0.08.")
    ap.add_argument("--linewidth", type=float, default=0.4,
                    help="Per-particle line width. Default 0.4.")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    import matplotlib
    if args.out is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 14})

    yi, t = _load_phase_space_module()._load_run(args.data)
    P, S, _ = yi.shape
    if not (1 <= args.mode <= S):
        raise ValueError(f"mode index {args.mode} out of range [1, {S}]")
    y = yi[:, args.mode - 1, :]                 # (P, T)

    # initial colour by sign of Y1, as in phase_space_video.py
    red = yi[:, 0, 0] < 0.0

    fig, ax = plt.subplots(figsize=(12, 5))
    for mask, colour in ((red, "red"), (~red, "blue")):
        if mask.any():
            # one Line2D per particle; NaN-joining them into a single line
            # would connect trajectories across particles
            ax.plot(t, y[mask].T, color=colour, alpha=args.alpha,
                    lw=args.linewidth)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time")
    ax.set_ylabel(rf"$Y_{{{args.mode}}}$")
    fig.tight_layout()

    if args.out is not None:
        fig.savefig(args.out, dpi=args.dpi)
        print(f"saved {args.out}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
