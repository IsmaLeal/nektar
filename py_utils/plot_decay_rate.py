#!/usr/bin/env python3
"""plot_decay_rate.py -- semilogy of |v(x, 0, t)| for several x values.

Usage:
    python3 plot_decay_rate.py path/to/axis_v.his [--out FILE] [--nx INT]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("his_file", type=Path)
    ap.add_argument("--out", type=Path, default=Path("decay_rate.pdf"))
    ap.add_argument("--nx", type=int, default=8,
                    help="number of x locations to plot (default: 8)")
    args = ap.parse_args()

    data = np.loadtxt(args.his_file, comments="#")   # (n_times*n_pts, 4)
    n_times = len(np.unique(data[:, 0]))
    n_pts   = len(data) // n_times
    data    = data.reshape(n_times, n_pts, 4)        # (T, N, 4)

    t = data[:, 0, 0]                                # (T,)
    x = data[0, :, 1] if data.shape[2] > 4 else \
        np.linspace(0, 20, n_pts)                    # fallback
    # col 0=t, 1=u, 2=v, 3=p — x coords are in the header, not data columns
    # recover x from header
    v = data[:, :, 2]                                # (T, N)

    # x coordinates from header: 0 to 20 in steps of 0.1
    x_pts = np.linspace(0, 20, n_pts)

    # select evenly spaced x > 0
    idx = np.linspace(1, n_pts - 1, args.nx, dtype=int)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, args.nx))
    for i, pt in enumerate(idx):
        signal = np.abs(v[:, pt])
        signal = np.where(signal == 0, np.nan, signal)
        ax.semilogy(t, signal, color=colors[i], lw=1.4,
                    label=f"x = {x_pts[pt]:.1f}")

    ax.set_xlabel("t", fontsize=11)
    ax.set_ylabel("|v(x, 0, t)|", fontsize=11)
    ax.set_title("Centreline v decay — Re=50", fontsize=11)
    ax.legend(fontsize=8, ncol=2, framealpha=0.8)
    ax.grid(True, which="both", alpha=0.2)
    ax.tick_params(labelsize=9)

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
