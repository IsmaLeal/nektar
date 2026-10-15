#!/usr/bin/env python3
"""
Reads a run's output/extract/grid.h5 and writes:
    <out>_u.npy   u on the extraction plane for x >= -1, every snapshot, float16 (T, ny, nx)
    <out>.npz     x, y (the plane's axes for that window), t, xs (x > 0), up, low (T, len(xs))

up and low are du/dy at the plane's first rows off the walls, y = 1.475 (sign flipped) and y = -1.475
    low = (u[2] - u[0]) / (2 dy)
    up = -(u[-1] - u[-3]) / (2 dy)

Example:
    python3 dudy_wall_data.py cases/se/runs/2026_09_13_jed_sweep/Re100_sig0.0144/dns_Re100_sig0.0144_seed45 \
        --out /tmp/dudy_Re100_sig0.0144_dns_seed45
"""
import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory (holding output/extract/grid.h5)")
    ap.add_argument("--out", required=True, help="output prefix: <out>_u.npy and <out>.npz")
    ap.add_argument("--xmin", type=float, default=-1.0, help="left end of the field window")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(Path(args.run) / "output/extract/grid.h5", "r") as f:
        x, y = f["x"][0], f["y"][:, 0]                  # load x, y
        t = f["t"][:]                                   # t
        j0 = int(np.searchsorted(x, args.xmin))         # returns index i s.t. x[i]
        u = f["u"][:, :, j0:]
    dy = y[1] - y[0]
    xm = x[j0:] > 0                                     # mask for points after the expansion
    low = (u[:, 2, xm] - u[:, 0, xm]) / (2 * dy)
    up = -(u[:, -1, xm] - u[:, -3, xm]) / (2 * dy)
    np.save(f"{out}_u.npy", u.astype(np.float16))
    np.savez(f"{out}.npz", x=x[j0:], y=y, t=t, xs=x[j0:][xm], up=up.astype(np.float64), low=low.astype(np.float64))
    print(f"{out}_u.npy {u.shape}, t {t[0]}..{t[-1]}, low {low.min():.3f}..{low.max():.3f}, "
          f"up {up.min():.3f}..{up.max():.3f}, u {u.min():.3f}..{u.max():.3f}")


if __name__ == "__main__":
    main()
