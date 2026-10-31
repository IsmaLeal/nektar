#!/usr/bin/env python3
"""
Every DNS checkpoint and every DO archive is interpolated by FieldConvert on
the same 901 x 121 plane (x in [-5, 40], y in [-1.5, 1.5]; the inlet walls
|y| > 0.5 for x < 0 are masked). The measures (A, M, Delta L, ...) are the
solver's (FilterDOMeasures, exact quadrature); this module keeps the planes:

  DNS checkpoint  -> u, v on the plane (float32).
  DO archive      -> with --keep-grid, the plane of the mean and of every
                     mode (the Y of the particles are the filter's Y_p).

Sub-commands (see main):
    reduce-dns  DAT OUT.npz
    reduce-do   DAT OUT.npz [--keep-grid]
"""

from __future__ import annotations

import re
import sys
import argparse
import numpy as np
from pathlib import Path


# SE grid
NX, NY = 901, 121
X0, X1 = -5, 40
Y0, Y1 = -1.5, 1.5
x_vals = np.linspace(X0, X1, NX)
y_vals = np.linspace(Y0, Y1, NY)
DX = x_vals[1] - x_vals[0]
DY = y_vals[1] - y_vals[0]
X, Y = np.meshgrid(x_vals, y_vals)         # (NY, NX)
INSIDE = ~((X < 0) & (np.abs(Y) > 0.5))    # mask out inlet walls

BOX = f"box={NX},{NY},1,{X0:g},{X1:g},{Y0:g},{Y1:g},0,0"   # FieldConvert interppoints argument


# readers
def read_dat(path):
    """
    Reads the .dat FieldConvert wrote for the plane:
        one (NY,NX) array per variable (u,v,p,mode_1_u,...);
        x,y,z columns (written by FieldConvert) dropped (same as this grid)
    Points outside the domain come back as 0.
    """
    with open(path) as f:
        head = f.readline()
    names = [s.strip() for s in head.split("=", 1)[1].split(",")]
    data = np.loadtxt(path, skiprows=3)
    if data.shape[0] != NX * NY:
        raise ValueError(f"{path}: {data.shape[0]} rows, expected {NX * NY}")
    out = {}
    for j, n in enumerate(names):
        if n in ("x", "y", "z"):
            continue
        out[n] = data[:, j].reshape(NY, NX)
    # sanity: the grid FieldConvert produced is the one assumed above
    if abs(data[0, 0] - X0) > 1e-9 or abs(data[NX, 1] - (Y0 + DY)) > 1e-9:
        raise ValueError(f"{path}: grid ordering is not x-fastest on the assumed box")
    return out


def modes_in(fields):
    """
    Sorted list of mode indices present in a DO archive plane.
    """
    ks = sorted({int(m.group(1)) for n in fields for m in [re.match(r"mode_(\d+)_u$", n)] if m})
    return ks


# the planes of one field
def reduce_dns(fields):
    return dict(u=fields["u"].astype(np.float32), v=fields["v"].astype(np.float32))


def reduce_do(fields, keep_grid=False):
    ks = modes_in(fields)
    S = len(ks)
    if S == 0:
        raise ValueError("no mode_i_u fields in this plane: not a DO archive")
    out = dict(S=S)
    if keep_grid:
        out["mean_u"], out["mean_v"] = fields["u"].astype(np.float32), fields["v"].astype(np.float32)
        out["mode_u"] = np.stack([fields[f"mode_{k}_u"] for k in ks]).astype(np.float32)   # (S, NY, NX)
        out["mode_v"] = np.stack([fields[f"mode_{k}_v"] for k in ks]).astype(np.float32)
    return out


# ===== CLI =====
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("reduce-dns"); a.add_argument("dat"); a.add_argument("out")
    b = sub.add_parser("reduce-do"); b.add_argument("dat"); b.add_argument("out")
    b.add_argument("--keep-grid", action="store_true")
    args = ap.parse_args(argv)

    fields = read_dat(args.dat)
    if args.cmd == "reduce-dns":
        out = reduce_dns(fields)
    else:
        out = reduce_do(fields, keep_grid=args.keep_grid)
    np.savez(args.out, **out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
