"""Asymmetry measure M(t) from a HistoryPoints axis file.

M is defined as in Ducimetiere, Boujo & Gallaire, PRF 9, 053905 (2024),
Eq. (25): M = sgn[V(x=2, ys)] * sqrt( int_0^L V(x, ys)^2 dx ), evaluated
on the expansion symmetry axis. The .his file is the one-row-per-point
format: each output time dumps npts rows of "t u v p", points ordered as
in the header.

Usage:
    python mbar_from_his.py <axis_v.his> [--tail 50] [--out M_of_t.txt]
"""

import argparse
import numpy as np


def load_his(path):
    xs = []
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            parts = line[1:].split()
            if len(parts) == 4 and parts[0].isdigit():
                xs.append(float(parts[1]))
    x = np.array(xs)
    npts = len(x)
    data = np.loadtxt(path, comments="#")
    nblocks = len(data) // npts
    blocks = data[: nblocks * npts].reshape(nblocks, npts, 4)
    return x, blocks


def m_of_t(x, blocks):
    j2 = int(np.argmin(np.abs(x - 2.0)))
    t = blocks[:, 0, 0]
    v = blocks[:, :, 2]
    mag = np.sqrt(np.trapz(v**2, x, axis=1))
    sgn = np.sign(v[:, j2])
    return t, sgn * mag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("his")
    ap.add_argument("--tail", type=int, default=50,
                    help="time units over which to average the final M")
    ap.add_argument("--out", default=None, help="write t, M(t) to file")
    args = ap.parse_args()

    x, blocks = load_his(args.his)
    t, M = m_of_t(x, blocks)

    # drop duplicated blocks (init flushes repeat t=0)
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, M = t[keep], M[keep]

    print(f"{args.his}: {len(t)} samples, t in [{t[0]:g}, {t[-1]:g}], "
          f"{len(x)} axis points, dx={x[1]-x[0]:g}")
    late = t >= t[-1] - args.tail
    print(f"final M               = {M[-1]:+.6f}")
    print(f"mean M (last {args.tail:g} t.u.) = {np.mean(M[late]):+.6f}"
          f"  (std {np.std(M[late]):.2e})")
    print(f"paper Mbar at Re=100  = +/-0.1935")

    if args.out:
        np.savetxt(args.out, np.column_stack([t, M]), header="t M")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
