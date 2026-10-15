#!/usr/bin/env python3
"""Merge the per-file .npz reductions of extract_one.sh into one HDF5.

    assemble.py dns RUN NPZDIR OUT.h5 [--dt 0.01] [--chk-steps 400]
    assemble.py do  RUN NPZDIR OUT.h5 [--dt 0.01]

DNS  -> grid.h5:   t (T,), u, v (T, 121, 901) float32, v_axis (T, 801),
                   u_wall (T, 2, 800), p_wall (T, 2, 801), A, dL, M_grid,
                   norm2 (T,), x, y, and the his file is NOT copied (rsync it).
DO   -> measures.h5: t (T,), step (T,), yi (T, Np, S) float64, a (T, S),
                   G (T, S, S), A0 (T,), Ai (T, S), mean_norm2 (T,),
                   v_axis_mean (T, 801), v_axis_modes (T, S, 801),
                   u_wall_mean (T, 2, 800), u_wall_modes (T, S, 2, 800),
                   p_wall_mean (T, 2, 801), p_wall_modes (T, S, 2, 801), x, y;
                   plus group grid/ with mean_u, mean_v (K, 121, 901) and
                   mode_u, mode_v (K, S, 121, 901) float32 for the K archives
                   reduced with --keep-grid, with their own t_grid.
The time of a DNS checkpoint N is N * chk_steps * dt; of a DO archive with
step NNNNNN it is NNNNNN * dt. Row/column conventions: extract_lib.py.
Exits non-zero if any index in the contiguous range is missing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_lib as L  # noqa: E402


def indexed(npzdir):
    files = {}
    for p in Path(npzdir).glob("*.npz"):
        m = re.fullmatch(r"(\d+)\.npz", p.name)
        if m:
            files[int(m.group(1))] = p
    if not files:
        sys.exit(f"no <index>.npz in {npzdir}")
    ks = sorted(files)
    return ks, files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["dns", "do"])
    ap.add_argument("run")
    ap.add_argument("npzdir")
    ap.add_argument("out")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--chk-steps", type=int, default=400)
    ap.add_argument("--allow-gaps", action="store_true")
    args = ap.parse_args()

    ks, files = indexed(args.npzdir)
    if args.mode == "dns":
        expected = list(range(ks[0], ks[-1] + 1))
    else:
        d = np.diff(ks)
        step = int(d.min()) if len(d) else 0
        expected = list(range(ks[0], ks[-1] + 1, step)) if step else ks
    missing = sorted(set(expected) - set(ks))
    if missing and not args.allow_gaps:
        sys.exit(f"{len(missing)} missing indices, first: {missing[:10]}")
    T = len(ks)
    first = np.load(files[ks[0]])

    with h5py.File(args.out, "w") as h:
        h["x"], h["y"] = L.X, L.Y
        h.attrs["run"] = str(args.run)
        h.attrs["mode"] = args.mode
        h.attrs["dt"] = args.dt
        h["index"] = np.array(ks)
        if args.mode == "dns":
            h.attrs["chk_steps"] = args.chk_steps
            h["t"] = np.array(ks, dtype=np.float64) * args.chk_steps * args.dt
            keys = ["v_axis", "u_wall", "p_wall", "A", "dL", "M_grid", "norm2"]
            keys = [k for k in keys if k in first]
            ds = {k: h.create_dataset(k, (T,) + np.asarray(first[k]).shape, dtype=np.float64) for k in keys}
            for f in ("u", "v"):
                ds[f] = h.create_dataset(f, (T, L.NY, L.NX), dtype=np.float32, chunks=(1, L.NY, L.NX))
            for i, k in enumerate(ks):
                z = np.load(files[k])
                for name in ds:
                    ds[name][i] = z[name]
        else:
            h["step"] = np.array(ks)
            h["t"] = np.array(ks, dtype=np.float64) * args.dt
            S = int(first["S"])
            h.attrs["S"] = S
            keys = ["yi", "a", "G", "A0", "Ai", "mean_norm2", "v_axis_mean", "v_axis_modes",
                    "u_wall_mean", "u_wall_modes", "p_wall_mean", "p_wall_modes"]
            keys = [k for k in keys if k in first]
            # yi and the scalar/matrix projections in float64; the line samples in float32
            f32 = {"v_axis_mean", "v_axis_modes", "u_wall_mean", "u_wall_modes", "p_wall_mean", "p_wall_modes"}
            ds = {k: h.create_dataset(k, (T,) + np.asarray(first[k]).shape,
                                      dtype=np.float32 if k in f32 else np.float64) for k in keys}
            grid_idx = [k for k in ks if "mean_u" in np.load(files[k])]
            if grid_idx:
                g = h.create_group("grid")
                g["step"] = np.array(grid_idx)
                g["t"] = np.array(grid_idx, dtype=np.float64) * args.dt
                K = len(grid_idx)
                gd = {
                    "mean_u": g.create_dataset("mean_u", (K, L.NY, L.NX), dtype=np.float32, chunks=(1, L.NY, L.NX)),
                    "mean_v": g.create_dataset("mean_v", (K, L.NY, L.NX), dtype=np.float32, chunks=(1, L.NY, L.NX)),
                    "mode_u": g.create_dataset("mode_u", (K, S, L.NY, L.NX), dtype=np.float32, chunks=(1, 1, L.NY, L.NX)),
                    "mode_v": g.create_dataset("mode_v", (K, S, L.NY, L.NX), dtype=np.float32, chunks=(1, 1, L.NY, L.NX)),
                }
            gi = 0
            for i, k in enumerate(ks):
                z = np.load(files[k])
                if int(z["S"]) != S:
                    sys.exit(f"index {k}: S={int(z['S'])} differs from {S}")
                for name in ds:
                    ds[name][i] = z[name]
                if grid_idx and "mean_u" in z:
                    for name in gd:
                        gd[name][gi] = z[name]
                    gi += 1
    print(f"{args.out}: {T} entries, index {ks[0]}..{ks[-1]}" + (f", {len(missing)} gaps" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
