#!/usr/bin/env python3
"""Assemble a load_chk-compatible .h5 for a DO run from a grid cache.

Reads per-archive plane interpolations from
RUN/output/py_utils_results/grid_cache/gd_STEP.dat (produced by FieldConvert
interppoints, columns x y z | u v p | mode_i_u mode_i_v ...), plus Yi from
the archive metadata, and writes an HDF5 with the same keys load_chk.py
produces, so downstream tools (animate_do_panels.py, reconstruct_fields.py)
work unchanged:
    x, y            (Ny, Nx)
    u, v, p         (Ny, Nx, T)     ensemble-mean fields
    mode_u, mode_v  (S, Ny, Nx, T)
    yi              (P, S, T)
    t, t_do         (T,)

The cache is incremental: run the FieldConvert loop again after the
simulation advances, then re-run this assembler; only missing archives need
interpolating.

Usage:
    python3 assemble_do_h5.py RUN_DIR [--out RUN_DIR/output/out_grid.h5]
                              [--nx 901] [--ny 121]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np

from do_quicklook import yi_series


def casefile_param(run: Path, name: str, default=None):
    txt = (run / "casefile.xml").read_text()
    m = re.search(rf"{name}\s*=\s*([0-9eE.+-/]+)", txt)
    if not m:
        return default
    try:
        return float(eval(m.group(1), {"__builtins__": {}}))
    except Exception:
        return default


def load_dat(path: Path, ncol: int) -> np.ndarray:
    rows = []
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) >= ncol:
            try:
                rows.append([float(v) for v in p[:ncol]])
            except ValueError:
                pass
    return np.array(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--nx", type=int, default=901)
    ap.add_argument("--ny", type=int, default=121)
    args = ap.parse_args()
    run = args.run
    out = args.out or run / "output/out_grid.h5"
    S = int(casefile_param(run, "DOModes"))
    dt = casefile_param(run, "TimeStep", 0.01)
    nx, ny = args.nx, args.ny
    ncol = 6 + 2 * S  # plane layout: x y z u v p + (mode_i_u, mode_i_v)*S

    files = sorted((run / "output/py_utils_results/grid_cache").glob("gd_*.dat"))
    steps = [int(re.search(r"gd_(\d+)", f.name).group(1)) for f in files]

    def togrid(d, c):
        firstx = d[0, 0] != d[1, 0]
        m = d[:, c].reshape((ny, nx) if firstx else (nx, ny))
        return m if firstx else m.T

    d0 = load_dat(files[0], ncol)
    X, Y = togrid(d0, 0), togrid(d0, 1)
    T = len(files)
    u = np.zeros((ny, nx, T)); v = np.zeros((ny, nx, T)); p = np.zeros((ny, nx, T))
    mu = np.zeros((S, ny, nx, T)); mv = np.zeros((S, ny, nx, T))
    for k, f in enumerate(files):
        d = load_dat(f, ncol)
        u[:, :, k] = togrid(d, 3)
        v[:, :, k] = togrid(d, 4)
        p[:, :, k] = togrid(d, 5)
        for i in range(S):
            mu[i, :, :, k] = togrid(d, 6 + 2 * i)
            mv[i, :, :, k] = togrid(d, 7 + 2 * i)

    t_yi, yi = yi_series(str(run / "output/do"), nmodes=S, dt=dt)
    steps_yi = list((np.round(t_yi / dt)).astype(int))
    sel = [steps_yi.index(s) for s in steps if s in steps_yi]
    yi = yi[sel]                              # (T, P, S)
    yi = np.transpose(yi, (1, 2, 0))          # -> (P, S, T)
    t = np.array(steps, dtype=float) * dt

    with h5py.File(out, "w") as f:
        for name, arr in [("x", X), ("y", Y), ("u", u), ("v", v), ("p", p),
                          ("mode_u", mu), ("mode_v", mv), ("yi", yi),
                          ("t", t), ("t_do", t)]:
            f.create_dataset(name, data=arr, compression="gzip",
                             compression_opts=2)
        f.create_dataset("metadata_json", data=np.bytes_(
            b'{"source": "assemble_do_h5", "run": "%s"}'
            % str(run).encode()))
    print(f"wrote {out}: T={T} frames, S={S}, P={yi.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
