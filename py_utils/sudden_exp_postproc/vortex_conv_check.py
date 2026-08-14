#!/usr/bin/env python3
"""Vortex dt-convergence check: splitting-vs-pressure-BC discriminant.

Successive-pair distances of the gauge-invariant observable pair
Q = (mean, {u'_p}) between the runs of the 2026_08_06_conv_vortex_dt* family
(periodic vortex, no walls), in the metric of convergence_orders.py but on the
vortex analysis grid of vortex_mc_check.py. The periodic domain has no
pressure-BC path, so a fluctuation order ~1 here confirms the mean<->fluctuation
splitting as a sufficient cause of the first order measured on the sudden
expansion; an order ~2 would indict the modes' low-order pressure BC instead.
Registered bands in the family README (dt0200).

Usage:
    python3 vortex_conv_check.py [--runs-root DIR]
        [--tags dt0200,dt0100,dt0050,dt0025] [--dts 0.002,0.001,0.0005,0.00025]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vortex_mc_check import CELL_AREA, S, grid
from convergence_orders import FC_DEFAULT, load_dat
from do_quicklook import yi_from_fld


def load_run(run: Path, nstep: int, fieldconvert: str):
    arch = run / "output" / "do" / f"casefile.do_{nstep:06d}.fld"
    d = load_dat(grid(run / "casefile.xml", arch,
                      run / f"convgrid_{nstep:06d}.dat", fieldconvert),
                 6 + 3 * S)
    mean = np.concatenate([d[:, 3], d[:, 4]])
    modes = np.column_stack(
        [np.concatenate([d[:, 6 + 2 * i], d[:, 7 + 2 * i]]) for i in range(S)])
    Y = yi_from_fld(arch, S)
    return mean, modes @ Y.T            # (2*npts,), (2*npts, Np)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/home/isma/nektar_src_full/cases/vortex/runs"))
    ap.add_argument("--tags", default="dt0200,dt0100,dt0050,dt0025")
    ap.add_argument("--dts", default="0.002,0.001,0.0005,0.00025")
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--fieldconvert", default=FC_DEFAULT)
    args = ap.parse_args()

    tags = args.tags.split(",")
    dts = [float(v) for v in args.dts.split(",")]
    nrm = lambda f: np.sqrt(np.sum(f ** 2) * CELL_AREA)

    states = []
    for tag, dt in zip(tags, dts):
        run = args.runs_root / f"2026_08_06_conv_vortex_{tag}"
        states.append(load_run(run, round(args.T / dt), args.fieldconvert))

    print(f"{'pair':>16}  {'d_mean':>10}  {'d_fluct':>10}  {'d_full':>10}")
    dm, df, dq = [], [], []
    for (m_a, f_a), (m_b, f_b), ta, tb in zip(states, states[1:],
                                              tags, tags[1:]):
        np_ = f_a.shape[1]
        d_mean = nrm(m_a - m_b)
        d_fluct = np.sqrt(sum(nrm(f_a[:, p] - f_b[:, p]) ** 2
                              for p in range(np_)) / np_)
        d_full = np.hypot(d_mean, d_fluct)
        dm.append(d_mean); df.append(d_fluct); dq.append(d_full)
        print(f"{ta+'-'+tb:>16}  {d_mean:10.3e}  {d_fluct:10.3e}  "
              f"{d_full:10.3e}")

    print(f"\n{'order from':>16}  {'p_mean':>8}  {'p_fluct':>8}  {'p_full':>8}")
    for k in range(len(dm) - 1):
        print(f"{tags[k]+'..'+tags[k+2]:>16}  "
              f"{np.log2(dm[k]/dm[k+1]):8.2f}  "
              f"{np.log2(df[k]/df[k+1]):8.2f}  "
              f"{np.log2(dq[k]/dq[k+1]):8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
