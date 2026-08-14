#!/usr/bin/env python3
"""Rank-count invariance of the DO ensemble statistics, research plan 3.3.

Compares the 2026_07_28_rankinv_np{1,2,4} runs (identical casefiles, forced
sigma=0.0144, restart from the same twin archive, replicated RNG): the state
is not expected to match mode by mode across rank counts, but the ensemble
invariants must. Measured here per run pair:

  - zero test: max |dYi| across the restored t=0 archives (must be exact);
  - covariance spectrum: max relative eigenvalue deviation at T;
  - mean field: L2 distance at T, absolute and relative;
  - reconstructed realisations: gauge-invariant particle-term distance
    (Gram algebra, basis rotation cancels), absolute and relative.

Usage:
    python3 rankinv_check.py [--runs-root DIR] [--T 2.0] [--dt 0.01]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from convergence_orders import CELL_AREA, PLANE, FC_DEFAULT, fc_env, load_dat
from do_quicklook import yi_from_fld

S = 6


def grid(run: Path, arch: Path, tag: str, fieldconvert: str) -> Path:
    out = run / f"rankgrid_{tag}.dat"
    if out.exists():
        return out
    subprocess.run(
        [fieldconvert, "-f", "-m",
         f"interppoints:fromxml={run/'geometry.xml'}:fromfld={arch}:{PLANE}",
         str(out)], env=fc_env(),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def state(run: Path, nstep: int, fieldconvert: str):
    arch = run / "output" / "do" / f"casefile.do_{nstep:06d}.fld"
    d = load_dat(grid(run, arch, f"{nstep:06d}", fieldconvert), 6 + 3 * S)
    mean = np.concatenate([d[:, 3], d[:, 4]])
    modes = np.column_stack(
        [np.concatenate([d[:, 6 + 2 * i], d[:, 7 + 2 * i]]) for i in range(S)])
    return mean, modes, yi_from_fld(arch, S)


def pair(sa, sb):
    ma, Ua, Ya = sa
    mb, Ub, Yb = sb
    dm = np.sqrt(np.sum((ma - mb) ** 2) * CELL_AREA)
    Gaa = (Ua.T @ Ua) * CELL_AREA
    Gbb = (Ub.T @ Ub) * CELL_AREA
    Gab = (Ua.T @ Ub) * CELL_AREA
    dp2 = (np.trace(Ya @ Gaa @ Ya.T) - 2 * np.trace(Ya @ Gab @ Yb.T)
           + np.trace(Yb @ Gbb @ Yb.T)) / Ya.shape[0]
    ca = np.sort(np.linalg.eigvalsh(Ya.T @ Ya / Ya.shape[0]))[::-1]
    cb = np.sort(np.linalg.eigvalsh(Yb.T @ Yb / Yb.shape[0]))[::-1]
    dc = np.max(np.abs(ca - cb) / np.abs(ca))
    return dm, np.sqrt(max(dp2, 0.0)), dc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/home/isma/nektar_src_full/cases/se/runs"))
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--fieldconvert", default=FC_DEFAULT)
    args = ap.parse_args()

    runs = {n: args.runs_root / f"2026_07_28_rankinv_{n}"
            for n in ("np1", "np2", "np3", "np4")}
    nT = int(round(args.T / args.dt))

    print("== zero test (restored t=0 archives) ==")
    y0 = {n: yi_from_fld(r / "output" / "do" / "casefile.do_000000.fld", S)
          for n, r in runs.items()}
    for n in ("np2", "np3", "np4"):
        print(f"  max|dYi| np1-vs-{n}: {np.max(np.abs(y0['np1'] - y0[n])):.3e}")

    print(f"== invariants at T={args.T} (step {nT}) ==")
    st = {n: state(r, nT, args.fieldconvert) for n, r in runs.items()}
    mref = np.sqrt(np.sum(st["np1"][0] ** 2) * CELL_AREA)
    G = (st["np1"][1].T @ st["np1"][1]) * CELL_AREA
    Y = st["np1"][2]
    pref = np.sqrt(np.trace(Y @ G @ Y.T) / Y.shape[0])
    print(f"  reference norms (np1): ||mean|| = {mref:.4e}, "
          f"||fluct|| = {pref:.4e}")
    for a, b in (("np1", "np2"), ("np1", "np3"), ("np1", "np4"),
                 ("np2", "np3"), ("np2", "np4"), ("np3", "np4")):
        dm, dp, dc = pair(st[a], st[b])
        print(f"  {a}-vs-{b}:  mean {dm:.3e} ({dm/mref:.1e} rel)   "
              f"particles {dp:.3e} ({dp/pref:.1e} rel)   "
              f"C-spectrum max rel {dc:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
