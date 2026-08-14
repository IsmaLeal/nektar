#!/usr/bin/env python3
"""Matched-ensemble MC correctness check of the DO closure -- report sec 35.

Compares the Np=24 DO vortex run (2026_07_28_mc_do) against the 24
parent-solver runs (2026_07_28_mc_fleet/p***) launched from the identical
composed initial ensemble, at T=2:

  - mean field: relative L2 difference DO-vs-MC-sample-mean;
  - covariance spectrum: DO C eigenvalues vs top-6 eigenvalues of the MC
    sample fluctuation covariance (Gram of the 24 fluctuation fields);
  - realisations: per-particle state distance || (ubar + sum_i Y_pi u_i)
    - u_p^MC ||, rms and worst, relative to the ensemble fluctuation norm.

Registered bands (sec 35): mean rel < 2e-2 and fluct rel < 0.1 PASS-class;
0.1-0.5 interpret with S=10; > 0.5 FAIL.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from convergence_orders import FC_DEFAULT, fc_env, load_dat
from do_quicklook import yi_from_fld

S = 6
NP = 24
L = 2.0 * np.pi
PLANE = ("plane=161,161,0,0,0,{L},0,0,{L},{L},0,0,{L},0".format(L=L))
CELL_AREA = (L / 160) ** 2


def grid(xml: Path, fld: Path, cache: Path, fieldconvert: str) -> Path:
    if cache.exists():
        return cache
    subprocess.run(
        [fieldconvert, "-f", "-m",
         f"interppoints:fromxml={xml}:fromfld={fld}:{PLANE}", str(cache)],
        env=fc_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--do-run", type=Path, default=Path(
        "/home/isma/nektar_src_full/cases/vortex/runs/2026_07_28_mc_do"))
    ap.add_argument("--fleet", type=Path, default=Path(
        "/home/isma/nektar_src_full/cases/vortex/runs/2026_07_28_mc_fleet"))
    ap.add_argument("--nstep", type=int, default=1000)
    ap.add_argument("--fieldconvert", default=FC_DEFAULT)
    args = ap.parse_args()

    arch = args.do_run / "output" / "do" / f"casefile.do_{args.nstep:06d}.fld"
    d = load_dat(grid(args.do_run / "casefile.xml", arch,
                      args.do_run / f"mcgrid_{args.nstep:06d}.dat",
                      args.fieldconvert), 6 + 3 * S)
    do_mean = np.concatenate([d[:, 3], d[:, 4]])
    do_modes = np.column_stack(
        [np.concatenate([d[:, 6 + 2 * i], d[:, 7 + 2 * i]]) for i in range(S)])
    Y = yi_from_fld(arch, S)
    assert Y.shape[0] == NP, f"expected {NP} particles, got {Y.shape[0]}"

    mc = []
    for p in range(NP):
        run = args.fleet / f"p{p:03d}"
        g = load_dat(grid(run / "casefile.xml",
                          run / "output" / "chks" / "casefile_40.chk",
                          run / "mcgrid_final.dat", args.fieldconvert), 6)
        mc.append(np.concatenate([g[:, 3], g[:, 4]]))
    mc = np.column_stack(mc)                     # (2*npts, NP)
    mc_mean = mc.mean(axis=1)

    nrm = lambda f: np.sqrt(np.sum(f ** 2) * CELL_AREA)
    W = mc - mc_mean[:, None]                    # MC fluctuations
    fluct_nrm = np.sqrt(np.trace((W.T @ W) * CELL_AREA) / NP)

    dm = nrm(do_mean - mc_mean)
    print(f"mean:  |DO-MC| = {dm:.3e}   rel = {dm/nrm(mc_mean):.2e}   "
          f"(||mean|| = {nrm(mc_mean):.3e})")

    c_do = np.sort(np.linalg.eigvalsh(Y.T @ Y / NP))[::-1]
    c_mc = np.sort(np.linalg.eigvalsh((W.T @ W) * CELL_AREA / NP))[::-1][:S]
    print("C spectrum  DO:", " ".join(f"{v:.3e}" for v in c_do))
    print("C spectrum  MC:", " ".join(f"{v:.3e}" for v in c_mc))
    print(f"C spectrum max rel dev (top {S}): "
          f"{np.max(np.abs(c_do - c_mc) / c_mc):.2e}")

    dists = [nrm(do_mean + do_modes @ Y[p] - mc[:, p]) for p in range(NP)]
    rms = np.sqrt(np.mean(np.square(dists)))
    print(f"realisations: rms state dist = {rms:.3e}  worst = {max(dists):.3e}"
          f"   rel to ||fluct|| = {fluct_nrm:.3e}:  rms {rms/fluct_nrm:.2e},"
          f" worst {max(dists)/fluct_nrm:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
