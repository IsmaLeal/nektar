#!/usr/bin/env python3
"""Measured temporal orders of the coupled DO scheme from a dt-refinement family.

Takes two families of otherwise-identical runs (full DO and S=0 control) at
dt, dt/2, dt/4, dt/8, and produces one figure demonstrating the three-component
error anatomy:

  1. global L2 differences are dominated by a sqrt(nu*dt) projection wall
     layer (data-selected startup exponent 0.5);
  2. interior field differences converge at second order;
  3. the ensemble-coefficient (particle) term converges at first order,
     consistent with the mean<->fluctuation operator splitting.

Distances are the gauge-invariant observable metric (mean field + reconstructed
per-particle fields), evaluated with S x S Gram algebra; see the companion note
convergence_orders.tex for what each panel shows and why.

Usage:
    python3 convergence_orders.py [--runs-root DIR] [--out FIG.png]
        [--do-tags dt100,dt050,dt025,dt0125] [--s0-tags s0dt100,...]
        [--dts 0.01,0.005,0.0025,0.00125] [--T 2.0]
        [--interior 4,30,0.9]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from do_quicklook import yi_from_fld

PLANE = "plane=901,121,-5,-1.5,0,40,-1.5,0,40,1.5,0,-5,1.5,0"
FC_DEFAULT = os.environ.get(
    "NEKTAR_FIELDCONVERT",
    "/home/isma/nektar_repro/build/utilities/FieldConvert/FieldConvert")
CELL_AREA = (45.0 / 900) * (3.0 / 120)   # uniform plane-grid quadrature weight


def fc_env():
    env = os.environ.copy()
    try:
        asan = subprocess.run(["gcc", "-print-file-name=libasan.so"],
                              capture_output=True, text=True).stdout.strip()
        if "/" in asan:
            env["LD_PRELOAD"] = asan
    except OSError:
        pass
    env["ASAN_OPTIONS"] = "detect_leaks=0"
    return env


def load_dat(path, ncol):
    rows = []
    for ln in open(path):
        p = ln.split()
        if len(p) >= ncol:
            try:
                rows.append([float(v) for v in p[:ncol]])
            except ValueError:
                pass
    return np.asarray(rows)


def ensure_grid(run: Path, source: Path, fieldconvert: str):
    """Interpolate `source` (.chk/.fld) onto the analysis plane, cached."""
    out = run / "gdT.dat"
    if out.exists():
        return out
    subprocess.run(
        [fieldconvert, "-f", "-m",
         f"interppoints:fromxml={run/'geometry.xml'}:fromfld={source}:{PLANE}",
         str(out)], env=fc_env(),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def do_state(run: Path, dt: float, T: float, fieldconvert: str, nmodes=6):
    nst = int(round(T / dt))
    arch = run / "output" / "do" / f"casefile.do_{nst:06d}.fld"
    d = load_dat(ensure_grid(run, arch, fieldconvert), 6 + 3 * nmodes)
    mean = np.concatenate([d[:, 3], d[:, 4]])
    modes = np.column_stack(
        [np.concatenate([d[:, 6 + 2 * i], d[:, 7 + 2 * i]])
         for i in range(nmodes)])
    return d[:, 0], d[:, 1], mean, modes, yi_from_fld(arch, nmodes)


def s0_state(run: Path, fieldconvert: str):
    chks = sorted((run / "output" / "chks").glob("casefile_*.chk"),
                  key=lambda p: int(re.search(r"_(\d+)", p.name).group(1)))
    d = load_dat(ensure_grid(run, chks[-1], fieldconvert), 6)
    return d[:, 0], d[:, 1], np.concatenate([d[:, 3], d[:, 4]])


def obs_dist(sa, sb, mask2=None):
    """Gauge-invariant distance: (mean part, particle part)."""
    ma, Ua, Ya = sa
    mb, Ub, Yb = sb
    if mask2 is not None:
        ma, mb = ma[mask2], mb[mask2]
        if Ua is not None:
            Ua, Ub = Ua[mask2], Ub[mask2]
    dm = np.sqrt(np.sum((ma - mb) ** 2) * CELL_AREA)
    if Ua is None:
        return dm, 0.0
    Gaa = (Ua.T @ Ua) * CELL_AREA
    Gbb = (Ub.T @ Ub) * CELL_AREA
    Gab = (Ua.T @ Ub) * CELL_AREA
    dp2 = (np.trace(Ya @ Gaa @ Ya.T) - 2 * np.trace(Ya @ Gab @ Yb.T)
           + np.trace(Yb @ Gbb @ Yb.T)) / Ya.shape[0]
    return dm, np.sqrt(max(dp2, 0.0))


def exponent_scan(E, dts):
    """LS residual of the two-term model over (startup, bulk) exponent grid.

    E is the 3x3 matrix of inner products between successive observable
    differences; the model e_i = a_i*PHI + b_i*PSI with a_i ~ dt^ps,
    b_i ~ dt^pb leaves 3 unknown scalars (<PHI,PHI>, <PHI,PSI>, <PSI,PSI>)
    for 6 knowns: the fit residual selects the exponent pair.
    """
    ps_grid = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
    pb_grid = np.array([1.5, 2.0, 2.5, 3.0])
    res = np.zeros((len(ps_grid), len(pb_grid)))
    for i, ps in enumerate(ps_grid):
        for j, pb in enumerate(pb_grid):
            a = dts[:3] ** ps * (1 - 0.5 ** ps)
            b = dts[:3] ** pb * (1 - 0.5 ** pb)
            A, y = [], []
            for k in range(3):
                for l in range(k, 3):
                    A.append([a[k] * a[l], a[k] * b[l] + a[l] * b[k],
                              b[k] * b[l]])
                    y.append(E[k, l])
            A, y = np.array(A), np.array(y)
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            res[i, j] = np.linalg.norm(A @ sol - y) / np.linalg.norm(y)
    return ps_grid, pb_grid, res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/home/isma/nektar_src_full/cases/se/runs"))
    ap.add_argument("--do-tags",
                    default="2026_07_19_conv_dt100,2026_07_19_conv_dt050,"
                            "2026_07_19_conv_dt025,2026_07_19_conv_dt0125")
    ap.add_argument("--s0-tags",
                    default="2026_07_19_conv_s0dt100,2026_07_19_conv_s0dt050,"
                            "2026_07_19_conv_s0dt025,2026_07_19_conv_s0dt0125")
    ap.add_argument("--dts", default="0.01,0.005,0.0025,0.00125")
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--interior", default="4,30,0.9",
                    help="xmin,xmax,ymax of the wall-excluding interior box")
    ap.add_argument("--fieldconvert", default=FC_DEFAULT)
    ap.add_argument("--out", type=Path, default=Path("convergence_orders.png"))
    args = ap.parse_args()

    dts = np.array([float(v) for v in args.dts.split(",")])
    do_runs = [args.runs_root / t for t in args.do_tags.split(",")]
    s0_runs = [args.runs_root / t for t in args.s0_tags.split(",")]

    DO = [do_state(r, dt, args.T, args.fieldconvert)
          for r, dt in zip(do_runs, dts)]
    S0 = [s0_state(r, args.fieldconvert) for r in s0_runs]

    x, y = DO[0][0], DO[0][1]
    xmin, xmax, ymax = (float(v) for v in args.interior.split(","))
    mask2 = np.concatenate([(x >= xmin) & (x <= xmax) & (np.abs(y) <= ymax)] * 2)

    do_obs = [(s[2], s[3], s[4]) for s in DO]
    s0_obs = [(s[2], None, None) for s in S0]

    # successive-pair distances, full domain and interior
    rows = {}
    for name, obs, msk in (("full", do_obs, None), ("int", do_obs, mask2),
                           ("s0full", s0_obs, None), ("s0int", s0_obs, mask2)):
        dm, dp = zip(*[obs_dist(obs[i], obs[i + 1], msk) for i in range(3)])
        rows[name] = (np.array(dm), np.array(dp))

    # inner-product matrix of full-domain observable differences (for the scan)
    def pairdot(u, v, obs):
        (mu, Uu, Yu), (mv, Uv, Yv) = obs[u], obs[v]
        val = np.sum(mu * mv) * CELL_AREA
        if Uu is not None:
            val += np.trace(Yu @ ((Uu.T @ Uv) * CELL_AREA) @ Yv.T) / Yu.shape[0]
        return val
    E = np.zeros((3, 3))
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3)]):
        for j, (c, d_) in enumerate([(0, 1), (1, 2), (2, 3)]):
            E[i, j] = (pairdot(a, c, do_obs) - pairdot(a, d_, do_obs)
                       - pairdot(b, c, do_obs) + pairdot(b, d_, do_obs))
    ps_grid, pb_grid, res = exponent_scan(E, dts)
    ib, jb = np.unravel_index(np.argmin(res), res.shape)

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    dpair = dts[:3]
    dcomb_full = np.sqrt(rows["full"][0] ** 2 + rows["full"][1] ** 2)

    series = [
        (dcomb_full,        "tab:red",    "o", "whole domain (walls included)",          0.5),
        (rows["s0full"][0], "tab:orange", "s", "whole domain, no DO (control)",          None),
        (rows["int"][0],    "tab:blue",   "o", "mean flow, away from walls",             2.0),
        (rows["int"][1],    "tab:green",  "^", "stochastic coefficients, away from walls", 1.0),
    ]
    for d, c, mk, lab, pref in series:
        ax.loglog(dpair, d, mk + "-", color=c, lw=1.8, ms=7, label=lab)
        if pref is not None:
            ref = d[-1] * (dpair / dpair[-1]) ** pref
            ax.loglog(dpair, ref * 1.35, "--", color=c, lw=1.0, alpha=0.6)
            ax.annotate(f"$\\Delta t^{{{pref:g}}}$",
                        (dpair[-1], ref[-1] * 1.45), fontsize=11, color=c,
                        ha="left")
    ax.set_xlabel("$\\Delta t$")
    ax.set_ylabel("difference between runs at $\\Delta t$ and $\\Delta t/2$")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")

    print("\nsuccessive-pair distances (combined, full domain):",
          np.array2string(dcomb_full, precision=3))
    print(f"exponent scan best: startup {ps_grid[ib]:g}, "
          f"bulk {pb_grid[jb]:g} (residual {100*res[ib, jb]:.1f}%)")
    print("finest-pair measured orders:")
    print(f"  fields, interior, no DO : "
          f"{np.log2(rows['s0int'][0][1]/rows['s0int'][0][2]):.2f}")
    print(f"  fields, interior, DO    : "
          f"{np.log2(rows['int'][0][1]/rows['int'][0][2]):.2f}")
    print(f"  coefficients, interior  : "
          f"{np.log2(rows['int'][1][1]/rows['int'][1][2]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
