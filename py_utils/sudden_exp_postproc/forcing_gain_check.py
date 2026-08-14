#!/usr/bin/env python3
"""White-noise injection audit of the DO forcing path, research plan 3.3.

Re-creation of the 2026-07-15 executed-gain audit (runs 2026_07_15_val_np*,
deleted 2026-07-21; the original analysis script was session-scoped and is
lost). Rig: S = 1, Np = 1000, one channel g = qdag, white-in-time branch
(DOForcingTau = 0: eta = sigma*sqrt(dt)*xi drawn i.i.d. per step, particle
and channel), per-step DO archives. Runs:
cases/se/runs/2026_07_30_frc_audit_np{1,2,4}.

Measured, per run:

  - draw sanity: per-step eta std against the coded sigma*sqrt(dt); lag-1
    autocorrelation (must vanish: the branch has no memory);
  - archive alignment: raw correlation E[eta^(n+s) dY^(n)] / E[eta^2] over
    shifts s. The draw stored in archive n is evaluated into R^(n) at the
    end of step n and drives the step from archive n to n+1;
  - executed weights: least squares of dY_p^(n) on eta^(n) (own slot),
    eta^(n-1), eta^(n+1) (acausal control) and the Y history. The own-slot
    coefficient is unbiased regardless of the drift's nonlinearity (the
    draw is independent of all state entering its step); the eta^(n-1)
    regression coefficient is NOT (that draw built Y^(n)), so the second
    slot is instead read from the raw-correlation identity
    raw(-2) = c(-2) + (1/3)*c(-1) + O(dt);
  - coded weights: the IMEX-BDF2 explicit extrapolation puts (4/3)*dt*g1
    on the step's own draw and -(2/3)*dt*g1 on the previous one, with
    g1 = <g, u_1> under the solver's mass normalisation of g. g1 is
    evaluated as the grid-quadrature cosine <g,u1>/(||g|| ||u1||) (unit
    norms are exact in the solver's own quadrature; the cosine cancels
    most of the grid bias), sampled over the run, with a finer-grid
    recompute bounding the residual quadrature error;
  - the audit number: executed/coded factor per injection slot, plus
    windowed factors against the g1(t) drift. A wrong amplitude is a
    lag-uniform factor; a wrong template or sub-step placement breaks the
    slot pattern.

2026-07-30 result (registration and verdicts: README of the np1 run):
own-slot factor 1.0002-1.0003 +- 0.0001, second slot 1.0010 at O(dt)
accuracy, slot ratio -1.9991, acausal zero, identical at 1/2/4 ranks with
bit-identical eta sequences. The historical 0.9919 is NOT reproduced on
the same binary: the executed injection matches the coded one to ~2e-4.

Usage:
    python3 forcing_gain_check.py --run DIR [--fine] [--nsample 21]
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np

from convergence_orders import CELL_AREA, PLANE, FC_DEFAULT, fc_env, load_dat
from do_quicklook import yi_from_fld

S = 1
PLANE_FINE = "plane=1801,241,-5,-1.5,0,40,-1.5,0,40,1.5,0,-5,1.5,0"
CELL_AREA_FINE = (45.0 / 1800) * (3.0 / 240)

_ETA_TAG = re.compile(
    r"<DOVelocityCorrectionScheme_ForcingEta_hex>([0-9a-fA-F]+)<")


def eta_from_fld(fld):
    fld = Path(fld)
    info = fld / "Info.xml" if fld.is_dir() else fld
    m = _ETA_TAG.search(info.read_text(errors="ignore"))
    if m is None:
        raise ValueError(f"no ForcingEta_hex metadata in {info}")
    return np.frombuffer(bytes.fromhex(m.group(1)), dtype="<f8")


def interp(run: Path, fld: Path, tag: str, fieldconvert: str,
           plane: str) -> Path:
    out = run / f"gaingrid_{tag}.dat"
    if out.exists():
        return out
    subprocess.run(
        [fieldconvert, "-f", "-m",
         f"interppoints:fromxml={run/'geometry.xml'}:fromfld={fld}:{plane}",
         str(out)], env=fc_env(),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def gain(run: Path, arch: Path, tag: str, fieldconvert: str,
         fine: bool) -> tuple[float, float]:
    """Grid-quadrature cosine <g,u1>/(||g|| ||u1||) and ||u1||_grid."""
    plane, area = ((PLANE_FINE, CELL_AREA_FINE) if fine
                   else (PLANE, CELL_AREA))
    suf = "f" if fine else "c"
    g = load_dat(interp(run, run / "qdag_channel.fld", f"g_{suf}",
                        fieldconvert, plane), 5)
    d = load_dat(interp(run, arch, f"{tag}_{suf}", fieldconvert, plane),
                 6 + 3 * S)
    gu = np.concatenate([g[:, 3], g[:, 4]])
    u1 = np.concatenate([d[:, 6], d[:, 7]])
    ip = np.sum(gu * u1) * area
    ng = np.sqrt(np.sum(gu ** 2) * area)
    n1 = np.sqrt(np.sum(u1 ** 2) * area)
    return ip / (ng * n1), n1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--tmin", type=int, default=20,
                    help="first archive index used in the fit")
    ap.add_argument("--nsample", type=int, default=21,
                    help="archives sampled for the g1(t) series")
    ap.add_argument("--fine", action="store_true",
                    help="add the finer-grid g1 recompute (quadrature bound)")
    ap.add_argument("--fieldconvert", default=FC_DEFAULT)
    args = ap.parse_args()
    run = args.run

    files = sorted(run.glob("output/do/*.do_*.fld"),
                   key=lambda f: int(re.search(r"do_(\d+)", f.name).group(1)))
    nT = len(files)
    eta = np.stack([eta_from_fld(f) for f in files])          # (T, Np)
    Y = np.stack([yi_from_fld(f, S)[:, 0] for f in files])    # (T, Np)
    Np = eta.shape[1]
    print(f"{run.name}: {nT} archives, Np = {Np}")

    # -- draw sanity ------------------------------------------------------
    e = eta[args.tmin:]
    sig_step = args.sigma * np.sqrt(args.dt)
    ac1 = np.mean(e[1:] * e[:-1]) / np.mean(e ** 2)
    print(f"  eta std {e.std():.5f} (coded sigma*sqrt(dt) = {sig_step:.5f}, "
          f"ratio {e.std()/sig_step:.4f});  lag-1 autocorr {ac1:+.4f};  "
          f"max |ensemble mean| {np.max(np.abs(e.mean(axis=1))):.2e}")

    # -- alignment scan ---------------------------------------------------
    dY = Y[1:] - Y[:-1]                       # dY[n]: archive n -> n+1
    var = np.mean(e ** 2)
    print("  raw correlation E[eta^(n+s) dY^(n)]/E[eta^2] by shift s:")
    for s in range(-3, 4):
        lo, hi = max(args.tmin, -s + 1), min(dY.shape[0], dY.shape[0] - s)
        r = np.mean(eta[lo + s + 1:hi + s + 1] * dY[lo:hi]) / var
        print(f"    s = {s:+d}: {r:+.4e}")

    # -- executed weights: joint LS on aligned draws + Y history ----------
    def fit(n0, n1):
        rows, cols = [], []
        for n in range(n0, n1):
            rows.append(dY[n])
            cols.append((eta[n], eta[n - 1], eta[n + 1], Y[n], Y[n - 1]))
        b = np.concatenate(rows)
        A = np.column_stack([np.concatenate([c[i] for c in cols])
                             for i in range(5)])
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = b - A @ sol
        se = (np.sqrt(np.sum(resid ** 2) / (b.size - 5))
              / np.sqrt(np.sum((A - A.mean(axis=0)) ** 2, axis=0)))
        return sol, se

    n0, n1 = args.tmin, nT - 2
    sol, se = fit(n0, n1)
    c0, cac = sol[0], sol[2]
    print(f"  executed weights: own-draw {c0:+.5e} (se {se[0]:.1e}), "
          f"acausal {cac:+.2e} (se {se[2]:.1e})")

    # -- coded weights ----------------------------------------------------
    idx = np.unique(np.linspace(args.tmin, nT - 1, args.nsample).astype(int))
    g1s, n1s = zip(*[gain(run, files[i], f"{i:06d}", args.fieldconvert,
                          False) for i in idx])
    g1 = float(np.mean(g1s))
    print(f"  g1 = <g,u1>: mean {g1:.5f}, range [{min(g1s):.5f}, "
          f"{max(g1s):.5f}] over {len(idx)} samples;  "
          f"||u1||_grid = {np.mean(n1s):.5f} (solver-exact: 1)")
    if args.fine:
        i0 = idx[len(idx) // 2]
        gf, _ = gain(run, files[i0], f"{i0:06d}", args.fieldconvert, True)
        gc, _ = gain(run, files[i0], f"{i0:06d}", args.fieldconvert, False)
        print(f"  quadrature bound: g1(archive {i0}) coarse {gc:.5f} vs "
              f"fine {gf:.5f} ({abs(gf-gc)/abs(gf):.2e} rel)")

    coded0 = (4.0 / 3.0) * args.dt * g1
    coded1 = -(2.0 / 3.0) * args.dt * g1
    print(f"  executed/coded, own-draw slot: {c0/coded0:.4f} "
          f"+- {se[0]/abs(coded0):.4f}")
    # Second slot from the raw identity (see module docstring).
    lo = args.tmin
    var2 = np.mean(eta[lo:] ** 2)
    rm1 = np.mean(eta[lo:nT - 1] * dY[lo:]) / var2
    rm2 = np.mean(eta[lo:nT - 2] * dY[lo + 1:]) / var2
    c1_impl = rm2 - rm1 / 3.0
    print(f"  second slot via raw identity: {c1_impl:+.5e} vs coded "
          f"{coded1:+.5e} -> factor {c1_impl/coded1:.4f} (O(dt) accuracy); "
          f"slot ratio {rm1/c1_impl:+.4f} (design -2.000)")

    # -- windowed factors: robustness against the g1(t) drift -------------
    g1t = np.interp(np.arange(nT), idx, g1s)
    nw = 4
    edges = np.linspace(n0, n1, nw + 1).astype(int)
    print(f"  windowed own-draw factor ({nw} windows, per-window g1):")
    for w in range(nw):
        sw, sew = fit(edges[w], edges[w + 1])
        g1w = g1t[edges[w]:edges[w + 1]].mean()
        cw = (4.0 / 3.0) * args.dt * g1w
        print(f"    steps {edges[w]:3d}-{edges[w+1]:3d}: g1 {g1w:.5f}, "
              f"factor {sw[0]/cw:.4f} +- {sew[0]/cw:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
