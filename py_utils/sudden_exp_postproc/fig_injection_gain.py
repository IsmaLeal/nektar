#!/usr/bin/env python3
"""Executed-vs-coded injection by lag, research plan 3.3 test (2).

Reads the white-noise audit runs (2026_07_30_frc_audit_np{1,2,4}) and
plots, (a), the magnitude of the measured draw-increment correlation
R_ell = E[zeta^(n) dY^(n+ell)] / E[zeta^2] against the coded reference
r_ell over lags 0..6, and, (b), their ratio G_ell over the lags the
reference dominates. The reference is parameter-free: the design
injection weights of the IMEX-BDF2 extrapolation ((4/3)*dt*g1 on the
step's own draw, -(2/3)*dt*g1 on the next) propagated by the bare
integrator bookkeeping, whose increments decay by 1/3 per lag; the
estimator's resolving power on the ratio shrinks with the reference,
threefold per lag, which is where the ratio panel stops.
A wrong amplitude shifts G uniformly; a wrong template or sub-step
placement bends the pattern.

Before correlating with the draw at step n, the increment is
residualised on everything that dominates its variance and is
independent of that draw (the pre-draw state and the other draws
entering the same increment): no expectation changes, only the
estimator variance drops.

Usage:
    python3 fig_injection_gain.py --out FIG.pdf
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from do_quicklook import yi_from_fld
from forcing_gain_check import eta_from_fld, gain, FC_DEFAULT

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

RUNS = Path("/home/isma/nektar_src_full/cases/se/runs")
BLUE = "#0072B2"
DT = 0.01
TMIN = 20
LAG_PROFILE = 6
LAG_RATIO = 4


def series(run: Path):
    files = sorted(run.glob("output/do/*.do_*.fld"),
                   key=lambda f: int(re.search(r"do_(\d+)", f.name).group(1)))
    eta = np.stack([eta_from_fld(f) for f in files])
    Y = np.stack([yi_from_fld(f, 1)[:, 0] for f in files])
    return files, eta, Y


def measure(run: Path, fieldconvert: str):
    files, eta, Y = series(run)
    nT = len(files)
    dY = Y[1:] - Y[:-1]

    idx = np.unique(np.linspace(TMIN, nT - 1, 21).astype(int))
    g1 = float(np.mean([gain(run, files[i], f"{i:06d}", fieldconvert,
                             False)[0] for i in idx]))
    w0, w1 = (4.0 / 3.0) * DT * g1, -(2.0 / 3.0) * DT * g1
    y = np.zeros(LAG_PROFILE + 3)
    for n in range(1, LAG_PROFILE + 3):
        y[n] = (4.0 / 3.0) * y[n - 1] - (1.0 / 3.0) * y[n - 2] \
            + (w0 if n == 1 else 0.0) + (w1 if n == 2 else 0.0)
    r = np.diff(y)[:LAG_PROFILE + 1]

    var = np.mean(eta[TMIN:] ** 2)
    R, Rse = [], []
    for ell in range(LAG_PROFILE + 1):
        n_hi = nT - 1 - ell
        d = dY[TMIN + ell:].ravel()
        offs = [-1] + list(range(1, ell + 1))
        cols = [Y[TMIN:n_hi].ravel(), Y[TMIN - 1:n_hi - 1].ravel()]
        cols += [eta[TMIN + j:n_hi + j].ravel() for j in offs]
        Z = np.column_stack(cols)
        d = d - Z @ np.linalg.lstsq(Z, d, rcond=None)[0]
        x = eta[TMIN:n_hi].ravel() * d
        R.append(np.mean(x) / var)
        Rse.append(np.std(x) / (np.sqrt(x.size) * var))
    return np.array(R), np.array(Rse), r, g1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fieldconvert", default=FC_DEFAULT)
    args = ap.parse_args()

    legs = [("np1", "o", "full", -0.12), ("np2", "s", "none", 0.0),
            ("np4", "^", "none", 0.12)]
    data = {n: measure(RUNS / f"2026_07_30_frc_audit_{n}", args.fieldconvert)
            for n, *_ in legs}
    for n, (R, Rse, r, g1) in data.items():
        print(f"{n}: g1 = {g1:.5f}, G by lag = "
              + " ".join(f"{Ri/ri:.4f}" for Ri, ri in zip(R, r)))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.2, 2.2),
                                   constrained_layout=True)
    lags = np.arange(LAG_PROFILE + 1)

    # (a) profile magnitudes, np1 (the three runs are indistinguishable)
    R, Rse, r, _ = data["np1"]
    hl, = axa.semilogy(lags, np.abs(r), "--", color="0.45", lw=0.8)
    hd = axa.errorbar(lags, np.abs(R), yerr=Rse, color=BLUE, ls="none",
                      marker="o", ms=3.5, elinewidth=0.7, capsize=1.5)
    axa.legend([hd, hl], ["measured", r"impulse response $r_\ell$"],
               frameon=False, fontsize=8, loc="upper right")
    axa.set_xlabel(r"lag $\ell$ (steps after the draw)", fontsize=10)
    axa.set_ylabel(r"$|R_\ell|$", fontsize=10)
    axa.set_xticks(lags)
    axa.text(0.04, 0.08, "(a)", transform=axa.transAxes, fontsize=10)

    # (b) ratio over the lags the reference dominates, all rank counts
    rl = np.arange(LAG_RATIO + 1)
    axb.axhline(1.0, color="0.45", lw=0.8, ls="--")
    hs = []
    for name, mk, fill, off in legs:
        R, Rse, r, _ = data[name]
        hs.append(axb.errorbar(
            rl + off, R[:LAG_RATIO + 1] / r[:LAG_RATIO + 1],
            yerr=Rse[:LAG_RATIO + 1] / np.abs(r[:LAG_RATIO + 1]),
            color=BLUE, ls="none", marker=mk, ms=3.5,
            mfc=(BLUE if fill == "full" else "none"),
            elinewidth=0.7, capsize=1.5))
    axb.legend(hs, ["1 rank", "2 ranks", "4 ranks"], frameon=False,
               fontsize=8, loc="upper left")
    axb.set_xlabel(r"lag $\ell$ (steps after the draw)", fontsize=10)
    axb.set_ylabel(r"$G_\ell = R_\ell / r_\ell$", fontsize=10)
    axb.set_xticks(rl)
    axb.set_ylim(0.97, 1.03)
    axb.text(0.04, 0.08, "(b)", transform=axb.transAxes, fontsize=10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print(f"Saved: {args.out} (+.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
