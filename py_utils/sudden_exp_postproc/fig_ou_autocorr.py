#!/usr/bin/env python3
"""Autocorrelation of the archived OU forcing states, research plan 3.3.

Reads DOVelocityCorrectionScheme_ForcingEta_hex from every DO archive of a
production run (default: the sigma=0.0144 total run, 1001 archives at unit
spacing, 1000 realisations), computes the ensemble autocorrelation of the
per-realisation zeta series, and fits the correlation time against the
coded tau. The executed update of Equation (ou-exact) is exact, so the
autocorrelation must fall on exp(-lag/tau).

Usage:
    python3 fig_ou_autocorr.py --out FIG.pdf
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

_ETA_TAG = re.compile(
    r"<DOVelocityCorrectionScheme_ForcingEta_hex>([0-9a-fA-F]+)<")

RUN = Path("/home/isma/nektar_src_full/cases/se/runs/"
           "2026_07_17_do_total_sig0p0144_t1000")

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, default=RUN)
ap.add_argument("--tau", type=float, default=25.0)
ap.add_argument("--maxlag", type=int, default=100)
ap.add_argument("--fitlag", type=int, default=50)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()


def eta_from_fld(fld):
    fld = Path(fld)
    info = fld / "Info.xml" if fld.is_dir() else fld
    m = _ETA_TAG.search(info.read_text(errors="ignore"))
    if m is None:
        raise ValueError(f"no ForcingEta_hex metadata in {info}")
    return np.frombuffer(bytes.fromhex(m.group(1)), dtype="<f8")


files = sorted(glob.glob(str(args.run / "output" / "do" / "*.do_*.fld")))
steps = np.array([int(re.search(r"do_(\d+)", f).group(1)) for f in files])
order = np.argsort(steps)
eta = np.stack([eta_from_fld(files[i]) for i in order])   # (T, Np)
print(f"eta series: {eta.shape[0]} samples x {eta.shape[1]} realisations, "
      f"std {eta.std():.4g}")

lags = np.arange(args.maxlag + 1)
acf = np.array([np.mean(eta[: eta.shape[0] - k] * eta[k:]) for k in lags])
acf /= acf[0]

fit = slice(0, args.fitlag + 1)
A = np.vstack([lags[fit], np.ones(args.fitlag + 1)]).T
sol, res, *_ = np.linalg.lstsq(A, np.log(acf[fit]), rcond=None)
tau_fit = -1.0 / sol[0]
ss_tot = np.sum((np.log(acf[fit]) - np.log(acf[fit]).mean()) ** 2)
r2 = 1.0 - (res[0] / ss_tot if res.size else 0.0)

fig, ax = plt.subplots(figsize=(3.1, 2.1), constrained_layout=True)
ax.semilogy(lags, np.exp(-lags / args.tau), "--", color="0.45", lw=0.8,
            label=rf"$e^{{-\ell/{args.tau:g}}}$")
ax.semilogy(lags, acf, color="#0072B2", lw=1.0, label=r"archived $\zeta$")
ax.set_xlabel(r"lag $\ell$ (steps after the draw)", fontsize=10)
ax.set_ylabel(r"autocorrelation of $\zeta$", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.text(0.05, 0.08, rf"$\tau_{{\mathrm{{fit}}}}={tau_fit:.2f}$",
        transform=ax.transAxes, ha="left", fontsize=9)

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out)
fig.savefig(args.out.with_suffix(".png"), dpi=200)
print(f"Saved: {args.out} (+.png)  tau_fit={tau_fit:.2f} "
      f"(coded {args.tau:g})  R2={r2:.4f}  fit over lags 0..{args.fitlag}")
