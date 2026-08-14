#!/usr/bin/env python3
"""Covariance spectrum lambda_i(t) of the sudden-expansion DO production run.

Reads the Yi metadata of every DO archive of the sigma=0.0144 total run and
plots the six covariance eigenvalues against time, with the Tikhonov floor
delta(t) = eps_reg * max_q C_qq drawn as the reference: the figure that shows
which modes the regularisation acts on and what rank the ensemble actually
uses (research plan 3.4.3 / Nobile question A1).

Usage:
    python3 fig_cov_spectrum.py --out FIG.pdf
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from do_quicklook import yi_from_fld

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

RUN = Path("/home/isma/nektar_src_full/cases/se/runs/"
           "2026_07_17_do_total_sig0p0144_t1000")
S = 6
EPS_REG = 1e-2

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, default=RUN)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

files = sorted(glob.glob(str(args.run / "output" / "do" / "*.do_*.fld")))
steps = np.array([int(re.search(r"do_(\d+)", f).group(1)) for f in files])
order = np.argsort(steps)
t, lam = [], []
for i in order:
    Y = yi_from_fld(files[i], S)
    lam.append(np.sort(np.linalg.eigvalsh(Y.T @ Y / Y.shape[0]))[::-1])
    t.append(steps[i])
t = np.array(t, dtype=float) / (t[1] - t[0]) if len(t) > 1 else np.array(t)
t = np.array([steps[i] for i in order], dtype=float)
t *= 1.0 / (steps[order][1] - steps[order][0]) if len(t) > 1 else 1.0
lam = np.array(lam)                                   # (T, S)
delta = EPS_REG * lam[:, 0]

print("stationary (t>300) lambda_i / lambda_1:",
      " ".join(f"{v:.2e}" for v in
               (lam[t > 300].mean(0) / lam[t > 300, 0].mean())))
print("modes with lambda_i < delta at final time:",
      int(np.sum(lam[-1] < delta[-1])))

fig, ax = plt.subplots(figsize=(4.1, 2.2), constrained_layout=True)
cols = ["#0072B2"] * 4 + ["#D55E00"] * 2
alphas = [1.0, 0.75, 0.55, 0.4, 1.0, 0.65]
for i in range(S):
    ax.semilogy(t, lam[:, i], color=cols[i], lw=0.8, alpha=alphas[i])

# right-margin labels, pushed apart to a minimum separation in log space
ypos = np.log10(lam[-1].copy())
GAP = 0.45
for i in range(1, S):
    ypos[i] = min(ypos[i], ypos[i - 1] - GAP)
for i in range(S):
    ax.text(1.012, 10.0 ** ypos[i], rf"$\lambda_{i + 1}$", color=cols[i],
            alpha=alphas[i], fontsize=8,
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            clip_on=False)
ax.semilogy(t, delta, "--", color="0.45", lw=0.8)
ax.text(0.02, 0.06, r"$\delta=\varepsilon_{\mathrm{reg}}\max_q C_{qq}$",
        color="0.45", fontsize=8, transform=ax.transAxes, ha="left")
ax.set_xlabel("$t$", fontsize=10)
ax.set_ylabel(r"$\lambda_i(\mathbf{C})$", fontsize=10)
ax.set_xlim(t[0], t[-1])

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out)
fig.savefig(args.out.with_suffix(".png"), dpi=200)
print(f"Saved: {args.out} (+.png)")
