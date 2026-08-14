#!/usr/bin/env python3
"""Two-panel figure for the research plan, Section 2.2:
(a) the double-well potential V(A) = -lambda A^2/2 - mu A^4/4 of the
    stochastically forced Stuart-Landau reduction, drawn with the
    coefficients of Ducimetiere et al. (PRF 9, 053905): lambda = 5.984,
    mu = -0.02962 -- wells at +/- Abar = sqrt(-lambda/mu), barrier
    height Delta V annotated;
(b) the asymmetry observable M(t) from a clean forced DNS
    (cases/se/runs/2026_07_15_dns_clean_seed42), showing exponentially
    long residences in the two attractors punctuated by fast switches.

Usage:
    python3 fig_se_doublewell_mt.py --his RUN/output/asymmetry/axis_v.his \
        --out figures/se_doublewell_mt.pdf
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

ap = argparse.ArgumentParser()
ap.add_argument("--his", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

# Stuart-Landau coefficients of Ducimetiere, Boujo & Gallaire (2024)
LAM, MU = 5.984, -0.02962
ABAR = np.sqrt(-LAM / MU)
V = lambda a: -LAM * a**2 / 2 - MU * a**4 / 4
DV = V(0.0) - V(ABAR)

spec = importlib.util.spec_from_file_location(
    "mh", Path(__file__).resolve().parent / "mbar_from_his.py")
mh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mh)
x, blocks = mh.load_his(args.his)
t, M = mh.m_of_t(x, blocks)
keep = np.concatenate(([True], np.diff(t) > 0))
t, M = t[keep], M[keep]

fig, (axa, axb) = plt.subplots(
    1, 2, figsize=(6.5, 2.2), width_ratios=[1, 2.2],
    constrained_layout=True)
fig.get_layout_engine().set(wspace=0.08)

# (a) double well
a = np.linspace(-1.35 * ABAR, 1.35 * ABAR, 400)
axa.plot(a, V(a), color="black", lw=1.2)
axa.annotate("", xy=(0, 0), xytext=(0, V(ABAR)),
             arrowprops=dict(arrowstyle="<->", lw=0.8))
axa.text(0.06 * ABAR, V(ABAR) / 2, r"$\Delta V$", fontsize=10)
axa.set_xticks([-ABAR, 0, ABAR])
axa.set_xticklabels([r"$X_{\min}$", r"$X_{\max}$", r"$X_{\min}$"])
axa.set_yticks([])
axa.set_xlabel("$X$", fontsize=12)
axa.set_ylabel("$V(X)$", fontsize=12)
# offset points, not an axes fraction: (b) is 2.2x wider than (a) and
# carries y-tick labels, which a fixed fractional offset runs into
axa.annotate("(a)", xy=(0, 1), xycoords="axes fraction", xytext=(-2, 4),
             textcoords="offset points", va="bottom", ha="right", fontsize=11)

# (b) DNS observable
axb.plot(t, M, color="#0072B2", lw=0.7)
axb.axhline(0.0, color="0.75", lw=0.6)
axb.set_xlim(t[0], t[-1])
axb.set_xlabel("$t$", fontsize=12)
axb.set_ylabel("$M(t)$", fontsize=12)
axb.annotate("(b)", xy=(0, 1), xycoords="axes fraction", xytext=(-2, 4),
             textcoords="offset points", va="bottom", ha="right", fontsize=11)

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out)
fig.savefig(args.out.with_suffix(".png"), dpi=200)
print(f"Saved: {args.out} (+.png)  Abar={ABAR:.2f}  DeltaV={DV:.1f}  "
      f"t=[{t[0]:.0f},{t[-1]:.0f}]")
