#!/usr/bin/env python3
"""
Projects the (unit-norm) adjoint mode q^dagger (qdag_channel.fld) on the DO modes.
Mode field values stored in measures.h5 only every 1000 t.u.

Usage:
    python3 qdag_projection.py [--qdag cases/se/templates/qdag_channel.npz]
        [--pair SWEEP/Re100_sig0.0144] [--runs RUN ...] [--out cases/se/figures/Re100_sig0.0144/qdag_proj.pdf]
"""
import re
import sys
import h5py
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent # resolve() makes the path absolute
sys.path.insert(0, str(HERE.parents[1] / "cases" / "se" / "templates"))
from extract_lib import ip
from figsave import save

REPO = HERE.parents[1]  # repo home (nektar_src_full)
SWEEP = REPO / "cases/se/runs/2026_09_13_jed_sweep"
FIGS = REPO / "cases/se/figures"

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--qdag", type=Path, default=REPO / "cases/se/templates/qdag_channel.npz", help="q^dagger")
ap.add_argument("--pair", type=Path, default=SWEEP / "Re100_sig0.0144")
ap.add_argument("--runs", nargs="+", default=["do_Re100_sig0.0144_S6_Np1000_seed45_eps0.0001",
                                              "do_Re100_sig0.0144_S12_Np1000_seed45_eps0.0001"])
ap.add_argument("--out", type=Path, default=FIGS / "Re100_sig0.0144/qdag_proj.pdf")
args = ap.parse_args()

z = np.load(args.qdag)
ku, kv = ("u", "v") if "u" in z else tuple(str(k) for k in z["fields"][:2])     # e.g. g1_u, g1_v
gu, gv = z[ku].astype(float), z[kv].astype(float)
gn = np.sqrt(ip(gu, gv, gu, gv))
gu, gv = gu / gn, gv / gn

fig, axes = plt.subplots(len(args.runs), 1, figsize=(6.3, 2.3 * len(args.runs)), sharex=True,
                         constrained_layout=True, squeeze=False)
for ax, run in zip(axes[:, 0], args.runs):
    with h5py.File(args.pair / run / "output/extract/measures.h5", "r") as h:
        t = h["grid/t"][:]
        mu, mv = h["grid/mode_u"][:], h["grid/mode_v"][:]            # (T, S, NY, NX)
    T, S = mu.shape[:2]
    c = np.array([[ip(gu, gv, mu[k, i], mv[k, i]) for i in range(S)] for k in range(T)])   # (T, S)
    tot = np.sqrt((c ** 2).sum(axis=1))
    Np, eps = re.search(r"_Np(\d+)_seed\d+(?:_eps([0-9.]+))?", run).groups()   # no _eps in the name was for eps=0.01
    cfg = rf"$S = {S}$, $N_p = {Np}$, $\varepsilon = 10^{{{np.log10(float(eps or 0.01)):g}}}$"
    print(f"{run}")
    print("   t     " + " ".join(f"{'c'+str(i+1):>7s}" for i in range(S)) + "   |sum c_i u_i|")
    for k in range(T):
        print(f"{t[k]:6.0f}  " + " ".join(f"{v:7.3f}" for v in c[k]) + f"   {tot[k]:.3f}")
    for i in range(S):
        ax.plot(t, np.abs(c[:, i]), "-", lw=0.6, label=f"$|c_{{{i+1}}}|$")
    ax.plot(t, tot, "-", lw=1.6, color="k", label=r"$\|\sum_i c_i u_i\|$")
    ax.set_ylim(0, 1.2); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(r"$c_i=\langle q^\dagger,\, u_i\rangle$")
    ax.set_title(cfg, loc="right", fontsize=9)
    ax.legend(fontsize=6, frameon=False, ncol=min(S + 1, 7), loc="upper left", columnspacing=0.8, handlelength=1.2)
axes[-1, 0].set_xlabel("$t$")
out = save(fig, args.out, also_png=True)
print("Saved:", out)
