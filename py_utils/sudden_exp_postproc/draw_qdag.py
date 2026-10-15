"""
Projects the (unit-norm) adjoint mode q^dagger (qdag_channel.fld) on the DO modes
of one run (panel.py --panel qdag). Mode field values stored in measures.h5 only
every 1000 t.u.

q^dagger on the plane comes from plane_from_fld.py; qdag=FILE.npz takes another one.

    python3 panel.py --sharex --panel qdag S6_Np1000_seed45_eps0.0001 --panel qdag S16_Np1000_seed45_eps0.0001 \
        --out cases/se/figures/Re100_sig0.0144/qdag_proj.pdf
"""
import re
import h5py
import numpy as np

import se

RC = {"text.usetex": True, "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9}
SIZE, DPI = (6.3, 2.3), 200                 # size of one panel, dpi of the png


def draw(ax, curves, qdag=se.REPO / "cases/se/templates/qdag_channel.npz"):
    """curves: one DO run, [([run directory], None)] (se.find)."""
    assert len(curves) == 1 and len(curves[0][0]) == 1, "qdag: one DO run per panel"
    run = curves[0][0][0]
    z = np.load(qdag)
    ku, kv = ("u", "v") if "u" in z else tuple(str(k) for k in z["fields"][:2])     # e.g. g1_u, g1_v
    gu, gv = z[ku].astype(float), z[kv].astype(float)
    gn = np.sqrt(se.ip(gu, gv, gu, gv))
    gu, gv = gu / gn, gv / gn

    with h5py.File(se.h5(run), "r") as h:
        t = h["grid/t"][:]
        mu, mv = h["grid/mode_u"][:], h["grid/mode_v"][:]            # (T, S, NY, NX)
    T, S = mu.shape[:2]
    c = np.array([[se.ip(gu, gv, mu[k, i], mv[k, i]) for i in range(S)] for k in range(T)])   # (T, S)
    tot = np.sqrt((c ** 2).sum(axis=1))
    Np, eps = re.search(r"_Np(\d+)_seed\d+(?:_eps([0-9.]+))?", run.name).groups()   # no _eps in the name was for eps=0.01
    cfg = rf"$S = {S}$, $N_p = {Np}$, $\varepsilon = 10^{{{np.log10(float(eps or 0.01)):g}}}$"
    print(f"{run.name}")
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
    ax.set_xlabel("$t$")
