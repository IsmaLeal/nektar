#!/usr/bin/env python3
"""Extract linear growth rates lambda(Re) from the 2026_07_17 eig campaign.

DNS kick runs: M(t) from output/asymmetry/axis_v.his (Ducimetiere Eq. 25);
lambda is the least-squares slope of ln |M| over the fit window.

DO runs: Yi from the DOArchive snapshots (phase_space_video loader);
lambda is half the slope of ln var(Y1). The slope of ln var(Y2) gives the
second eigenvalue's rate estimate for free.

Windows: start after the kick / mode-alignment transient; growth runs are
capped before nonlinear saturation (|M| <= 0.02, std(Y1) <= 1e-2).

Usage:
    python3 eig_from_runs.py [--plots DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

RUNS = Path("/home/isma/nektar_src_full/cases/se/runs")

# (dirname, kind, Re, t0)  --  t0 skips the transient
CASES = [
    ("2026_07_17_eig_dns_Re70",   "dns", 70.0,  50.0),
    ("2026_07_17_eig_dns_Re90",   "dns", 90.0,  50.0),
    ("2026_07_17_eig_dns_Re110",  "dns", 110.0, 50.0),
    ("2026_07_17_eig_do_Re70",    "do",  70.0,  40.0),
    ("2026_07_17_eig_do_Re79p3",  "do",  79.3,  40.0),
    ("2026_07_17_eig_do_Re90",    "do",  90.0,  40.0),
    # merged t=0-150 tree: 07_17 run (t<=100) + 2026_07_23 DORestartFile
    # continuation (seam bit-exact, max|dYi|=3.3e-20; see its README).
    # t0=60 past the alignment transient (t0=70 -> +0.015583, R^2=0.99998)
    ("2026_07_23_eig_do_Re110_total", "do", 110.0, 60.0),
    # kick DNS on the 2026_07_23 rebuilt Re=70 baseflow (relaxed from the
    # Re=79.3 symmetric state); same kick recipe as the 07_17 campaign,
    # stopped early at t~140 (fit already converged, R^2=1.0 at t0=80)
    ("2026_07_23_eig_dns_Re70", "dns", 70.0, 80.0),
    # growth side: the 07_12 kick experiment from the unstable Re=100 base;
    # t0=80 skips the kick transient, M_CAP ends the window at t~208
    # (lam=+0.011900, R^2=1.00000, Arnoldi +0.0118956)
    ("2026_07_12_m03_Re100_kick_pos", "dns", 100.0, 80.0),
    # kick DNS at Re=79.3 (2026_07_23, from the 07_12 symmetric base):
    # lam=-0.001868, R^2=1.00000, Arnoldi -0.00196867 (-5.1%)
    ("2026_07_23_eig_dns_Re79p3", "dns", 79.3, 80.0),
    # unforced S=2 eig-protocol DO at Re=100 (2026_07_23), mean started on
    # the unstable symmetric base; t0=70 past the alignment transient
    # (t0=40/50/60/70 -> 0.01144/0.01174/0.011855/0.011905, R^2 rising)
    ("2026_07_23_eig_do_Re100", "do", 100.0, 70.0),
]

# converged Arnoldi growth rates from the .evl files, where available.
# 70/90/110 from the 2026_07_23_m03_Re*_arnoldi_direct runs (build_fast,
# residuals < 1.3e-6); 79.3/81.3/100 from the 07_12-07_13 campaign.
ARNOLDI = {70.0: -0.012641, 79.3: -0.00196867, 81.3: -0.000123534,
           90.0: 0.0064899, 100.0: 0.0118956, 110.0: 0.015632}

# Per-run fit pngs are stamped with the PRODUCTION date (they are cheap
# regenerated artifacts); when two cases share kind+Re, the run date is
# appended so the files stay distinguishable.
TODAY = time.strftime("%Y_%m_%d")
_RESTS: dict[str, int] = {}
for _n, *_ in CASES:
    _RESTS[_n[11:]] = _RESTS.get(_n[11:], 0) + 1


def out_png(name: str) -> str:
    rest = name[11:]
    if _RESTS.get(rest, 0) > 1:
        return f"{TODAY}_{rest}_run{name[5:10].replace('_', '')}"
    return f"{TODAY}_{rest}"

M_CAP = 0.02      # DNS linearity cap on |M|
Y_CAP = 1e-2      # DO linearity cap on std(Y1)


def _import(name: str, rel: str):
    mod_path = Path(__file__).resolve().parent / rel
    spec = importlib.util.spec_from_file_location(name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_loglinear(t, y, t0, cap=None):
    """Slope of ln y over [t0, first crossing of cap]; returns
    (slope, r2, window)."""
    m = (t >= t0) & (y > 0)
    if cap is not None:
        over = (y > cap) & (t >= t0)
        if over.any():
            m &= t < t[over][0]
    t_w, ln_y = t[m], np.log(y[m])
    A = np.vstack([t_w, np.ones_like(t_w)]).T
    (slope, b), res, *_ = np.linalg.lstsq(A, ln_y, rcond=None)
    ss_tot = np.sum((ln_y - ln_y.mean()) ** 2)
    r2 = 1.0 - (res[0] / ss_tot if len(res) and ss_tot > 0 else 0.0)
    return slope, r2, (t_w[0], t_w[-1]), (t_w, ln_y, slope, b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", type=Path,
                    default=RUNS / "2026_07_17_eig_fits")
    args = ap.parse_args()
    args.plots.mkdir(exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "mathtext.fontset": "cm", "font.size": 9})

    mhis = _import("mbar_from_his", "mbar_from_his.py")
    psv = _import("phase_space_video", "phase_space_video.py")

    rows = []
    for name, kind, re_, t0 in CASES:
        run = RUNS / name
        fig, ax = plt.subplots(figsize=(7, 4))
        if kind == "dns":
            x, blocks = mhis.load_his(run / "output/asymmetry/axis_v.his")
            t, M = mhis.m_of_t(x, blocks)
            keep = np.concatenate(([True], np.diff(t) > 0))
            t, M = t[keep], np.abs(M[keep])
            lam, r2, win, (tw, ln_y, s, b) = fit_loglinear(t, M, t0, M_CAP)
            ax.plot(t, np.log(np.where(M > 0, M, np.nan)), lw=0.8,
                    label="ln |M|")
            rows.append((name, re_, "DNS", lam, r2, win))
        else:
            yi, t = psv._load_run(run)          # (P, S, T)
            v1 = yi[:, 0, :].var(axis=0)
            v2 = yi[:, 1, :].var(axis=0)
            cap = Y_CAP ** 2
            s1, r2, win, (tw, ln_y, s, b) = fit_loglinear(t, v1, t0, cap)
            lam = 0.5 * s1
            s2, r2_2, win2, _ = fit_loglinear(t, v2, t0, None)
            ax.plot(t, np.log(v1), lw=0.8, label="ln var(Y1)")
            ax.plot(t, np.log(v2), lw=0.8, color="gray",
                    label="ln var(Y2)")
            rows.append((name, re_, "DO", lam, r2, win))
            rows.append((name, re_, "DO(m2)", 0.5 * s2, r2_2, win2))
        ax.plot(tw, s * tw + b, "r--", lw=1.2,
                label=f"fit slope {s:+.5f}")
        ax.axvspan(*win, color="orange", alpha=0.15)
        ax.set_xlabel("t")
        ax.legend(fontsize=9)
        ax.set_title(name, fontsize=10)
        fig.tight_layout()
        fig.savefig(args.plots / f"{out_png(name)}.png", dpi=150)
        plt.close(fig)

    print(f"\n{'Re':>6} {'method':>8} {'lambda':>12} {'R^2':>8} "
          f"{'window':>16} {'Arnoldi':>12} {'diff %':>8}")
    for name, re_, meth, lam, r2, win in rows:
        ref = ARNOLDI.get(re_)
        ref_s = f"{ref:+.6f}" if ref is not None and meth != "DO(m2)" else "-"
        diff = (f"{100*(lam-ref)/abs(ref):+.1f}"
                if ref is not None and meth != "DO(m2)" else "-")
        print(f"{re_:>6g} {meth:>8} {lam:>+12.6f} {r2:>8.4f} "
              f"[{win[0]:>6.1f},{win[1]:>6.1f}] {ref_s:>12} {diff:>8}")
    # lambda(Re) summary plot, replicating the original 2026_07_17 style
    # (the original generator lived in a deleted job tmp).
    # Marks: filled square/diamond with a white separation edge, and the
    # reference method as a larger OPEN ring drawn on top, so coincident
    # points nest without occlusion. Colors: Okabe-Ito blue/vermillion
    # (CVD-checked), black reserved for the reference.
    # built at its final printed width: 0.55\textwidth of 495.08pt
    fig, ax = plt.subplots(figsize=(3.77, 2.83))
    res = sorted(ARNOLDI)
    dns = [(re_, lam) for name, re_, meth, lam, *_ in rows if meth == "DNS"]
    do = [(re_, lam) for name, re_, meth, lam, *_ in rows if meth == "DO"]
    ax.plot(*zip(*sorted(dns)), "s", color="#0072B2", ms=5,
            ls="-", lw=0.8, label="DNS", zorder=3)
    ax.plot(res, [ARNOLDI[r] for r in res], "D", color="black",
            ms=5, mec="white", mew=0.4, ls="-", lw=0.8, zorder=5,
            label="Arnoldi")
    ax.plot(*zip(*sorted(do)), "o", color="#D55E00", ms=3.5, mec="white", mew=0.5,
            ls="-", lw=0.8, label="DO", zorder=6)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel(r"Re", fontsize=10)
    ax.set_ylabel(r"Growth rate $\lambda$", fontsize=10)
    ax.tick_params(labelsize=8)
    # ax.set_title(r"Eigenvalue vs Re cross-validation")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(args.plots / "lambda_vs_Re.pdf", dpi=150)
    plt.close(fig)

    print(f"\nplots in {args.plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
