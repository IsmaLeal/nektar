#!/usr/bin/env python3
"""Transition rates of the forced DNS sweep (Re x sigma) from the JED
extraction: one grid.h5 per run, M_grid(t) every 4 time units.

Well level m per Re: |M| of the unforced fixed point of that Re when the
sweep holds its relaxation run (dns_Re<Re>_sig0.0_seed0: 0.1266 / 0.1696 /
0.1945 for Re 90 / 100 / 110), else the median of |M| (t >= tmin) in the run
with the smallest sigma of that Re. The forcing inflates |M| (M is quadratic in v
on the axis: at Re 100 the median |M| is 0.171, 0.211, 0.281 for sigma
0.0072, 0.0144, 0.0288 against 0.1696 at the unforced fixed point), so a
per-run median would raise the threshold with sigma and undercount the
sigma 0.0288 runs; the smallest-sigma median is within 3% of the fixed
point where that is known (Re 100). Transitions follow the hysteresis rule
se.crossings (state flips when M crosses -c*m from +c*m or the reverse;
c = 0.8); lambda = N / (t_end - tmin) with the
Poisson error lambda / sqrt(N). --m-per-run switches to the per-run median.

Runs: every dns_Re<Re>_sig<sig>_seed<seed> of the sweep; the DNS twins of a
DO particle (..._Np<Np>_S<S>p0) are left out, they are not seeds of the
sweep. The nine runs of the grid share seed 45, i.e. the same forcing realisation: at a given
sigma the transition times coincide across Re to within a few time units
(common random numbers), so the three Re series are not independent
samples and their differences carry less statistical weight than N suggests.

Outputs: a table on stdout, the same table as a LaTeX tabular, and a
figure lambda vs 1/sigma^2 with one series per Re (Arrhenius axes).

Usage:
    python3 sweep_rates.py [--sweep DIR] [--out PREFIX] [--c 0.8]
        [--tmin 30]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import se

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

COLORS = {90: "#009E73", 100: "#0072B2", 110: "#D55E00"}
MARKERS = {90: "^", 100: "o", 110: "s"}


def rate_of_run(run: Path, c: float, tmin: float, m: float | None = None):
    d = se.dns(run)
    t, M = d["t"], d["M"]
    w = t >= tmin
    if m is None:
        m = float(np.median(np.abs(M[w])))
    times = se.crossings(t[w], M[w], c * m)
    n = len(times)
    T = float(t[-1] - tmin)
    lam = n / T
    err = lam / np.sqrt(n) if n else np.nan
    return dict(m=m, n=n, T=T, lam=lam, err=err, times=np.array(times),
                t_end=float(t[-1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=se.SWEEP)
    ap.add_argument("--out", type=Path,
                    default=se.SWEEP / "sweep_rates")
    ap.add_argument("--c", type=float, default=se.C)
    ap.add_argument("--tmin", type=float, default=30.0)
    ap.add_argument("--m-per-run", action="store_true",
                    help="well level from each run's own median |M|")
    args = ap.parse_args()

    files = {}
    for run in se.runs("dns", sorted(args.sweep.glob("Re*_sig*"))):
        p = se.params(run)
        if p["twin"]:
            continue                       # a DNS twin of a DO particle, not a seed of the sweep
        files[(p["Re"], float(p["sig"]), p["seed"])] = run
    # well level per Re: the unforced fixed point (sigma = 0 run) when present
    m_of_Re = {}
    if not args.m_per_run:
        for Re in {k[0] for k in files}:
            fp = [k for k in files if k[0] == Re and k[1] == 0.0]
            if fp:
                m_of_Re[Re] = float(abs(se.dns(files[fp[0]])["M"][-1]))
            else:
                k0 = min((k for k in files if k[0] == Re), key=lambda k: k[1])
                m_of_Re[Re] = rate_of_run(files[k0], args.c, args.tmin)["m"]
    rows = []
    for (Re, sigma, seed), f in files.items():
        if sigma == 0.0:
            continue
        r = rate_of_run(f, args.c, args.tmin, m_of_Re.get(Re))
        rows.append(dict(Re=Re, sigma=sigma, seed=seed, **r))
    rows.sort(key=lambda r: (r["Re"], r["sigma"], r["seed"]))

    hdr = (f"{'Re':>4} {'sigma':>7} {'seed':>4} {'t_end':>6} {'m':>7} {'N':>4} "
           f"{'lambda':>9} {'+-':>8} {'tau':>7}")
    print(hdr)
    for r in rows:
        print(f"{r['Re']:4d} {r['sigma']:7.4f} {r['seed']:4d} {r['t_end']:6.0f} "
              f"{r['m']:7.4f} {r['n']:4d} {r['lam']:9.2e} {r['err']:8.1e} "
              f"{1 / r['lam'] if r['n'] else np.nan:7.1f}")

    # LaTeX tabular of the same numbers
    lines = [r"\begin{tabular}{rrrrrrrr}", r"\hline",
             r"$Re$ & $\sigma$ & seed & $t_{\rm end}$ & $m$ & $N$ & "
             r"$\lambda$ & $\tau = 1/\lambda$ \\", r"\hline"]
    for r in rows:
        lines.append(
            f"{r['Re']} & {r['sigma']:.4f} & {r['seed']} & {r['t_end']:.0f} & "
            f"{r['m']:.4f} & {r['n']} & ${r['lam']:.2e}".replace("e-0", r"\times 10^{-")
            + r"}$ & " + f"{1 / r['lam']:.0f}" + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    tex = args.out.with_suffix(".tex")
    tex.write_text("\n".join(lines) + "\n")

    # figure: lambda vs 1/sigma^2, one series per Re, dashed Arrhenius fit
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
    slopes = {}
    for Re in sorted({r["Re"] for r in rows}):
        rr = [r for r in rows if r["Re"] == Re and r["n"] > 0]
        xs = np.array([1 / r["sigma"] ** 2 for r in rr])
        ys = np.array([r["lam"] for r in rr])
        es = np.array([r["err"] for r in rr])
        col = COLORS[Re]
        s45 = np.array([r["seed"] == 45 for r in rr])
        for sel, mfc in ((s45, col), (~s45, "white")):     # open markers: new seeds
            if sel.any():
                ax.errorbar(xs[sel], ys[sel], yerr=es[sel], fmt=MARKERS[Re],
                            color=col, mfc=mfc, ms=3.5, lw=1.0, capsize=2,
                            elinewidth=0.8, ls="none")
        if len(xs) >= 2:
            p = np.polyfit(xs, np.log(ys), 1)
            slopes[Re] = -p[0]
            xf = np.linspace(xs.min(), xs.max(), 20)
            ax.plot(xf, np.exp(np.polyval(p, xf)), "--", color=col, lw=0.8)
        # label at the small-sigma end, where the three series separate
        i0 = int(np.argmax(xs))
        ax.annotate(f"$Re={Re}$", (xs[i0], ys[i0]), xytext=(5, 0),
                    textcoords="offset points", fontsize=8, color=col,
                    ha="left", va="center")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1/\sigma^2$", fontsize=10)
    ax.set_ylabel(r"$\lambda$ (transitions per time unit)", fontsize=9)
    ax.set_xlim(0, 2.2e4)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=200)

    print(f"\nArrhenius slopes -d ln(lambda) / d(1/sigma^2): "
          + ", ".join(f"Re {Re}: {s:.2e}" for Re, s in slopes.items()))
    print(f"Saved: {tex}, {args.out.with_suffix('.pdf')} (+.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
