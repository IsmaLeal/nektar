#!/usr/bin/env python3
"""Per-particle transition rate of the DO convergence runs (Re 100, sigma
0.0144) from the JED extraction, against the forced DNS at the same
(Re, sigma).

measures.h5 holds, every time unit, the centreline v of the mean and of every
mode and the particle coefficients Yi, so the per-particle measure
    M_p(t) = M[v_mean(t) + sum_i Y_pi(t) v_i(t)]
is rebuilt on the same 801 axis points and with the same trapezoid rule as
the DNS (se.do_all). Transitions per particle follow the hysteresis rule
se.crossings with the DNS well level m of Re 100
(sweep_rates.py: median |M| of the sigma 0.0072 DNS, 0.1706) and c = 0.8;
the ensemble rate is
    lambda_DO = (total transitions) / (Np (t_end - tmin)),
Poisson error lambda / sqrt(N). The DNS rate comes from sweep_rates.rate_of_run
on the grid.h5 of dns_Re100_sig0.0144_seed45 with the same m, c, tmin.

Usage:
    python3 do_rates.py [--sweep DIR] [--out PREFIX] [--c 0.8] [--tmin 30]
        [--m 0.1706]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import se
from sweep_rates import rate_of_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=se.SWEEP)
    ap.add_argument("--out", type=Path,
                    default=se.SWEEP / "do_rates")
    ap.add_argument("--c", type=float, default=se.C)
    ap.add_argument("--tmin", type=float, default=30.0)
    ap.add_argument("--m", type=float, default=se.M_WELL, help="well level of Re 100 (se.M_WELL)")
    args = ap.parse_args()

    ref = args.sweep / "Re100_sig0.0144/dns_Re100_sig0.0144_seed45"
    dns = rate_of_run(ref, args.c, args.tmin, args.m)
    d = se.dns(ref)
    dns["medM"] = float(np.median(np.abs(d["M"][d["t"] >= args.tmin])))
    print(f"DNS Re100 sigma0.0144: N={dns['n']} lambda={dns['lam']:.2e} "
          f"+- {dns['err']:.1e}  tau={1 / dns['lam']:.0f}  med|M|={dns['medM']:.4f}")

    rows = []
    for run in se.runs("do", sorted(args.sweep.glob("Re100_sig*"))):
        prm = se.params(run)
        d = se.do_all(run)
        t, Mp = d["t"], d["M"]
        w = t >= args.tmin
        n = np.array([len(tc) for tc in se.crossings(t[w], Mp[w], args.c * args.m)])
        N = int(n.sum())
        T = float(t[-1] - args.tmin)
        lam = N / (prm["Np"] * T)
        rows.append(dict(**prm, t_end=float(t[-1]), N=N, lam=lam,
                         err=lam / np.sqrt(N) if N else np.nan,
                         ratio=lam / dns["lam"],
                         f0=float((n == 0).mean()),
                         medM=float(np.median(np.abs(Mp[w])))))
    rows.sort(key=lambda r: (r["S"], r["Np"], r["eps"]))

    print(f"{'S':>3} {'Np':>5} {'eps':>7} {'t_end':>6} {'N':>6} {'lambda':>9} "
          f"{'+-':>8} {'DO/DNS':>7} {'f(0 tr)':>8} {'med|M|':>7}")
    for r in rows:
        print(f"{r['S']:3d} {r['Np']:5d} {se.eps_str(r['eps']):>7} {r['t_end']:6.0f} "
              f"{r['N']:6d} {r['lam']:9.2e} {r['err']:8.1e} {r['ratio']:7.2f} "
              f"{r['f0']:8.3f} {r['medM']:7.4f}")

    lines = [r"\begin{tabular}{rrrrrrrr}", r"\hline",
             r"$S$ & $N_p$ & $\epsilon$ & $t_{\rm end}$ & $N$ & $\lambda_{\rm DO}$ & "
             r"$\lambda_{\rm DO}/\lambda_{\rm DNS}$ & median $|M|$ \\", r"\hline"]
    for r in rows:
        lines.append(f"{r['S']} & {r['Np']} & $10^{{{np.log10(r['eps']):.0f}}}$ & {r['t_end']:.0f} & "
                     f"{r['N']} & ${r['lam']:.2e}".replace("e-0", r"\times 10^{-")
                     + r"}$ & " + f"{r['ratio']:.2f} & {r['medM']:.3f}" + r" \\")
    lines += [r"\hline",
              f"\\multicolumn{{8}}{{l}}{{DNS: $N={dns['n']}$, "
              f"$\\lambda={dns['lam']:.2e}".replace("e-0", r"\times 10^{-")
              + f"}}$, median $|M|$ = {dns['medM']:.3f}}} \\\\",
              r"\end{tabular}"]
    tex = args.out.with_suffix(".tex")
    tex.write_text("\n".join(lines) + "\n")
    print(f"Saved: {tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
