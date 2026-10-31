#!/usr/bin/env python3
"""Per-particle transition rate of the DO convergence runs (Re 100, sigma
0.0144) from the JED extraction, against the forced DNS at the same
(Re, sigma).

measures.h5 holds, every time unit, the per-particle A_p(t): the filter's
A_p, or (python extraction) A of the mean, A of every mode and the particle
coefficients Yi, from which se.do_all rebuilds
    A_p(t) = A[v_mean(t)] + sum_i Y_pi(t) A[v_i(t)].
Transitions per particle follow the hysteresis rule se.crossings on A with
the threshold c * A_fp (c = 0.5; A_fp = se.a_fp(100) = 0.701, A of the
unforced Re 100 fixed point, the threshold of every A figure); the ensemble
rate is
    lambda_DO = (total transitions) / (Np (t_end - tmin)),
Poisson error lambda / sqrt(N). The DNS rate counts the same crossings on
the A of dns_Re100_sig0.0144_seed45 with the same c, tmin.

Usage:
    python3 do_rates.py [--sweep DIR] [--out PREFIX] [--c 0.5] [--tmin 100]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=se.RUNS)
    ap.add_argument("--out", type=Path,
                    default=se.RUNS / "do_rates")
    ap.add_argument("--c", type=float, default=0.5, help="threshold as a fraction of A_fp")
    ap.add_argument("--tmin", type=float, default=100.0)
    args = ap.parse_args()
    h = args.c * se.a_fp(100)

    ref = args.sweep / "Re100_sig0.0144/dns_Re100_sig0.0144_seed45"
    d = se.dns(ref)
    t, A = d["t"], d["A"]
    w = t >= args.tmin
    n_dns = len(se.crossings(t[w], A[w], h))
    T_dns = float(t[-1] - args.tmin)
    dns = dict(n=n_dns, lam=n_dns / T_dns,
               err=n_dns / T_dns / np.sqrt(n_dns) if n_dns else np.nan,
               medA=float(np.median(np.abs(A[w]))))
    print(f"DNS Re100 sigma0.0144: N={dns['n']} lambda={dns['lam']:.2e} "
          f"+- {dns['err']:.1e}  tau={1 / dns['lam']:.0f}  med|A|={dns['medA']:.4f}")

    rows = []
    for run in se.runs("do", sorted(args.sweep.glob("Re100_sig*"))):
        prm = se.params(run)
        d = se.do_all(run)
        t, Ap = d["t"], d["A"]
        w = t >= args.tmin
        n = np.array([len(tc) for tc in se.crossings(t[w], Ap[w], h)])
        N = int(n.sum())
        T = float(t[-1] - args.tmin)
        lam = N / (prm["Np"] * T)
        rows.append(dict(**prm, t_end=float(t[-1]), N=N, lam=lam,
                         err=lam / np.sqrt(N) if N else np.nan,
                         ratio=lam / dns["lam"],
                         f0=float((n == 0).mean()),
                         medA=float(np.median(np.abs(Ap[w])))))
    rows.sort(key=lambda r: (r["S"], r["Np"], r["eps"]))

    print(f"{'S':>3} {'Np':>5} {'eps':>7} {'t_end':>6} {'N':>6} {'lambda':>9} "
          f"{'+-':>8} {'DO/DNS':>7} {'f(0 tr)':>8} {'med|A|':>7}")
    for r in rows:
        print(f"{r['S']:3d} {r['Np']:5d} {se.eps_str(r['eps']):>7} {r['t_end']:6.0f} "
              f"{r['N']:6d} {r['lam']:9.2e} {r['err']:8.1e} {r['ratio']:7.2f} "
              f"{r['f0']:8.3f} {r['medA']:7.4f}")

    lines = [r"\begin{tabular}{rrrrrrrr}", r"\hline",
             r"$S$ & $N_p$ & $\epsilon$ & $t_{\rm end}$ & $N$ & $\lambda_{\rm DO}$ & "
             r"$\lambda_{\rm DO}/\lambda_{\rm DNS}$ & median $|A|$ \\", r"\hline"]
    for r in rows:
        lines.append(f"{r['S']} & {r['Np']} & $10^{{{np.log10(r['eps']):.0f}}}$ & {r['t_end']:.0f} & "
                     f"{r['N']} & ${r['lam']:.2e}".replace("e-0", r"\times 10^{-")
                     + r"}$ & " + f"{r['ratio']:.2f} & {r['medA']:.3f}" + r" \\")
    lines += [r"\hline",
              f"\\multicolumn{{8}}{{l}}{{DNS: $N={dns['n']}$, "
              f"$\\lambda={dns['lam']:.2e}".replace("e-0", r"\times 10^{-")
              + f"}}$, median $|A|$ = {dns['medA']:.3f}}} \\\\",
              r"\end{tabular}"]
    tex = args.out.with_suffix(".tex")
    tex.write_text("\n".join(lines) + "\n")
    print(f"Saved: {tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
