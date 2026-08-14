#!/usr/bin/env python3
"""House-style figures for the two 2026-07-28 validation measurements, 3.3.

(1) mc_closure.pdf -- matched-ensemble MC check (small-amplitude
    discriminant; every one of the 24 initial states is evolved twice,
    once inside the DO ensemble and once as an independent full-solver
    run): (a) DO vs Monte-Carlo covariance spectrum at T=2;
    (b) parity plot of each realisation's fluctuation amplitude,
    full-solver run vs DO reconstruction of the same realisation.
(2) rank_invariance.pdf -- rank-count invariance: relative deviation of
    the three ensemble invariants; parallel-vs-parallel sits at 1e-12,
    serial-vs-parallel at the elliptic-solver-tolerance level (the two
    serial pairs coincide to plotting accuracy).

Usage:
    python3 fig_closure_checks.py --outdir DIR
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from convergence_orders import FC_DEFAULT, load_dat
from do_quicklook import yi_from_fld
from rankinv_check import pair, state
from vortex_mc_check import CELL_AREA as MC_AREA
from vortex_mc_check import NP, S, grid

RUNS = Path("/home/isma/nektar_src_full/cases/se/runs")
VRUNS = Path("/home/isma/nektar_src_full/cases/vortex/runs")
SE_AREA = (45.0 / 900) * (3.0 / 120)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})


def fig_mc(outdir: Path, fieldconvert: str):
    do_run = VRUNS / "2026_07_28_mc_do_sa"
    fleet = VRUNS / "2026_07_28_mc_fleet_sa"
    arch = do_run / "output" / "do" / "casefile.do_001000.fld"
    d = load_dat(grid(do_run / "casefile.xml", arch,
                      do_run / "mcgrid_001000.dat", fieldconvert), 6 + 3 * S)
    do_mean = np.concatenate([d[:, 3], d[:, 4]])
    do_modes = np.column_stack(
        [np.concatenate([d[:, 6 + 2 * i], d[:, 7 + 2 * i]]) for i in range(S)])
    Y = yi_from_fld(arch, S)
    mc = []
    for p in range(NP):
        run = fleet / f"p{p:03d}"
        g = load_dat(grid(run / "casefile.xml",
                          run / "output" / "chks" / "casefile_40.chk",
                          run / "mcgrid_final.dat", fieldconvert), 6)
        mc.append(np.concatenate([g[:, 3], g[:, 4]]))
    mc = np.column_stack(mc)
    mc_mean = mc.mean(axis=1)
    W = mc - mc_mean[:, None]
    nrm = lambda f: np.sqrt(np.sum(f ** 2) * MC_AREA)
    fluct = np.sqrt(np.trace((W.T @ W) * MC_AREA) / NP)
    mean_rel = nrm(do_mean - mc_mean) / nrm(mc_mean)
    c_do = np.sort(np.linalg.eigvalsh(Y.T @ Y / NP))[::-1]
    c_mc = np.sort(np.linalg.eigvalsh((W.T @ W) * MC_AREA / NP))[::-1][:S]
    dist = np.array([nrm(do_mean + do_modes @ Y[p] - mc[:, p])
                     for p in range(NP)]) / fluct

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.5, 2.35),
                                   constrained_layout=True)
    idx = np.arange(1, S + 1)
    axa.semilogy(idx, c_do, "o-", color="#0072B2", lw=1.0, ms=3.5,
                 label="DO")
    axa.semilogy(idx, c_mc, "s-", color="0.45", lw=1.0, ms=3.5,
                 label="Monte Carlo")
    axa.legend(fontsize=8, frameon=False)
    axa.set_xticks(idx)
    axa.set_xlabel("mode $i$", fontsize=10)
    axa.set_ylabel(r"$\lambda_i(\mathbf{C})$", fontsize=10)
    axa.text(-0.12, 1.07, "(a)", transform=axa.transAxes, va="top",
             ha="right", fontsize=11)

    a_do = np.array([nrm(do_modes @ Y[p]) for p in range(NP)])
    a_mc = np.array([nrm(W[:, p]) for p in range(NP)])
    lo = 0.95 * min(a_do.min(), a_mc.min())
    hi = 1.05 * max(a_do.max(), a_mc.max())
    axb.plot([lo, hi], [lo, hi], "--", lw=0.7, color="0.45")
    axb.plot(a_mc, a_do, "o", color="#0072B2", ms=3.5)
    axb.set_xlim(lo, hi)
    axb.set_ylim(lo, hi)
    axb.set_aspect("equal")
    # the two axes carry the same quantity, so they carry the same ticks
    ticks = [t for t in axb.get_xticks() if lo <= t <= hi]
    axb.set_xticks(ticks)
    axb.set_yticks(ticks)
    axb.set_xlim(lo, hi)
    axb.set_ylim(lo, hi)
    axb.set_xlabel(r"$\Vert\mathbf{u}'_p\Vert$, Monte Carlo", fontsize=10)
    axb.set_ylabel(r"$\Vert\mathbf{u}'_p\Vert$, DO", fontsize=10)
    axb.text(-0.02, 1.07, "(b)", transform=axb.transAxes, va="top",
             ha="right", fontsize=11)
    fig.savefig(outdir / "mc_closure.pdf")
    fig.savefig(outdir / "mc_closure.png", dpi=200)
    print(f"mc_closure: mean_rel {mean_rel:.2e}, "
          f"rms {np.sqrt((dist**2).mean()):.2e}")


def fig_rank(outdir: Path, fieldconvert: str):
    names = ("np1", "np2", "np3", "np4")
    runs = {n: RUNS / f"2026_07_28_rankinv_{n}" for n in names}
    st = {n: state(r, 200, fieldconvert) for n, r in runs.items()}
    mref = np.sqrt(np.sum(st["np1"][0] ** 2) * SE_AREA)
    G = (st["np1"][1].T @ st["np1"][1]) * SE_AREA
    Y = st["np1"][2]
    pref = np.sqrt(np.trace(Y @ G @ Y.T) / Y.shape[0])
    allpairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    rel = {}
    for a, b in allpairs:
        dm, dp, dc = pair(st[a], st[b])
        rel[a, b] = np.array([dm / mref, dp / pref, dc])

    amp = {}
    for n in names:
        G = (st[n][1].T @ st[n][1]) * SE_AREA
        Y = st[n][2]
        amp[n] = np.sqrt(np.einsum("pi,ij,pj->p", Y, G, Y))
    abar = amp["np1"].mean()
    npart = amp["np1"].size
    curves = {(a, b): np.clip(np.sort(np.abs(amp[a] - amp[b]))[::-1] / abar,
                              1e-17, None) for a, b in allpairs}

    fig, ax = plt.subplots(figsize=(3.1, 2.2), constrained_layout=True)
    dists = {p: np.abs(amp[p[0]] - amp[p[1]]) / abar for p in allpairs}
    order = [("np1", "np2"), ("np1", "np3"), ("np1", "np4"),
             ("np2", "np3"), ("np2", "np4"), ("np3", "np4")]
    pos = [0, 1.25, 2.5, 4.4, 5.65, 6.9]
    for p, xp in zip(order, pos):
        col = "#D55E00" if p[0] == "np1" else "#0072B2"
        ax.boxplot([np.clip(dists[p], 1e-17, None)], positions=[xp],
                   widths=0.55, whis=(5, 95), showfliers=False,
                   boxprops=dict(color=col, lw=1.0),
                   whiskerprops=dict(color=col, lw=0.8),
                   capprops=dict(color=col, lw=0.8),
                   medianprops=dict(color=col, lw=1.4))
    ax.set_yscale("log")
    ax.annotate("serial vs parallel", (1.25, 6e-7), fontsize=8,
                color="#D55E00", ha="center", va="bottom")
    ax.annotate("between parallel runs", (5.5, 3e-12), fontsize=8,
                color="#0072B2", ha="center", va="bottom")
    ax.set_ylim(1e-16, 1e-5)
    ax.set_xticks(pos)
    ax.set_xticklabels(["1 vs 2", "1 vs 3", "1 vs 4",
                        "2 vs 3", "2 vs 4", "3 vs 4"], fontsize=8)
    ax.set_ylabel(r"relative deviation of $\Vert\mathbf{u}'_p\Vert$",
                  fontsize=10)
    ax.set_xlim(-0.7, 7.6)
    fig.savefig(outdir / "rank_invariance.pdf")
    fig.savefig(outdir / "rank_invariance.png", dpi=200)
    print("rank_invariance invariants:",
          {f"{a}-{b}": [f"{v:.1e}" for v in r] for (a, b), r in rel.items()})
    print("amplitude deviation medians:",
          {f"{a[2]}v{b[2]}": f"{np.median(np.abs(amp[a]-amp[b]))/abar:.1e}"
           for a, b in allpairs})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--fieldconvert", default=FC_DEFAULT)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_mc(args.outdir, args.fieldconvert)
    fig_rank(args.outdir, args.fieldconvert)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
