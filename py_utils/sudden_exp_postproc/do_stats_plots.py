#!/usr/bin/env python3
"""Statistical plots of a DO run's transition behaviour.

All plots are views of the same per-particle reconstruction
(t, Y_i(t), M_p(t)) provided by do_quicklook.mp_series (axis-cache-aware,
incremental). Available plots:

  traj   Y_1(t) and M_p(t). Default: two stacked panels with a grey sample
         of particles and every hopper coloured. With --particle P: a
         single panel for that particle, Y_1 on the left axis, M on the
         right.
  pdfM   PDF of |M| pooled over particles and time (t >= tmin), with Mbar
         marked. --normalize divides the axis by Mbar.
  pdfT   PDF of escape (residence) times: intervals between consecutive
         transitions of the same particle, pooled. Right-censored
         intervals (residencies not terminated by run end) are excluded;
         the exponential with the sample mean is overlaid.
  yheat  Occupation portrait: x = particle index (sorted by mean Y_i
         unless --no-sort), y = Y_i value, colour = that particle's own
         time-PDF of Y_i over t >= tmin. --mode selects i (1-based).

Usage:
    python3 do_stats_plots.py RUN_DIR [--plot all|traj|pdfM|pdfT|yheat]
        [--out PREFIX] [--tmin 30] [--mbar 0.1935] [--c 0.8]
        [--particle P] [--mode 1] [--bins 60] [--normalize] [--no-sort]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from do_quicklook import mp_series


def transitions_per_particle(t, Mp, mbar, c, tmin):
    """List of transition-time arrays, one per particle (t >= tmin)."""
    thr = c * mbar
    w = t >= tmin
    tw = t[w]
    state = np.where(Mp[w] > thr, 1, np.where(Mp[w] < -thr, -1, 0))
    out = []
    for p in range(Mp.shape[1]):
        sv = state[:, p]
        idx = np.where(sv != 0)[0]
        s = sv[idx]
        times = [tw[idx[a]] for a in range(1, len(s)) if s[a] != s[a - 1]]
        out.append(np.array(times))
    return out


def dns_m_series(dns_dir, xsensor=2.0):
    """(t, M) of a single-realization run from its axis_v.his."""
    from do_quicklook import read_his
    trapz = getattr(np, "trapezoid", None) or np.trapz
    t, pts = read_his(Path(dns_dir) / "output/asymmetry/axis_v.his")
    v = pts[:, :, 1]
    x = 0.1 * np.arange(v.shape[1])
    isen = int(round(xsensor / (x[1] - x[0])))
    M = np.sign(v[:, isen]) * np.sqrt(trapz(v**2, x, axis=1))
    return t, M


def spells_from_trans(trans_list, t0, tend):
    """(durations, event_flags, n_events) from per-trajectory transition
    times; the last spell of each trajectory is right-censored."""
    spells = []
    nev = 0
    for tr in trans_list:
        marks = [t0] + list(tr)
        nev += len(tr)
        for a in range(len(marks) - 1):
            spells.append((marks[a + 1] - marks[a], True))
        spells.append((tend - marks[-1], False))
    dur = np.array([s[0] for s in spells])
    evt = np.array([s[1] for s in spells])
    return dur, evt, nev


def km_curve(dur, evt):
    """Kaplan-Meier survival curve from (durations, event_flags)."""
    order = np.argsort(dur)
    d, e = dur[order], evt[order]
    S = 1.0
    km_t, km_S = [0.0], [1.0]
    at_risk = len(d)
    for k in range(len(d)):
        if e[k]:
            S *= (1 - 1.0 / at_risk)
            km_t.append(d[k])
            km_S.append(S)
        at_risk -= 1
    return km_t, km_S


def plot_traj(t, yi, Mp, args, hoppers, dns_traj=None, trans=None):
    dns_traj = dns_traj or []
    if args.particle is not None:
        p = args.particle
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.plot(t, yi[:, p, 0], lw=0.9, color="tab:blue", label="$Y_1$")
        ax.set_ylabel("$Y_1$", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(t, Mp[:, p], lw=0.9, color="0.2", label="M")
        ax2.axhline(args.mbar, ls=":", color="tab:red", lw=0.8)
        ax2.axhline(-args.mbar, ls=":", color="tab:red", lw=0.8)
        ax2.set_ylabel("$M$")
        ax.set_xlabel("t")
        ax.set_title(f"particle {p}: $Y_1(t)$ and $M(t)$")
        fig.tight_layout()
        return fig

    YLIM = (-0.4, 0.4)
    nrows = 2 if dns_traj else 1
    fig, axs = plt.subplots(nrows, 1, figsize=(11, 4.2 * nrows), squeeze=False)
    axs = axs[:, 0]
    ax = axs[0]
    rng = np.random.default_rng(0)
    sample = rng.choice(Mp.shape[1], size=min(50, Mp.shape[1]), replace=False)
    for p in sample:
        ax.plot(t, Mp[:, p], lw=0.4, color="0.78", zorder=1)
    for p in hoppers[:40]:
        ax.plot(t, Mp[:, p], lw=0.8, zorder=2)

    if trans is not None:
        nhops = [len(tr) for tr in trans]
        p_max = int(np.argmax(nhops))
        ax.plot(t, Mp[:, p_max], lw=2.2, color="black", zorder=3,
               label=f"particle {p_max} ({nhops[p_max]} hops, most of any)")
        ax.legend(fontsize=9, loc="upper right")

    for y in (args.mbar, -args.mbar):
        ax.axhline(y, ls=":", color="tab:red", lw=0.8)
    ax.set_ylabel("$M$")
    ax.set_xlabel("t")
    ax.set_ylim(*YLIM)
    ax.set_title("DO")

    if dns_traj:
        ax_dns = axs[1]
        tmax = t[-1]
        for label, td, Md in dns_traj:
            w = td <= tmax
            ax_dns.plot(td[w], Md[w], lw=0.6)
        for y in (args.mbar, -args.mbar):
            ax_dns.axhline(y, ls=":", color="tab:red", lw=0.8)
        ax_dns.set_ylabel("$M$")
        ax_dns.set_xlabel("t")
        ax_dns.set_xlim(0, tmax)
        ax_dns.set_ylim(*YLIM)
        ax_dns.set_title("DNS")

    fig.tight_layout()
    return fig


STYLES = [dict(ls="-.", marker="s", ms=4, color="tab:red"),
          dict(ls="--", marker="o", ms=4, color="tab:blue"),
          dict(ls=":", marker="^", ms=4, color="tab:green"),
          dict(ls="-", marker="d", ms=4, color="tab:orange")]


def plot_pdfM(sources, args):
    """PDF of |M| for any combination of sources (one plot).

    sources: list of (label, values) with values = pooled |M| samples of
    that source (DO: particles x time; DNS: its time series). --paper
    reproduces the axes of Ducimetiere et al. Fig. 4: x = |M|/Mbar,
    y = probability per bin (their P(|M|) bars sum to 1 with bin width
    ~0.1 in |M|/Mbar; NOT a unit-area density, which is why their peaks
    sit at ~0.1 while a true density peaks near 1). Curve style follows
    their Fig.-4 DNS rendering (markers + broken lines, no bars)."""
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update({"font.size": 13, "axes.grid": True})
    fig, ax = plt.subplots(figsize=(12, 7))
    signed = getattr(args, "signed", False)
    for j, (label, vals) in enumerate(sources):
        sty = STYLES[j % len(STYLES)]
        if args.paper:
            z = vals / args.mbar
            lo = min(-2.2, z.min() - 0.1) if signed else 0.0
            edges = np.arange(lo, max(2.2, z.max() + 0.1), 0.1)
            h, _ = np.histogram(z, bins=edges)
            h = h / z.size
        else:
            scale = args.mbar if args.normalize else 1.0
            h, edges = np.histogram(vals / scale, bins=args.bins,
                                    density=True)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(ctr, h, lw=1.2, label=label, **sty)
    m_sym = r"M" if signed else r"|M|"
    if args.paper:
        ax.set_xlabel(rf"${m_sym}/\bar{{M}}$")
        ax.set_ylabel(rf"$P({m_sym})$")
    else:
        if not args.normalize:
            ax.axvline(args.mbar, ls=":", color="0.4", label=r"$\bar{M}$")
        ax.set_xlabel(rf"${m_sym} / \bar{{M}}$" if args.normalize else rf"${m_sym}$")
        ax.set_ylabel("PDF (unit-area density)")
    if signed:
        ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_title(rf"${m_sym}$ distribution")
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def plot_pdfT(t, Mp, args, trans):
    """Escape-time statistics with censoring handled properly.

    Each particle contributes residence spells: from tmin (or from each
    transition) to the next transition, the final spell right-censored at
    run end. The Kaplan-Meier survival curve uses every spell (complete
    and censored); the censored MLE of the mean residence time is
    (total watched time)/(number of transitions). A naive histogram of the
    few complete inter-transition intervals is biased low (it samples only
    the fastest re-flippers), so it is only drawn as an inset when at
    least 20 complete intervals exist.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for j, (label, trans_list, t0, tend) in enumerate(trans):
        sty = STYLES[j % len(STYLES)]
        dur, evt, nev = spells_from_trans(trans_list, t0, tend)
        if nev == 0:
            continue
        Tmle = dur.sum() / nev
        km_t, km_S = km_curve(dur, evt)
        ax.step(km_t, km_S, where="post", lw=1.4, color=sty["color"],
                label=f"{label}: {nev} events, "
                      rf"$\langle \Delta T\rangle$ = {Tmle:.0f}")
        xx = np.linspace(0, max(km_t) if len(km_t) > 1 else tend, 200)
        ax.plot(xx, np.exp(-xx / Tmle), ls=":", lw=1.0, color=sty["color"])
    ax.set_xlabel(r"escape time $\Delta T$")
    ax.set_ylabel(r"survival $S(\Delta T)$")
    ax.set_ylim(0, 1.02)
    ax.set_title("escape-time survival (censoring-aware; dotted = "
                 "exponential at the censored MLE mean)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_ypdf(t, yi, args):
    """Stacked pooled PDFs of the DO coefficients: one curve per mode,
    pooled over all particles and t >= tmin (Boujo's 'stack the PDFs of
    the Yi'). Bimodality of Y_1 = transitions visible at the ensemble
    level; the other modes stay unimodal."""
    w = t >= args.tmin
    S = yi.shape[2]
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update({"font.size": 13, "axes.grid": True})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(S):
        Y = yi[w, :, i].ravel()
        lim = np.percentile(np.abs(Y), 99.8)
        edges = np.linspace(-lim, lim, args.bins + 1)
        h, _ = np.histogram(Y, bins=edges, density=True)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(ctr, h, lw=2.2 if i == 0 else 1.0,
                label=f"$Y_{i+1}$")
    ax.set_xlabel("$Y_i$")
    ax.set_ylabel("full PDF (particles x time)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)
    ax.set_title(rf"Stochastic coefficient PDFs, $t\geq$ {args.tmin:g}")
    ax.legend(ncol=3, fontsize=12)
    fig.tight_layout()
    return fig


def plot_yheat(t, yi, args):
    i = args.mode - 1
    w = t >= args.tmin
    Y = yi[w, :, i]                       # (Tw, P)
    P = Y.shape[1]
    lim = np.percentile(np.abs(Y), 99.5) * 1.05
    edges = np.linspace(-lim, lim, args.bins + 1)
    H = np.stack([np.histogram(Y[:, p], bins=edges, density=True)[0]
                  for p in range(P)], axis=1)   # (bins, P)
    order = np.argsort(Y.mean(axis=0)) if not args.no_sort else np.arange(P)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pc = ax.pcolormesh(np.arange(P), 0.5 * (edges[1:] + edges[:-1]),
                       H[:, order], cmap="magma", shading="auto")
    fig.colorbar(pc, ax=ax, label="time-PDF")
    ax.set_xlabel("particle (sorted by mean $Y_%d$)" % args.mode
                  if not args.no_sort else "particle index")
    ax.set_ylabel(f"$Y_{args.mode}$")
    ax.set_title(f"per-particle occupation portrait of $Y_{args.mode}$ "
                 f"(t >= {args.tmin:g})")
    fig.tight_layout()
    return fig


def plot_fig4(t, Mp, args):
    """Ducimetiere Fig.-4-style panel: PDF of |M| (raw axis), DO ensemble
    with the DNS overlay(s) from --dns run directories."""
    from do_quicklook import read_his
    trapz = getattr(np, "trapezoid", None) or np.trapz
    w = t >= args.tmin
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(np.abs(Mp[w]).ravel(), bins=args.bins, density=True,
            color="0.45", alpha=0.8, label="DO ensemble")
    for dns in (args.dns or []):
        td, pts = read_his(Path(dns) / "output/asymmetry/axis_v.his")
        v = pts[:, :, 1]
        x = 0.1 * np.arange(v.shape[1])
        Md = np.sign(v[:, 20]) * np.sqrt(trapz(v**2, x, axis=1))
        Md = Md[td >= args.tmin]
        ax.hist(np.abs(Md), bins=args.bins, density=True,
                histtype="step", lw=1.6, label=f"DNS {Path(dns).name[-6:]}")
    ax.axvline(args.mbar, ls=":", color="tab:red", label=r"$\bar{M}$")
    ax.set_xlabel(r"$|M|$")
    ax.set_ylabel("PDF")
    ax.set_title("PDF of |M|: DO vs DNS (Ducimetiere Fig. 4 style)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, nargs="?", default=None,
                    help="DO run directory (optional if --dns given)")
    ap.add_argument("--plot", default="all",
                    choices=["all", "traj", "pdfM", "pdfT", "yheat",
                             "ypdf", "fig4"])
    ap.add_argument("--dns", type=Path, nargs="*", default=None,
                    help="DNS run dir(s) to overlay in pdfM/pdfT/fig4 "
                         "(may also be used without a DO run)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output prefix (default: RUN/output/"
                         "py_utils_results/stats)")
    ap.add_argument("--tmin", type=float, default=30.0)
    ap.add_argument("--mbar", type=float, default=0.1935)
    ap.add_argument("--c", type=float, default=0.8)
    ap.add_argument("--particle", type=int, default=None)
    ap.add_argument("--mode", type=int, default=1)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--signed", action="store_true",
                    help="pdfM: plot M/Mbar (signed) instead of |M|/Mbar")
    ap.add_argument("--paper", action="store_true",
                    help="pdfM: Ducimetiere Fig.-4 axes "
                         "(|M|/Mbar, probability per bin)")
    ap.add_argument("--no-sort", action="store_true")
    ap.add_argument("--pool-dns", action="store_true",
                    help="merge all --dns runs into one pooled DNS curve")
    args = ap.parse_args()
    if args.run is None and not args.dns:
        ap.error("give a DO run and/or --dns run(s)")

    # ---- load sources ----
    t = yi = Mp = None
    trans = hoppers = None
    m_sources = []          # (label, pooled |M| samples) for pdfM
    t_sources = []          # (label, [transition-time arrays], t0, tend)
    dns_traj = []            # (label, td, Md) for traj -- raw, unpooled
    if args.run is not None:
        t, yi, Mp, _ = mp_series(args.run)
        trans = transitions_per_particle(t, Mp, args.mbar, args.c, args.tmin)
        hoppers = [p for p, tr in enumerate(trans) if len(tr)]
        nev = sum(len(tr) for tr in trans)
        print(f"{args.run.name}: {nev} transitions by {len(hoppers)} "
              f"particles (t >= {args.tmin:g})")
        m_vals = Mp[t >= args.tmin] if args.signed else np.abs(Mp[t >= args.tmin])
        m_sources.append((f"DO ({Mp.shape[1]} particles)", m_vals.ravel()))
        t_sources.append(("DO", trans, args.tmin, t[-1]))
    for dns in (args.dns or []):
        td, Md = dns_m_series(dns, args.xsensor if hasattr(args, "xsensor")
                              else 2.0)
        wd = td >= args.tmin
        # transitions of the DNS trajectory with the same protocol
        thr = args.c * args.mbar
        state = 0
        times = []
        for k in range(len(td)):
            if td[k] < args.tmin:
                continue
            s = 1 if Md[k] > thr else (-1 if Md[k] < -thr else 0)
            if s != 0:
                if state != 0 and s != state:
                    times.append(td[k])
                state = s
        label = f"DNS {Path(dns).name.split('_')[-1]}"
        print(f"{Path(dns).name}: {len(times)} transitions "
              f"(t >= {args.tmin:g})")
        m_sources.append((label, Md[wd] if args.signed else np.abs(Md[wd])))
        dns_traj.append((label, td, Md))
        t_sources.append((label, [np.array(times)], args.tmin, td[-1]))
    if args.pool_dns and args.dns and len(args.dns) > 1:
        nd = len(args.dns)
        pooled_m = np.concatenate([v for lab, v in m_sources[-nd:]])
        pooled_tr = [tr for lab, trl, *_ in t_sources[-nd:] for tr in trl]
        tend_min = min(te for *_, te in t_sources[-nd:])
        del m_sources[-nd:], t_sources[-nd:]
        # label with the total simulated time across pooled seeds
        m_sources.append((f"DNS ({int(round(nd*tend_min)):d} time units)",
                          pooled_m))
        t_sources.append((f"DNS ({int(round(nd*tend_min)):d} time units)",
                          pooled_tr, args.tmin, tend_min))

    do_only = {"traj", "yheat", "ypdf", "fig4"}
    todo = (["traj", "pdfM", "pdfT", "yheat", "ypdf"]
            if args.plot == "all" else [args.plot])
    if args.run is None:
        todo = [n for n in todo if n not in do_only] or \
            ap.error(f"--plot {args.plot} needs a DO run")

    base = args.run if args.run is not None else args.dns[0]
    prefix = args.out or (base / "output/py_utils_results/stats")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name in todo:
        fig = {"traj": lambda: plot_traj(t, yi, Mp, args, hoppers, dns_traj, trans),
               "pdfM": lambda: plot_pdfM(m_sources, args),
               "pdfT": lambda: plot_pdfT(t, Mp, args, t_sources),
               "yheat": lambda: plot_yheat(t, yi, args),
               "ypdf": lambda: plot_ypdf(t, yi, args),
               "fig4": lambda: plot_fig4(t, Mp, args)}[name]()
        out = Path(f"{prefix}_{name}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
