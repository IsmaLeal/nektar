#!/usr/bin/env python3
"""
Every histogram where one thing varies. A line = the DO runs of one pair that differ in one of S, Np, eps and
share the rest; for every line, one figure per quantity, with the DNS of the pair drawn too (every seed pooled,
the twins left out). The repeats of a DO run with other forcing seeds (_frc<N>) are always pooled into its curve,
never a line of their own. The figure itself is draw_hist.py's.

    python3 vary_hists.py                            every line of every pair, q = M, A, escape_A, escape_M
    python3 vary_hists.py --list                     the lines and their runs, nothing drawn
    python3 vary_hists.py --line vary_S_Np1000_eps1e-4 --q A
                                                     one line, one quantity

Writes cases/se/figures/<pair>/vary_<p>_<the rest>/hist_<q>.png and the .cmd next to it:
vary_S_Np1000_eps1e-4/hist_A.png = S varies, Np 1000 and eps 1e-4 fixed. The runs are those with their
measures.h5 under the sweep root (se.runs): the absTol runs, moved out of it, are not drawn.
"""
import argparse
import os
import shlex
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import se
import draw_hist
from figsave import save

KEYS = ("S", "Np", "eps")
QS = ("M", "A", "escape_A", "escape_M")


def lines(pair):
    """{folder name: the curves of the line, in increasing order of what varies} of one pair; a curve = a DO run
    and its repeats with other forcing seeds, pooled"""
    curves = defaultdict(list)                                    # (S, Np, eps, seed) -> its runs, every forcing seed
    for r in se.runs("do", [pair]):
        p = se.params(r)
        curves[(p["S"], p["Np"], p["eps"], p["seed"])].append(r)
    par = {c: dict(zip(KEYS + ("seed",), c)) for c in curves}
    seeds = {p["seed"] for p in par.values()}
    out = {}
    for v in KEYS:
        rest = [k for k in KEYS if k != v] + ["seed"]
        groups = defaultdict(list)
        for c, p in par.items():
            groups[tuple(p[k] for k in rest)].append(c)
        for key, cs in groups.items():
            if len(cs) < 2:
                continue
            f = dict(zip(rest, key))
            name = [f"vary_{v}"] + [f"{k}{se.eps_str(f[k]) if k == 'eps' else f[k]}" for k in KEYS if k != v]
            name += [f"seed{f['seed']}"] if len(seeds) > 1 else []
            out["_".join(name)] = [curves[c] for c in sorted(cs, key=lambda c: par[c][v])]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs="+", help="pairs (default: every pair of the sweep with DO runs)")
    ap.add_argument("--line", nargs="+", help="only these lines (folder names, as --list prints them)")
    ap.add_argument("--q", nargs="+", default=list(QS), choices=QS, help="quantities (default all four)")
    ap.add_argument("--list", action="store_true", help="print the lines and their runs, draw nothing")
    args = ap.parse_args()

    pairs = sorted({r.parent for r in se.runs("do", args.pair)})
    for pair in pairs:
        dns = [r for r in se.runs("dns", [pair]) if not se.params(r)["twin"]]
        for name, cs in lines(pair).items():
            if args.line and name not in args.line:
                continue
            print(f"{pair.name}/{name}: " + ", ".join("+".join(se.short_eps(r.name[len(f"do_{pair.name}_"):]) for r in rs) for rs in cs))
            if args.list:
                continue
            curves = ([(dns, None)] if dns else []) + [(rs, None) for rs in cs]
            for q in args.q:
                plt.rcParams.update(draw_hist.RC)
                fig, ax = plt.subplots(figsize=draw_hist.SIZE, constrained_layout=True)
                draw_hist.draw(ax, curves, q=q)
                out = save(fig, se.FIGS / pair.name / name / f"hist_{q}.png", dpi=draw_hist.DPI)
                plt.close(fig)
                cmd = ["python3", str(Path(__file__).resolve()), "--pair", pair.name, "--line", name, "--q", q]
                out.with_suffix(".cmd").write_text(f"cd {os.getcwd()}\n{shlex.join(cmd)}\n")
                print(f"  {out}")


if __name__ == "__main__":
    main()
