#!/usr/bin/env python3
"""
Builds the grid of histograms of one quantity.

Rows S, columns Np; cells show DNS and one DO curve per eps (with several forcing seeds pooled)

Uses draw_hist.py.

    python3 grid_hists.py                       q = M, A, Y1, dL and their escape times; S 8 12 16; Np 1000 2000 5000;
                                                eps 1e-4 1e-8 (Y1 is the DO's alone: no DNS curve)
    python3 grid_hists.py --q M --S 8 12 --Np 1000 5000 --eps 0.0001 --out grid.png

Writes cases/se/figures/<pair>/vary_S_Np_eps/hist_<q>.png (or --out, one q) and the .cmd next to it.

Two interpreters: this script reads the h5 files (system python), the drawing runs in the divulg venv (manim, no
h5py), MANIM_PYTHON overrides its path. The data handed from one to the other lives in a temporary directory.
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import se
import draw_hist

HERE = Path(__file__).resolve().parent

QS = ("M", "A", "Y1", "dL", "escape_M", "escape_A", "escape_Y1", "escape_dL")
MANIM_PYTHON = os.environ.get("MANIM_PYTHON", str(Path.home() / "todisimo/despojos/divulg/.venv/bin/python"))
STILL = HERE / "grid_hists_still.py"


def pooled(runs, q, h):
    """The samples of q of several runs, every particle and time pooled (draw_hist.samples)."""
    return np.concatenate([draw_hist.samples(*se.series(r, q.replace("escape_", "")), q, h) for r in runs])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="Re100_sig0.0144")
    ap.add_argument("--S", type=int, nargs="+", default=[8, 12, 16], help="rows")
    ap.add_argument("--Np", type=int, nargs="+", default=[1000, 2000, 5000], help="columns")
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-4, 1e-8], help="one DO curve per eps in every cell")
    ap.add_argument("--q", nargs="+", default=list(QS), choices=QS)
    ap.add_argument("--out", type=Path, help="figure path (one --q); default <pair>/vary_S_Np_eps/hist_<q>.png")
    args = ap.parse_args()
    assert not args.out or len(args.q) == 1, "--out takes one --q"

    pair = se.pair_dir(args.pair)
    RE = se.params(se.runs("do", [pair])[0])["Re"]
    dns = [r for r in se.runs("dns", [pair]) if not se.params(r)["twin"]]
    do = defaultdict(list)                                         # (S, Np, eps) -> its runs, every forcing seed
    for r in se.runs("do", [pair]):
        p = se.params(r)
        do[(p["S"], p["Np"], p["eps"])].append(r)
    missing = [(S, Np, eps) for S in args.S for Np in args.Np for eps in args.eps if (S, Np, eps) not in do]
    assert not missing, f"no run of {missing} in {pair}"

    for q in args.q:
        h = draw_hist.threshold(q, RE)
        bins = draw_hist.escape_bins if q.startswith("escape") else draw_hist.bins
        if q.endswith("Y1"):                                       # a DNS has no Y_1
            dns_dens, dns_edges = np.zeros(0), np.zeros(0)
        else:
            dns_dens, dns_edges = np.histogram(pooled(dns, q, h), bins=bins, density=True)
        shape = (len(args.S), len(args.Np), len(args.eps))
        nb = len(bins) - 1 if np.ndim(bins) else bins
        do_dens, do_edges = np.zeros(shape + (nb,)), np.zeros(shape + (nb + 1,))
        for i, S in enumerate(args.S):
            for j, Np in enumerate(args.Np):
                for k, eps in enumerate(args.eps):
                    do_dens[i, j, k], do_edges[i, j, k] = np.histogram(pooled(do[(S, Np, eps)], q, h), bins=bins, density=True)
                    print(f"  S {S}, Np {Np}, eps {se.eps_str(eps)}: " + ", ".join(r.name for r in do[(S, Np, eps)]))
        out = args.out or se.FIGS / pair.name / "vary_S_Np_eps" / f"hist_{q}.png"
        with tempfile.TemporaryDirectory() as tmp:
            prefix = f"{tmp}/data"
            np.savez(f"{prefix}.npz", q=q, Re=RE, sig=pair.name.split("_sig")[1], h=h if h is not None else np.nan,
                     S=args.S, Np=args.Np, eps=args.eps, dns_edges=dns_edges, dns_dens=dns_dens,
                     do_edges=do_edges, do_dens=do_dens)
            r = subprocess.run([MANIM_PYTHON, str(STILL), prefix, "--out", str(out)], capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"hist_{q}: the drawing failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
        cmd = ["python3", str(Path(__file__).resolve()), "--pair", pair.name, "--q", q, "--S", *map(str, args.S),
               "--Np", *map(str, args.Np), "--eps", *(se.eps_str(e) for e in args.eps)] + (["--out", str(out)] if args.out else [])
        out.with_suffix(".cmd").write_text(f"cd {os.getcwd()}\n{shlex.join(cmd)}\n")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
