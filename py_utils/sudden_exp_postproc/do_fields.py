#!/usr/bin/env python3
"""
The still of phi and mode K of a DO run (do_fields_still.py, manim house style) for every DO run of the sweep,
or for the runs named:

    python3 do_fields.py                                  every DO run with its measures.h5
    python3 do_fields.py S16_Np5000_seed45_eps1e-4      one run (names as in panel.py: the part after
                                                          do_<pair>_, or the whole directory name)
    python3 do_fields.py --mode 2 --t 3000 RUN            mode 2 at t = 3000 instead of mode 1 at the last instant

Writes cases/se/figures/<pair>/<run>/phi_u<K>.png (phi_u<K>_t<t>.png for an instant other than the last) and the
.cmd next to it. The mode is taken at one of the instants where measures.h5 keeps the modes on the whole plane
(grid/t); the axes below the channels hold <u_K, phi>^2 and <u_K - phi, u_K - phi>^2 at those same instants,
computed on the plane (legacy.ip, <phi, phi> = 1).
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

import legacy
import se

MANIM_PYTHON = os.environ.get("MANIM_PYTHON", sys.executable)
STILL = Path(__file__).resolve().parent / "do_fields_still.py"


def data(run, out, t=None, mode=1):
    """Writes <out>.npz for do_fields_still.py; returns the instant of the mode."""
    phi = np.load(se.TPL / f"phi_Re{se.params(run)['Re']}.npz")
    pu, pv = phi["phi_u"], phi["phi_v"]
    with h5py.File(se.planes(run), "r") as f:
        tg = f["grid/t"][:]
        k = len(tg) - 1 if t is None else int(np.flatnonzero(tg == t)[0])
        a, g = np.empty(len(tg)), np.empty(len(tg))
        for j in range(len(tg)):
            uj, vj = f["grid/mode_u"][j, mode - 1].astype(float), f["grid/mode_v"][j, mode - 1].astype(float)
            a[j], g[j] = legacy.ip(uj, vj, pu, pv), legacy.ip(uj, vj, uj, vj)
            if j == k:
                u, v = uj, vj
    np.savez(f"{out}.npz", x=se.x, y=se.y, phi_u=pu, phi_v=pv, u=u, v=v, inside=se.INSIDE,
             t=tg[k], k=mode, run=Path(run).name, eps=se.params(run)['eps'], ts=tg, a=a, g=g)
    return tg[k], tg[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="DO runs; none = every DO run of the sweep")
    ap.add_argument("--pair", default="Re100_sig0.0144", help="the pair of the runs named (default Re100_sig0.0144)")
    ap.add_argument("--mode", type=int, default=1, help="mode K, 1 = the first")
    ap.add_argument("--t", type=float, help="instant of the mode (one of grid/t of measures.h5); default the last")
    args = ap.parse_args()

    if args.runs:
        runs = [r for a in args.runs for r in se.find(se.pair_dir(args.pair), a)[0]]
    else:
        runs = se.runs("do")
    for run in runs:
        assert se.params(run)["kind"] == "do", f"{run.name}: not a DO run"
        with tempfile.TemporaryDirectory() as tmp:
            prefix = f"{tmp}/data"
            t, t_last = data(run, prefix, args.t, args.mode)
            name = f"phi_u{args.mode}" + ("" if t == t_last else f"_t{t:g}") + ".png"
            out = se.FIGS / run.parent.name / run.name / name
            r = subprocess.run([MANIM_PYTHON, str(STILL), prefix, "--out", str(out)], capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"{run.name}: the drawing failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
        cmd = ["python3", str(Path(__file__).resolve()), run.name, "--pair", run.parent.name, "--mode", str(args.mode)]
        cmd += ["--t", f"{args.t:g}"] if args.t is not None else []
        out.with_suffix(".cmd").write_text(f"cd {os.getcwd()}\n{shlex.join(cmd)}\n")
        print(out)


if __name__ == "__main__":
    main()
