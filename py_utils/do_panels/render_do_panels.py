#!/usr/bin/env python3
"""The do_panels video of a DO run, on any domain: FieldConvert interppoints
of every archive onto a box grid, cached in
RUN/output/py_utils_results/grid_cache/gd_STEP.dat (same header and row
layout as the grid_cache of the earlier runs, e.g.
2026_07_15_do_clean_dnspod_sig0p0144_t250), then assemble_do_h5.py into
RUN/output/out_grid.h5, then the data of the movie
(RUN/output/py_utils_results/do_panels_data.npz, _u.npy, _modes.npy) and
do_panels_movie.py (manim, the divulg venv) into the .mp4: the mean of u,
modes 1 to 4 and the realisation of particle 0 on a 3 by 3 grid, the colour
keys, the clock and the standard deviation of every coefficient in time.

The run dir holds geometry.xml, casefile.xml and the archives
output/do/*.do_*.fld (DOArchive, Format Field). The grid is
--box NX,NY,XMIN,XMAX,YMIN,YMAX:
    the sudden-expansion channel    --box 901,121,-5,40,-1.5,1.5
    the cow (cases/cow)             --box 367,201,-3,8,-3,3
Points outside the fluid (a body inside the box) come out as 0.

FieldConvert module: interppoints (eCreatePts) -- must be called with NO
positional xml/fld inputs (only fromxml=/fromfld= config keys), else it
aborts with "should not use xml or fld inputs" (utilities/FieldConvert/
FieldConvert.cpp:731-750). Binary and environment: archives.FIELDCONVERT,
archives.fc_env (the ASan build, libasan preloaded).

Incremental: a complete gd_*.dat is kept, so after the simulation advances
only the new archives are interpolated.

    python3 render_do_panels.py RUN_DIR --box NX,NY,XMIN,XMAX,YMIN,YMAX [--out FILE] [--seconds SEC] [--quality low|high] [--jobs N]
The default is the preview (low quality); the final render is --quality high.
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

import archives

HERE = Path(__file__).resolve().parent
MANIM_PYTHON = sys.executable


def box_arg(s):
    """NX,NY,XMIN,XMAX,YMIN,YMAX -> nx, ny, the box argument of FieldConvert interppoints."""
    nx, ny, x0, x1, y0, y1 = s.split(",")
    nx, ny = int(nx), int(ny)
    return nx, ny, f"box={nx},{ny},1,{float(x0):g},{float(x1):g},{float(y0):g},{float(y1):g},0,0"


def interp_one(run: Path, fld: Path, box: str, out_dat: Path, expected_lines: int) -> bool:
    """The box grid of one archive (archives.interp); True when the .dat is complete."""
    archives.interp(run / "geometry.xml", fld, box, out_dat)
    if not out_dat.exists():
        return False
    with open(out_dat) as f:
        n = sum(1 for _ in f)
    return n == expected_lines


def build_grid_cache(run: Path, box: str, expected_lines: int, jobs: int) -> int:
    """every archive of run/output/do onto the box grid, jobs FieldConvert at a time; complete .dat files are kept"""
    cache = run / "output/py_utils_results/grid_cache"
    cache.mkdir(parents=True, exist_ok=True)
    flds = sorted((run / "output/do").glob("*.do_*.fld"))
    todo = []
    done = 0
    for fld in flds:
        step = fld.name.split("do_")[1].split(".")[0]
        out_dat = cache / f"gd_{step}.dat"
        if out_dat.exists():
            with open(out_dat) as f:
                if sum(1 for _ in f) == expected_lines:
                    done += 1
                    continue
            out_dat.unlink()
        todo.append((fld, out_dat))
    t0 = time.time()
    if todo:
        archives.progress(0, len(todo), t0, "archives extracted")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        jobs_ = {pool.submit(interp_one, run, fld, box, out_dat, expected_lines): (fld, out_dat) for fld, out_dat in todo}
        for k, fut in enumerate(as_completed(jobs_), 1):
            fld, out_dat = jobs_[fut]
            if fut.result():
                done += 1
            else:
                print(f"\n  FAILED: {fld.name}")
                if out_dat.exists():
                    out_dat.unlink()
            archives.progress(k, len(todo), t0, "archives extracted")
    print(f"grid_cache: {done}/{len(flds)} archives OK ({len(todo)} new, {jobs} at a time)")
    return done


def movie_data(h5: Path, prefix: Path) -> None:
    """What do_panels_movie.py reads, from the h5: the axes, the times, the mean of u (T, ny, nx) and the modes of u
    (S, T, ny, nx) as float32, the standard deviation of every coefficient (S, T), the body mask (points where every
    field is exactly 0 at every time: outside the fluid), the coefficients of particle 0 (S, T) and the colour scales."""
    with h5py.File(h5, "r") as f:
        x, y = f["x"][0, :], f["y"][:, 0]
        t = f["t"][:]
        u = np.ascontiguousarray(np.transpose(f["u"][:], (2, 0, 1)), dtype=np.float32)          # (T, ny, nx)
        modes = np.ascontiguousarray(np.transpose(f["mode_u"][:], (0, 3, 1, 2)), dtype=np.float32)  # (S, T, ny, nx)
        yi = f["yi"][:]                                                                          # (P, S, T)
    sigma, y0 = yi.std(axis=0), yi[0]                                                            # (S, T) each
    body = (u == 0).all(0) & (modes == 0).all((0, 1))
    np.savez(str(prefix) + ".npz", x=x, y=y, t=t, sigma=sigma, y0=y0, body=body,
             u_pos=float(u.max()), u_neg=float(max(0.05, -u.min())), m_scale=float(np.percentile(np.abs(modes), 99.5)))
    np.save(str(prefix) + "_u.npy", u)
    np.save(str(prefix) + "_modes.npy", modes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--box", required=True,
                    help="NX,NY,XMIN,XMAX,YMIN,YMAX of the grid (the channel: 901,121,-5,40,-1.5,1.5)")
    ap.add_argument("--out", type=Path, default=None, help="the movie (default RUN/output/py_utils_results/do_panels_u.mp4)")
    ap.add_argument("--seconds", type=float, default=None, help="video seconds for the whole run (default: one per time unit)")
    ap.add_argument("--quality", default="low", choices=["low", "high"],
                    help="low: the preview, 854 by 480 at 15 frames per second (default); high: 1920 by 1080 at 60")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="FieldConvert extractions run at a time (default: half the CPUs)")
    args = ap.parse_args()
    run = args.run.resolve()
    nx, ny, box = box_arg(args.box)
    expected_lines = nx * ny + 3  # Variables line + blank + Zone line + data

    n = build_grid_cache(run, box, expected_lines, args.jobs)
    if n == 0:
        print("no archives interpolated; aborting")
        return 1

    h5 = run / "output/out_grid.h5"
    rc = subprocess.run(
        ["python3", str(HERE / "assemble_do_h5.py"), str(run), "--out", str(h5),
         "--nx", str(nx), "--ny", str(ny)]
    ).returncode
    if rc != 0:
        return rc

    results = run / "output/py_utils_results"
    out = args.out or results / "do_panels_u.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = results / "do_panels_data"
    movie_data(h5, prefix)
    cmd = [str(MANIM_PYTHON), str(HERE / "do_panels_movie.py"), str(prefix), "--out", str(out), "--quality", args.quality]
    if args.seconds is not None:
        cmd += ["--seconds", str(args.seconds)]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
